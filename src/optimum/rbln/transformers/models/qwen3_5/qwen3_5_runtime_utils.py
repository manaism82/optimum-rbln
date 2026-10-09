# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator

import torch

from ...modeling_outputs import RBLNDecoderOnlyOutput
from ..decoderonly.decoderonly_runtime_utils import RBLNRuntimeModel


def _as_row(value, what: str) -> int:
    try:
        return operator.index(value)
    except TypeError as e:
        raise ValueError(f"{what} must be an integer, got {value!r}.") from e


def plan_linear_state_windows(
    prefix_cached_len: int,
    query_length: int,
    prefill_chunk_size: int,
    batch_idx: int | None,
    batch_size: int,
    snapshot_slots: int,
    state_restore_row: int | None = None,
    state_capture: dict[int, int] | None = None,
) -> list[tuple[int | None, int | None, bool]]:
    """Plan which GatedDeltaNet state row each prefill window reads and writes.

    Returns one `(read_row, write_row, carry)` per prefill window (`ceil(query_length / prefill_chunk_size)` of
    them). The window reads its carried state from `read_row`, multiplied by ones (`carry`) or zeros (fresh
    sequence), and writes its final state to `write_row`. The rows index the linear-state caches: rows
    `[0, batch_size)` are live requests, rows `[batch_size, batch_size + snapshot_slots)` are prefix snapshots.

    - No prefix (`prefix_cached_len == 0`): window 0 reads `batch_idx` with zero masks; every window writes
      `batch_idx` and later windows carry it.
    - Restore (`prefix_cached_len == b > 0`, a multiple of `prefill_chunk_size`): the prompt resumes at `b`. The
      first window reads the snapshot `state_restore_row` (the state after exactly `b` tokens) with ones masks.
    - Capture (`state_capture[c] = row`): the window that ENDS at absolute position `c` writes `row` instead of
      `batch_idx`; the next window reads `row` and writes `batch_idx` again. `c` must be a multiple of
      `prefill_chunk_size` inside the prompt and before the last window, so the last window always writes
      `batch_idx` (the row decode continues from).
    """
    if snapshot_slots == 0 and (state_restore_row is not None or state_capture):
        raise ValueError(
            "state_restore_row / state_capture need linear-state snapshot rows, but this model was compiled with "
            "linear_state_snapshot_slots=0."
        )

    if prefix_cached_len % prefill_chunk_size != 0:
        raise ValueError(
            f"The cached prefix length ({prefix_cached_len}) must be a multiple of prefill_chunk_size "
            f"({prefill_chunk_size}) for the Qwen3.5 hybrid model."
        )
    if prefix_cached_len > 0 and state_restore_row is None:
        raise ValueError(
            f"Prefill starts at cache position {prefix_cached_len}, but no linear-state snapshot to resume from was "
            "given. The Qwen3.5 hybrid model can only skip a cached prefix when `state_restore_row` names the "
            "snapshot row that holds the GatedDeltaNet state after exactly that prefix"
            + (" (compile with linear_state_snapshot_slots > 0)." if snapshot_slots == 0 else ".")
        )
    if prefix_cached_len == 0 and state_restore_row is not None:
        raise ValueError("state_restore_row is set, but the prefill starts at cache position 0 (no cached prefix).")

    snapshot_rows = range(batch_size, batch_size + snapshot_slots)
    if snapshot_slots > 0:
        # With snapshot rows in the same cache, a bad live row would silently overwrite a snapshot.
        batch_idx = None if batch_idx is None else _as_row(batch_idx, "batch_idx")
        if batch_idx is None or not 0 <= batch_idx < batch_size:
            raise ValueError(
                f"batch_idx must be in [0, {batch_size}) with linear-state snapshot rows, got {batch_idx}."
            )
    if state_restore_row is not None:
        state_restore_row = _as_row(state_restore_row, "state_restore_row")
        if state_restore_row not in snapshot_rows:
            raise ValueError(
                f"state_restore_row must be a snapshot row in [{snapshot_rows.start}, {snapshot_rows.stop}), "
                f"got {state_restore_row}."
            )

    capture: dict[int, int] = {}
    for boundary, row in (state_capture or {}).items():
        boundary, row = _as_row(boundary, "state_capture boundary"), _as_row(row, "state_capture row")
        if row not in snapshot_rows:
            raise ValueError(
                f"state_capture rows must be snapshot rows in [{snapshot_rows.start}, {snapshot_rows.stop}), "
                f"got {row} for boundary {boundary}."
            )
        if row in capture.values():
            raise ValueError(f"state_capture writes snapshot row {row} more than once.")
        # The capturing window must be a full window that is not the prompt's last one (that one always writes
        # batch_idx, the row decode continues from): P < c < P + query_length.
        if boundary % prefill_chunk_size != 0 or not prefix_cached_len < boundary < prefix_cached_len + query_length:
            raise ValueError(
                f"state_capture boundary {boundary} must be a multiple of prefill_chunk_size ({prefill_chunk_size}) "
                f"with {prefix_cached_len} < boundary < {prefix_cached_len + query_length}."
            )
        capture[boundary] = row

    if prefix_cached_len > 0:
        read_row, carry = state_restore_row, True
    else:
        read_row, carry = batch_idx, False
    schedule = []
    num_windows = -(-query_length // prefill_chunk_size)
    for window in range(num_windows):
        window_end = prefix_cached_len + (window + 1) * prefill_chunk_size
        write_row = capture.get(window_end, batch_idx)
        schedule.append((read_row, write_row, carry))
        read_row, carry = write_row, True
    return schedule


class RBLNQwen3_5RuntimeModel(RBLNRuntimeModel):
    """Runtime for the hybrid Qwen3.5 text backbone (batch_size == 1 for now).

    ``full_attention`` layers use the on-device paged KV cache (handled by the base runtime, whose
    buffers are static and never passed at call time). The ``linear_attention`` (GatedDeltaNet) layers
    carry two extra states per layer — ``conv_state`` and ``recurrent_state`` — which are ALSO on-device
    STATIC caches: they are marked static (``mark_static_address``) in the Qwen3.5 compile context
    (``_qwen3_5_build_compile_context``) and read + written entirely in-graph via ``rbln_cache_update``.
    So, like the KV cache, they live in device DRAM and are NEVER passed at call time — this runtime does
    NOT hold state values on the host:

        prefill window 0 -> ... -> prefill window N -> decode step 0 -> decode step 1 -> ...

    The runtime's only linear-state job is to inject two 0/1 control masks per call — ``conv_state_mask``
    and ``recurrent_state_mask`` — which the GatedDeltaNet multiplies into the state it reads: ZEROS on
    prefill window 0 (fresh sequence, so the stale static cache is discarded) and ONES afterwards (carry
    whatever the previous window/step wrote). ``_run`` maps the named inputs onto the runtime's own
    input order via ``_index_to_input_name``.

    Hybrid prefix caching (``linear_state_snapshot_slots`` K > 0): the state caches have K extra snapshot
    rows after the ``batch_size`` live rows, and each prefill window gets the row it reads
    (``state_src_idx``) separately from the row it writes (``batch_idx``). ``forward`` then accepts
    ``state_restore_row`` (resume a prompt whose ``cache_position`` starts at a cached boundary from that
    snapshot row) and ``state_capture`` (``{boundary: row}``: save the state after ``boundary`` tokens into
    that snapshot row). See ``plan_linear_state_windows`` for the per-window schedule.
    """

    def __init__(
        self,
        *args,
        conv_state_shape=None,
        recurrent_state_shape=None,
        state_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv_state_shape = tuple(conv_state_shape)
        self.recurrent_state_shape = tuple(recurrent_state_shape)
        self.state_dtype = state_dtype

        # Prefill state masks: zeros on the first window of a fresh sequence, ones whenever the window carries
        # the state it reads (later windows, or the first window resuming from a prefix snapshot).
        self._conv_mask_zeros = torch.zeros(self.conv_state_shape, dtype=state_dtype)
        self._conv_mask_ones = torch.ones(self.conv_state_shape, dtype=state_dtype)
        self._recurrent_mask_zeros = torch.zeros(self.recurrent_state_shape, dtype=state_dtype)
        self._recurrent_mask_ones = torch.ones(self.recurrent_state_shape, dtype=state_dtype)
        self._valid_mask_prefill_full = torch.ones(1, self.rbln_config.prefill_chunk_size, 1, dtype=state_dtype)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        batch_idx: int | None = None,
        block_tables: torch.Tensor | None = None,
        position_embed: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        local_block_tables: torch.Tensor | None = None,
        lora_int_ids: torch.Tensor | None = None,
        state_restore_row: int | None = None,
        state_capture: dict[int, int] | None = None,
    ):
        """Same dispatch as `RBLNRuntimeModel.forward`, plus the prefill-only linear-state snapshot arguments.

        Args:
            state_restore_row (int | None): Snapshot row (`batch_size + s`) that holds the GatedDeltaNet state
                after exactly `cache_position[0, 0]` tokens. Required when the prefill starts after a cached
                prefix; the first window resumes from it.
            state_capture (dict[int, int] | None): `{boundary: snapshot_row}`. The window that ends at absolute
                position `boundary` writes its state to `snapshot_row` instead of `batch_idx`.
        """
        if self.phase == "decode" and (state_restore_row is not None or state_capture):
            raise ValueError("state_restore_row / state_capture only apply to prefill.")

        inputs = self.inputs_embeddings_if_needed(input_ids, inputs_embeds)
        block_tables, local_block_tables, is_external_block_tables = (
            self.page_table_manager.get_block_tables_if_needed(
                self.batch_size,
                cache_position,
                batch_idx=batch_idx,
                phase=self.phase,
                block_tables=block_tables,
                local_block_tables=local_block_tables,
            )
        )

        if self.phase == "decode":
            return self.decode_forward(
                inputs,
                cache_position,
                block_tables,
                is_external_block_tables,
                attention_mask=attention_mask,
                position_embed=position_embed,
                position_ids=position_ids,
                local_block_tables=local_block_tables,
                lora_int_ids=lora_int_ids,
            )
        return self.prefill_forward(
            inputs,
            cache_position,
            attention_mask,
            batch_idx,
            block_tables,
            is_external_block_tables=is_external_block_tables,
            position_embed=position_embed,
            token_type_ids=token_type_ids,
            local_block_tables=local_block_tables,
            lora_int_ids=lora_int_ids,
            state_restore_row=state_restore_row,
            state_capture=state_capture,
        )

    def _run(self, named_inputs: dict):
        """Order inputs by the runtime's own signature and invoke; return (logits, hidden_states).

        When output_hidden_states is set, the trailing `num_hidden_layers + 1` outputs are the per-layer
        hidden states — taking the LAST n_hidden avoids having to count the new_states.
        """
        order = self.runtime._index_to_input_name
        args = [named_inputs[order[k]] for k in range(len(order))]
        out = super(RBLNRuntimeModel, self).forward(*args)
        hidden_states = None
        if self.rbln_config.output_hidden_states:
            n_hidden = self.config.num_hidden_layers + 1
            hidden_states = tuple(out[-n_hidden:])
        return out[0], hidden_states

    def prefill_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        batch_idx: int | None = None,
        block_tables: torch.Tensor | None = None,
        is_external_block_tables: bool | None = None,
        position_ids: torch.Tensor | None = None,
        position_embed: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        local_block_tables: torch.Tensor | None = None,
        lora_int_ids: torch.Tensor | None = None,
        state_restore_row: int | None = None,
        state_capture: dict[int, int] | None = None,
    ) -> RBLNDecoderOnlyOutput:
        if self.rbln_config.use_lora and lora_int_ids is None:
            if self.lora_int_ids is None:
                raise ValueError(
                    "lora_int_id is required when using LoRA. "
                    "You should call set_lora_int_ids() before forward() or pass lora_int_id to forward()."
                )
            if batch_idx is not None:
                lora_int_ids = self.lora_int_ids[batch_idx : batch_idx + 1].clone()
            else:
                lora_int_ids = self.lora_int_ids.clone()

        (
            inputs,
            cache_position,
            chunked_attention_mask,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids,
        ) = self._prepare_prefill_inputs(
            inputs, cache_position, attention_mask, position_ids, position_embed, token_type_ids=token_type_ids
        )

        chunk = self.rbln_config.prefill_chunk_size
        # A cached prefix (cache_position starting at P > 0) is resumable only from a GatedDeltaNet snapshot row:
        # the plan gives every window the state row it reads, the row it writes, and whether it carries the state.
        prefix_cached_len = cache_position[0][0].item()
        snapshot_slots = self.rbln_config.linear_state_snapshot_slots
        window_plan = plan_linear_state_windows(
            prefix_cached_len,
            query_length,
            chunk,
            batch_idx,
            self.rbln_config.batch_size,
            snapshot_slots,
            state_restore_row=state_restore_row,
            state_capture=state_capture,
        )
        if self.rbln_config.use_attention_mask and prefix_cached_len > 0:
            chunked_attention_mask[:, :, :, :prefix_cached_len] = 1
        logits = None
        # For logits_to_keep == 0 (bare text model) the graph emits full-chunk hidden states as
        # "logits", so every window must be collected — keeping only the last window would drop
        # all earlier windows of a multi-chunk prompt and return padded chunk width.
        collect_full_logits = self.rbln_config.logits_to_keep == 0
        all_logits = [] if collect_full_logits else None
        all_hidden_states = [] if self.rbln_config.output_hidden_states else None

        for (read_row, write_row, carry), step in zip(window_plan, range(0, inputs.shape[1], chunk), strict=True):
            input_chunk = inputs[:, step : step + chunk]
            cache_pos_chunk = cache_position[:, step : step + chunk]
            position_embed_chunk = (
                position_embed[:, :, :, step : step + chunk, :] if position_embed is not None else None
            )

            # Reveal the current chunk (and previously seen tokens) in the causal attention mask.
            if self.rbln_config.use_attention_mask:
                if step > 0:
                    chunked_attention_mask[:, :, :, prefix_cached_len : prefix_cached_len + step] = 1
                chunked_attention_mask[:, :, :, step + prefix_cached_len : step + prefix_cached_len + chunk] = (
                    self.causal_mask
                )

            query_position = (
                torch.tensor(
                    (query_length - 1) % chunk if step + chunk >= query_length else chunk - 1, dtype=torch.int16
                )
                if self.rbln_config.logits_to_keep > 0
                else None
            )

            named = {"inputs_embeds" if self.rbln_config.use_inputs_embeds else "input_ids": input_chunk}
            named["cache_position"] = cache_pos_chunk
            if block_tables is not None:
                named["block_tables"] = block_tables
            if position_embed_chunk is not None:
                named["position_emb"] = position_embed_chunk
            if self.rbln_config.logits_to_keep > 0:
                named["query_position"] = query_position
            if self.rbln_config.use_attention_mask:
                named["attention_mask"] = chunked_attention_mask
            if self.rbln_config.use_lora:
                named["lora_int_ids"] = lora_int_ids

            # State masks: ZERO the read state on the first window of a fresh sequence (no prior context), ONES
            # when the window carries it (a previous window's state, or a restored prefix snapshot).
            named["conv_state_mask"] = self._conv_mask_ones if carry else self._conv_mask_zeros
            named["recurrent_state_mask"] = self._recurrent_mask_ones if carry else self._recurrent_mask_zeros

            # Which row of the linear state caches this per-item (batch=1) prefill writes (`batch_idx`) and, with
            # snapshot rows, reads (`state_src_idx`). Without snapshot rows the graph reads and writes `batch_idx`.
            if write_row is not None:
                named["batch_idx"] = torch.tensor(write_row, dtype=torch.int16)
            if snapshot_slots > 0:
                named["state_src_idx"] = torch.tensor(read_row, dtype=torch.int16)

            # Per-token validity (1=real, 0=right-padding); the GatedDeltaNet multiplies it into g/beta to drop
            # padding. Full windows reuse the precomputed all-ones; the partial last window is built on demand.
            valid_count = max(0, min(chunk, query_length - step))
            if valid_count >= chunk:
                valid_mask = self._valid_mask_prefill_full
            else:
                valid_mask = torch.zeros(input_chunk.shape[0], chunk, 1, dtype=self.state_dtype)
                valid_mask[:, :valid_count] = 1.0
            named["valid_mask"] = valid_mask

            # For logits_to_keep == 1 every window overwrites the single logits row, so the final value
            # is the last window's (the next-token logits). Intermediate windows only advance the states.
            logits, hidden = self._run(named)
            if collect_full_logits:
                # keep only this window's valid (non-right-padding) tokens
                all_logits.append(logits[:, :valid_count, :])
            if self.rbln_config.output_hidden_states:
                # keep only this window's valid (non-right-padding) tokens
                all_hidden_states.append(tuple(h[:, :valid_count, :] for h in hidden))

        def place_at_mask_slots(valid_output: torch.Tensor) -> torch.Tensor:
            # `_prepare_prefill_inputs` strips padding (inputs[:, mask_bool]) before the graph, so the
            # graph only produced `query_length` valid tokens. Scatter them back to their mask slots
            # (padding stays zero) so outputs line up with the attention mask.
            if attention_mask is None:
                return valid_output
            full_len = attention_mask.shape[-1]
            start = int(torch.nonzero(attention_mask.reshape(-1), as_tuple=False)[0].item())
            buf = torch.zeros(1, full_len, valid_output.shape[-1], dtype=valid_output.dtype)
            buf[:, start : start + query_length, :] = valid_output
            return buf

        if collect_full_logits:
            # Concat per-window valid tokens along seq -> [1, query_length, hidden].
            logits = place_at_mask_slots(torch.cat(all_logits, dim=1))

        final_hidden_states = None
        if self.rbln_config.output_hidden_states:
            # Concat each layer's per-window valid tokens along seq -> [1, query_length, hidden].
            n_hidden = len(all_hidden_states[0])
            final_hidden_states = tuple(
                place_at_mask_slots(torch.cat([window[layer] for window in all_hidden_states], dim=1))
                for layer in range(n_hidden)
            )

        # padded_cache_lengths (from _prepare_prefill_inputs) is threaded back so the base
        # RBLNDecoderOnlyModelForCausalLM.forward can accumulate it per batch (the VL forward ignores it).
        return RBLNDecoderOnlyOutput(
            logits=logits, padded_cache_lengths=padded_cache_lengths, hidden_states=final_hidden_states
        )

    def decode_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        attention_mask: torch.Tensor | None = None,
        position_embed: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        local_block_tables: torch.Tensor | None = None,
        lora_int_ids: torch.Tensor | None = None,
    ) -> RBLNDecoderOnlyOutput:
        if self.rbln_config.use_lora and lora_int_ids is None:
            if self.lora_int_ids is None:
                raise ValueError(
                    "lora_int_id is required when using LoRA. "
                    "You should call set_lora_int_ids() before forward() or pass lora_int_id to forward()."
                )
            lora_int_ids = self.lora_int_ids
        if lora_int_ids is not None and lora_int_ids.shape[0] != self.batch_size:
            raise ValueError(f"lora_int_ids size mismatch: got {lora_int_ids.shape[0]}, expected {self.batch_size}.")

        if self.rbln_config.use_attention_mask and attention_mask is None:
            for b_idx in range(self.batch_size):
                decoding_step = cache_position[b_idx].item()
                if not (0 <= decoding_step < self.dec_attn_mask.shape[-1]):
                    raise ValueError(
                        f"Decoding step {decoding_step} out of bounds for attention mask "
                        f"with shape {self.dec_attn_mask.shape}."
                    )
                self.dec_attn_mask[b_idx, :, :, decoding_step] = 1
            attention_mask = self.dec_attn_mask

        named = {"inputs_embeds" if self.rbln_config.use_inputs_embeds else "input_ids": inputs}
        named["cache_position"] = cache_position
        if block_tables is not None:
            named["block_tables"] = block_tables
        if position_embed is not None:
            named["position_emb"] = position_embed
        if self.rbln_config.use_attention_mask:
            named["attention_mask"] = attention_mask
        if self.rbln_config.use_lora:
            named["lora_int_ids"] = lora_int_ids

        logits, hidden_states = self._run(named)
        return RBLNDecoderOnlyOutput(logits=logits, hidden_states=hidden_states)
