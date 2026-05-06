"""Qwen3.5-35B-A3B RBLN model implementation.

Hybrid architecture: 30 DeltaNet + 10 Full Attention + MoE (256 experts + shared).
No DeepStack (deepstack_visual_indexes=[]).

Based on Qwen3-VL-30B-A3B-Instruct/rbln_qwen3_vl_moe/modeling_qwen3_vl_moe.py.
"""

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)

from transformers import AutoModelForVision2Seq  # patched by _compat_shim
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.modeling_utils import no_init_weights  # patched above for transformers 5.x
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeModel,
    Qwen3_5MoeVisionModel,
    Qwen3_5MoeVisionPatchEmbed,
    Qwen3_5MoeVisionRotaryEmbedding,
    Qwen3_5MoeTextRotaryEmbedding,
)

from optimum.rbln.configuration_utils import CONFIG_MAPPING, RBLNCompileConfig
from optimum.rbln.modeling import RBLNModel
from optimum.rbln.utils.model_utils import MODEL_MAPPING
from optimum.rbln.transformers.modeling_outputs import RBLNDecoderOnlyOutput, _validate_output_hidden_states
from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import (
    RBLNDecoderOnlyModel,
    RBLNDecoderOnlyModelForCausalLM,
)
from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import (
    RBLNRuntimeModel as _BaseRBLNRuntimeModel,
    RBLNPageTableManager,
)
from optimum.rbln.utils.runtime_utils import RBLNPytorchRuntime

from .configuration_qwen3_5_moe import (
    RBLNQwen3_5MoeForConditionalGenerationConfig,
    RBLNQwen3_5MoeVisionModelConfig,
)
from .qwen3_5_moe_architecture import Qwen3_5VisionTransformerWrapper, Qwen3_5Moe_LanguageModelWrapper


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


# Register config classes
CONFIG_MAPPING["RBLNQwen3_5MoeForConditionalGenerationConfig"] = RBLNQwen3_5MoeForConditionalGenerationConfig
CONFIG_MAPPING["RBLNQwen3_5MoeVisionModelConfig"] = RBLNQwen3_5MoeVisionModelConfig


# ==============================================================================
# Vision Model
# ==============================================================================


class RBLNQwen3_5MoeVisionModel(RBLNModel):
    """RBLN optimized Qwen3.5 vision transformer.

    Same architecture as Qwen3-VL (27 blocks, full attention, no DeepStack).
    """

    auto_model_class = None
    _supports_non_fp32 = True
    # Without this flag, `RBLNModelConfig.filter_parameters` silently drops
    # `tensor_parallel_size` from the visual submodule config (see
    # `optimum/rbln/configuration_utils.py:633-635`), which causes the vision
    # encoder to compile with TP=1 even when the user requests TP>1.
    _tp_support = True

    def __post_init__(self, **kwargs):
        self.transformer = self.model[0]
        self.max_seq_lens = torch.tensor(sorted(self.rbln_config.max_seq_lens, reverse=False))
        config = self.config
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5MoeVisionRotaryEmbedding(head_dim // 2)

        with no_init_weights():
            self.patch_embed = Qwen3_5MoeVisionPatchEmbed(config=config)
            self.pos_embed = torch.nn.Embedding(config.num_position_embeddings, config.hidden_size)

        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)

        artifacts = torch.load(
            self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False
        )
        self.patch_embed.load_state_dict(artifacts["patch_embed"])
        if "pos_embed_state" in artifacts:
            self.pos_embed.load_state_dict(artifacts["pos_embed_state"])

    @classmethod
    def save_torch_artifacts(cls, model, save_dir_path: Path, subfolder: str, rbln_config):
        save_dict = {
            "patch_embed": model.patch_embed.state_dict(),
        }
        if hasattr(model, "pos_embed"):
            save_dict["pos_embed"] = model.pos_embed.weight.data
            save_dict["pos_embed_state"] = model.pos_embed.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config):
        return Qwen3_5VisionTransformerWrapper(model, rbln_config).eval()

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(RBLNQwen3_5MoeVisionModel, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def _update_rbln_config(cls, preprocessors, model=None, model_config=None, rbln_config=None):
        hidden_size = model_config.hidden_size
        num_heads = model_config.num_heads
        head_dim = hidden_size // num_heads

        input_infos = []
        for max_seq_len in rbln_config.max_seq_lens:
            input_info = [
                ("hidden_states", [max_seq_len, hidden_size], rbln_config.dtype),
                ("attn_masks", [1, 1, max_seq_len, max_seq_len], rbln_config.dtype),
                ("cos", [1, 1, max_seq_len, head_dim], rbln_config.dtype),
                ("sin", [1, 1, max_seq_len, head_dim], rbln_config.dtype),
            ]
            input_infos.append(input_info)

        rbln_compile_config = RBLNCompileConfig(input_info=input_infos)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute rotary position embeddings for vision encoder."""
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset: offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        """Interpolate position embeddings for variable resolution."""
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        hidden_states = self.patch_embed(hidden_states).to(self.rbln_config.dtype)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos().to(self.rbln_config.dtype), emb.sin().to(self.rbln_config.dtype))

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        num_images = len(cu_seqlens) - 1
        output_hidden_states = []

        for i in range(num_images):
            image_s, image_e = cu_seqlens[i], cu_seqlens[i + 1]
            image_len = image_e - image_s

            try:
                ws_index = torch.searchsorted(self.max_seq_lens, image_len).item()
                max_seq_len = self.max_seq_lens[ws_index]
            except Exception as e:
                raise ValueError(
                    f"Required seq_len({image_len}) > max_seq_lens({self.max_seq_lens.tolist()})."
                ) from e

            img_hidden = hidden_states[image_s:image_e]
            img_cos = position_embeddings[0][image_s:image_e]
            img_sin = position_embeddings[1][image_s:image_e]

            if img_hidden.shape[0] < max_seq_len:
                pad_size = max_seq_len - img_hidden.shape[0]
                img_hidden = torch.cat([img_hidden, torch.zeros(pad_size, img_hidden.shape[-1], dtype=img_hidden.dtype)])
                img_cos = torch.cat([img_cos, torch.zeros(pad_size, img_cos.shape[-1], dtype=img_cos.dtype)])
                img_sin = torch.cat([img_sin, torch.zeros(pad_size, img_sin.shape[-1], dtype=img_sin.dtype)])

            attn_masks = torch.ones(1, 1, max_seq_len, max_seq_len, dtype=img_hidden.dtype)
            if image_len < max_seq_len:
                attn_masks[:, :, image_len:, :] = 0
                attn_masks[:, :, :, image_len:] = 0

            head_dim = self.config.hidden_size // self.config.num_heads
            outputs = self.transformer(
                img_hidden,
                attn_masks,
                img_cos[None, None, :, :head_dim],
                img_sin[None, None, :, :head_dim],
            )

            valid_merged_len = image_len // self.spatial_merge_unit
            output_hidden_states.append(outputs[:valid_merged_len])

        hidden_states = torch.cat(output_hidden_states)
        return hidden_states


MODEL_MAPPING["RBLNQwen3_5MoeVisionModel"] = RBLNQwen3_5MoeVisionModel


# ==============================================================================
# Language Runtime Model
# ==============================================================================


class Qwen3_5MoeRuntimeModel(_BaseRBLNRuntimeModel):
    """Runtime model managing hybrid state (KV-cache + DeltaNet recurrent/conv)."""

    def __init__(self, num_deltanet_layers=30, text_config=None, **kwargs):
        super().__init__(**kwargs)
        self.num_deltanet_layers = num_deltanet_layers
        self._text_config = text_config
        self._conv_states = None
        self._recurrent_states = None

    def _init_deltanet_states(self, batch_size, dtype):
        """Initialize DeltaNet states to zeros."""
        tc = self._text_config
        conv_dim = tc.linear_num_key_heads * tc.linear_key_head_dim * 2 + tc.linear_num_value_heads * tc.linear_value_head_dim
        conv_states = [
            torch.zeros(batch_size, conv_dim, tc.linear_conv_kernel_dim - 1, dtype=dtype)
            for _ in range(self.num_deltanet_layers)
        ]
        rec_states = [
            torch.zeros(batch_size, tc.linear_num_value_heads, tc.linear_key_head_dim, tc.linear_value_head_dim, dtype=dtype)
            for _ in range(self.num_deltanet_layers)
        ]
        return conv_states, rec_states

    def forward(self, conv_states=None, recurrent_states=None, **kwargs):
        self._conv_states = conv_states
        self._recurrent_states = recurrent_states
        return super().forward(**kwargs)

    def prefill_forward(
        self,
        inputs,
        cache_position=None,
        attention_mask=None,
        batch_idx=None,
        block_tables=None,
        is_external_block_tables=None,
        position_ids=None,
        position_embed=None,
        token_type_ids=None,
        local_block_tables=None,
        lora_int_ids=None,
    ):
        if self.rbln_config.use_lora and lora_int_ids is None:
            if self.lora_int_ids is None:
                raise ValueError("lora_int_id is required when using LoRA.")
            if batch_idx is not None:
                lora_int_ids = self.lora_int_ids[batch_idx: batch_idx + 1].clone()
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

        out_buffers, output_logits, output_hidden_states = self._prepare_prefill_outputs(query_length, attention_mask)

        # Initialize DeltaNet states
        conv_states, recurrent_states = self._init_deltanet_states(1, inputs.dtype)
        if self._conv_states is not None:
            conv_states = self._conv_states
        if self._recurrent_states is not None:
            recurrent_states = self._recurrent_states

        padded_seq_len = inputs.shape[1]

        prefix_cached_len = cache_position[0][0].item()
        if prefix_cached_len > 0:
            if prefix_cached_len % self.rbln_config.prefill_chunk_size != 0:
                raise NotImplementedError(
                    "Prefix Caching is not supported for non-multiple of prefill_chunk_size."
                )
            if self.rbln_config.use_attention_mask:
                if self.rbln_config.use_position_ids:
                    chunked_attention_mask[:, :prefix_cached_len] = 1
                else:
                    chunked_attention_mask[:, :, :, :prefix_cached_len] = 1

        for i, step in enumerate(range(0, query_length, self.rbln_config.prefill_chunk_size)):
            s, e = step, step + self.rbln_config.prefill_chunk_size
            input_chunk = inputs[:, s:e]
            cache_pos_chunk = cache_position[:, s:e]
            position_ids_chunk = position_ids[:, s:e] if self.rbln_config.use_position_ids else None
            position_embed_chunk = position_embed[:, :, :, s:e, :].contiguous() if position_embed is not None else None

            if self.rbln_config.use_attention_mask:
                if self.rbln_config.use_position_ids:
                    if step > 0:
                        prev_s = s - self.rbln_config.prefill_chunk_size + prefix_cached_len
                        prev_e = s + prefix_cached_len
                        chunked_attention_mask[:, prev_s:prev_e] = 1
                    cur_s = s + prefix_cached_len
                    cur_e = min(e, query_length) + prefix_cached_len
                    if cur_e > cur_s:
                        chunked_attention_mask[:, cur_s:cur_e] = 1
                else:
                    if step > 0:
                        prev_s = s - self.rbln_config.prefill_chunk_size + prefix_cached_len
                        prev_e = s + prefix_cached_len
                        chunked_attention_mask[:, :, :, prev_s:prev_e] = 1
                    cur_s = s + prefix_cached_len
                    cur_e = e + prefix_cached_len
                    chunked_attention_mask[:, :, :, cur_s:cur_e] = self.causal_mask

            if self.rbln_config.use_local_attention or self.rbln_config.logits_to_keep > 0:
                query_position = (
                    torch.tensor((query_length - 1) % self.rbln_config.prefill_chunk_size, dtype=torch.int16)
                    if e >= query_length
                    else torch.tensor(self.rbln_config.prefill_chunk_size - 1, dtype=torch.int16)
                )
            else:
                query_position = None

            runtime_outputs = RBLNPytorchRuntime.forward(
                self,
                input_chunk,
                cache_pos_chunk,
                block_tables,
                local_block_tables,
                position_embed_chunk,
                *conv_states,
                *recurrent_states,
                query_position,
                chunked_attention_mask if self.rbln_config.use_attention_mask else None,
                position_ids_chunk,
                lora_int_ids if self.rbln_config.use_lora else None,
                out=out_buffers[i],
            )

            # Extract updated DeltaNet states from runtime outputs
            if isinstance(runtime_outputs, (list, tuple)):
                # Output order: logits, conv_states..., recurrent_states...
                n_dn = self.num_deltanet_layers
                if len(runtime_outputs) > 1:
                    conv_states = list(runtime_outputs[1:1 + n_dn])
                    recurrent_states = list(runtime_outputs[1 + n_dn:1 + 2 * n_dn])

        padding_size = (self.rbln_config.prefill_chunk_size - query_length) % self.rbln_config.prefill_chunk_size
        if self.rbln_config.logits_to_keep == 1:
            output_logits = output_logits
        elif self.rbln_config.logits_to_keep > 1:
            output_logits = output_logits[:, -padding_size - self.rbln_config.logits_to_keep: -padding_size, :]
        else:
            output_logits = output_logits[:, :-padding_size, :]

        all_hidden_states = None
        if self.rbln_config.output_hidden_states:
            all_hidden_states = tuple(h[:, :-padding_size, :] for h in output_hidden_states)

        if self.rbln_config.can_generate and not is_external_block_tables and self.rbln_config.use_attention_mask:
            if self.rbln_config.use_position_ids:
                self.dec_attn_mask[batch_idx: batch_idx + 1] = chunked_attention_mask
            else:
                self.dec_attn_mask[batch_idx].fill_(0)
                self.dec_attn_mask[batch_idx, :, :, :query_length] = 1

        # Save DeltaNet states for decode phase
        self._conv_states = conv_states
        self._recurrent_states = recurrent_states

        return RBLNDecoderOnlyOutput(
            logits=output_logits, padded_cache_lengths=padded_cache_lengths, hidden_states=all_hidden_states
        )

    def decode_forward(
        self,
        inputs,
        cache_position=None,
        block_tables=None,
        is_external_block_tables=None,
        attention_mask=None,
        position_embed=None,
        position_ids=None,
        local_block_tables=None,
        lora_int_ids=None,
    ):
        if self.rbln_config.use_lora and lora_int_ids is None:
            if self.lora_int_ids is None:
                raise ValueError("lora_int_id is required when using LoRA.")
            lora_int_ids = self.lora_int_ids

        batch_size = inputs.shape[0]
        if batch_size != self.batch_size:
            raise RuntimeError(
                f"Batch size mismatch: got {batch_size}, expected {self.batch_size}."
            )

        if self.rbln_config.use_attention_mask and attention_mask is None:
            for b_idx in range(batch_size):
                decoding_step = cache_position[b_idx].item()
                if self.rbln_config.use_position_ids:
                    self.dec_attn_mask[b_idx, decoding_step] = 1
                else:
                    if is_external_block_tables:
                        self.dec_attn_mask[b_idx].fill_(0)
                        self.dec_attn_mask[b_idx, :, :, : decoding_step + 1] = 1
                    else:
                        self.dec_attn_mask[b_idx, :, :, decoding_step] = 1
            attention_mask = self.dec_attn_mask

        # Get current DeltaNet states
        conv_states = self._conv_states
        recurrent_states = self._recurrent_states

        if conv_states is None or recurrent_states is None:
            conv_states, recurrent_states = self._init_deltanet_states(batch_size, inputs.dtype)

        outputs = RBLNPytorchRuntime.forward(
            self,
            inputs,
            cache_position,
            block_tables,
            local_block_tables,
            position_embed,
            *conv_states,
            *recurrent_states,
            attention_mask if self.rbln_config.use_attention_mask else None,
            position_ids if self.rbln_config.use_position_ids else None,
            lora_int_ids if self.rbln_config.use_lora else None,
            out=self.out_buffers,
        )

        # Parse outputs: logits + updated DeltaNet states
        if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
            n_dn = self.num_deltanet_layers
            logits = outputs[0]
            self._conv_states = list(outputs[1:1 + n_dn])
            self._recurrent_states = list(outputs[1 + n_dn:1 + 2 * n_dn])
        else:
            logits = outputs

        # Hidden states aren't surfaced from the compiled graph (the wrapper packs
        # only [logits, conv_states..., recurrent_states...]). Reject the request
        # rather than silently returning None.
        if self.rbln_config.output_hidden_states:
            raise NotImplementedError(
                "RBLNQwen3_5MoeForConditionalGeneration does not expose hidden_states "
                "from compiled decode. Recompile with output_hidden_states=False."
            )
        return RBLNDecoderOnlyOutput(logits=logits, hidden_states=None)


# ==============================================================================
# Full Model
# ==============================================================================


class RBLNQwen3_5MoeModel(RBLNDecoderOnlyModel):
    """RBLN Qwen3.5-MoE model (base model)."""

    auto_model_class = AutoModelForVision2Seq
    _decoder_wrapper_cls = Qwen3_5Moe_LanguageModelWrapper
    _use_rotary_emb = False
    _rbln_submodules = [{"name": "visual"}]
    _config_class = Qwen3_5MoeConfig

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        # transformers 5.x removed use_auth_token in favor of token
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return super().get_pytorch_model(*args, **kwargs)


    def __post_init__(self, **kwargs):
        if not isinstance(self.config.text_config, PretrainedConfig):
            self.config = Qwen3_5MoeConfig(
                text_config=self.config.text_config,
                vision_config=self.config.vision_config,
            )

        super().__post_init__(**kwargs)
        self.visual = self.rbln_submodules[0]

        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(self.config.text_config)

        self._layer_types = list(self.config.text_config.layer_types)
        self._num_deltanet_layers = self._layer_types.count("linear_attention")
        self._num_attn_layers = self._layer_types.count("full_attention")

        if not self.can_generate():
            self.block_tables = torch.arange(self.rbln_config.kvcache_num_blocks, dtype=torch.int16)

    def setup_runtime(self):
        page_table_manager = RBLNPageTableManager(self.rbln_config)
        if self.rbln_config.use_position_ids:
            dec_attn_mask = torch.zeros(self.rbln_config.batch_size, self.rbln_config.max_seq_len, dtype=self.dtype)
        else:
            dec_attn_mask = torch.zeros(
                self.rbln_config.batch_size, 1, 1, self.rbln_config.max_seq_len, dtype=self.dtype
            )

        common_kwargs = {
            "main_input_name": "inputs_embeds" if self.rbln_config.use_inputs_embeds else "input_ids",
            "embed_tokens": self.embed_tokens,
            "dec_attn_mask": dec_attn_mask,
            "page_table_manager": page_table_manager,
            "rbln_config": self.rbln_config,
            "config": self.config,
        }

        self.prefill_decoder = Qwen3_5MoeRuntimeModel(
            runtime=self.model[0],
            phase="prefill",
            batch_size=self.rbln_config.batch_size,
            logits_last_dim=self.logits_last_dim,
            num_deltanet_layers=self._num_deltanet_layers,
            text_config=self.config.text_config,
            **common_kwargs,
        )
        if self.can_generate():
            self.decoders = {}
            for i, batch_size in enumerate(self.rbln_config.decoder_batch_sizes):
                self.decoders[batch_size] = Qwen3_5MoeRuntimeModel(
                    runtime=self.model[i + 1],
                    phase="decode",
                    batch_size=batch_size,
                    num_deltanet_layers=self._num_deltanet_layers,
                    text_config=self.config.text_config,
                    **common_kwargs,
                )
            self.decoder = self.decoders[self.rbln_config.batch_size]

    @property
    def logits_last_dim(self):
        return self.config.text_config.vocab_size if self.can_generate() else self.config.text_config.hidden_size

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = torch.nn.Embedding(
                self.config.text_config.vocab_size,
                self.config.text_config.hidden_size,
                getattr(self.config.text_config, "pad_token_id", None),
            )
        return embed_tokens

    @classmethod
    def _update_rbln_config(cls, preprocessors=None, model=None, model_config=None, rbln_config=None):
        text_config = getattr(model_config, "text_config", model_config)

        # Override num_hidden_layers to only count full attention layers for KV-cache
        num_attn_layers = text_config.layer_types.count("full_attention")
        original_num_layers = text_config.num_hidden_layers
        text_config.num_hidden_layers = num_attn_layers

        result = super()._update_rbln_config(
            preprocessors=preprocessors, model=model, model_config=text_config, rbln_config=rbln_config
        )

        # Restore original
        text_config.num_hidden_layers = original_num_layers
        return result

    @classmethod
    def get_input_info(cls, batch_size, query_length, rbln_config, model_config):
        # Get base input info (with num_hidden_layers = num_attn_layers for KV-cache)
        input_info = super().get_input_info(batch_size, query_length, rbln_config, model_config)

        head_dim = getattr(model_config, "head_dim", None) or model_config.hidden_size // model_config.num_attention_heads
        pos_idx = 3

        # Insert position embeddings
        input_info.insert(
            pos_idx,
            ("position_emb", [2, batch_size, 1, query_length, head_dim], rbln_config.dtype),
        )

        # Insert DeltaNet state inputs after position_emb
        layer_types = model_config.layer_types
        num_dn = layer_types.count("linear_attention")
        conv_dim = (
            model_config.linear_num_key_heads * model_config.linear_key_head_dim * 2
            + model_config.linear_num_value_heads * model_config.linear_value_head_dim
        )
        conv_kernel = model_config.linear_conv_kernel_dim
        num_v_heads = model_config.linear_num_value_heads
        key_dim = model_config.linear_key_head_dim
        value_dim = model_config.linear_value_head_dim

        insert_idx = pos_idx + 1
        # Conv states
        for i in range(num_dn):
            input_info.insert(
                insert_idx,
                (f"conv_state_{i}", [batch_size, conv_dim, conv_kernel - 1], "float32"),
            )
            insert_idx += 1
        # Recurrent states
        for i in range(num_dn):
            input_info.insert(
                insert_idx,
                (f"recurrent_state_{i}", [batch_size, num_v_heads, key_dim, value_dim], "float32"),
            )
            insert_idx += 1

        return input_info

    def _get_position_embeddings(self, hidden_states, position_ids):
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos = cos.unsqueeze(1).to(self.rbln_config.dtype)
        sin = sin.unsqueeze(1).to(self.rbln_config.dtype)
        return torch.stack([cos, sin]).contiguous()

    def _get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        # transformers 5.x `Qwen3_5MoeModel.get_rope_index` requires `mm_token_type_ids`
        # (text=0, image=1, video=2) AND uses `self.get_vision_position_ids`. Build a
        # proxy that exposes both so we can call the unbound method without holding a
        # full Qwen3_5MoeModel instance after compile.
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
        image_token_id = getattr(self.config, "image_token_id", None)
        video_token_id = getattr(self.config, "video_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2

        class _ModelProxy:
            pass

        proxy = _ModelProxy()
        proxy.config = self.config
        # `get_vision_position_ids` is an instance method but doesn't actually use `self`
        # other than as a binding — pass the unbound function so the proxy lookup works.
        proxy.get_vision_position_ids = Qwen3_5MoeModel.get_vision_position_ids.__get__(
            proxy, _ModelProxy
        )

        return Qwen3_5MoeModel.get_rope_index(
            proxy,
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    def _preprocess_prefill(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.embed_tokens(input_ids).to(self.rbln_config.dtype)
        max_inputs_len = input_ids.shape[1]

        if pixel_values is not None:
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image tokens ({n_image_tokens}) != features ({n_image_features})"
                )
            mask = input_ids == self.config.image_token_id
            mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)

            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video tokens ({n_video_tokens}) != features ({n_video_features})"
                )
            mask = input_ids == self.config.video_token_id
            mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, video_embeds)

        head_dim = getattr(self.config.text_config, "head_dim", None) or (
            self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
        )
        all_position_embeds = torch.zeros(
            2, batch_size, 1, max_inputs_len, head_dim, dtype=self.rbln_config.dtype
        )
        all_rope_deltas = []

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        image_idx, video_idx = 0, 0

        for b_idx in range(batch_size):
            input_id = input_ids[b_idx: b_idx + 1][:, attention_mask[b_idx].bool()]
            vision_start_indices = torch.argwhere(input_id == vision_start_token_id).squeeze(1)
            vision_tokens = input_id[0][vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            position_ids, rope_deltas = self._get_rope_index(
                input_id,
                image_grid_thw[image_idx: image_idx + image_nums] if image_grid_thw is not None else None,
                video_grid_thw[video_idx: video_idx + video_nums] if video_grid_thw is not None else None,
            )
            image_idx += image_nums
            video_idx += video_nums

            position_embed = self._get_position_embeddings(inputs_embeds, position_ids)
            mask_indices = torch.nonzero(attention_mask[b_idx], as_tuple=True)[0]
            all_position_embeds[:, b_idx: b_idx + 1].index_copy_(dim=-2, index=mask_indices, source=position_embed)
            all_rope_deltas.append(rope_deltas)

        rope_deltas = torch.stack(all_rope_deltas)
        return inputs_embeds, all_position_embeds, rope_deltas


class RBLNQwen3_5MoeForConditionalGeneration(RBLNQwen3_5MoeModel, RBLNDecoderOnlyModelForCausalLM):
    """RBLN Qwen3.5-MoE for conditional generation."""

    auto_model_class = AutoModelForVision2Seq
    _decoder_wrapper_cls = Qwen3_5Moe_LanguageModelWrapper
    _supports_non_fp32 = True
    _use_rotary_emb = False
    _rbln_submodules = [{"name": "visual"}]

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.rope_deltas = torch.zeros(self.rbln_config.batch_size)

    def can_generate(self):
        return True

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        model.model.lm_head = model.lm_head
        # Expose visual at top level so RBLN submodule loading can find it
        model.visual = model.model.visual
        return model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        generate_idx=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        model_inputs = {}
        is_prefill_phase = generate_idx is None

        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            cache_position = None
            model_inputs["input_ids"] = input_ids
        else:
            input_ids = input_ids[:, -1:]
            cache_position = generate_idx
            generate_idx = generate_idx + 1
            model_inputs["input_ids"] = input_ids

        model_inputs.update({
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "generate_idx": generate_idx,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
        })
        return model_inputs

    def _preprocess_decoder(self, input_ids, cache_position):
        inputs_embeds = self.embed_tokens(input_ids).to(self.rbln_config.dtype)
        position_embeds = []
        for b_idx in range(self.rbln_config.batch_size):
            delta = cache_position[b_idx] + self.rope_deltas[b_idx]
            position_ids = torch.arange(1).view(1, -1).add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            position_embed = self._get_position_embeddings(
                torch.zeros(1, dtype=self.rbln_config.dtype), position_ids
            )
            position_embeds.append(position_embed)

        position_embeds = torch.cat(position_embeds, dim=1)
        return inputs_embeds, position_embeds

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        generate_idx=None,
        return_dict=None,
        output_hidden_states=None,
        **kwargs,
    ):
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)

        # Prefill
        if cache_position is None:
            inputs_embeds, position_embed, rope_deltas = self._preprocess_prefill(
                input_ids, attention_mask, pixel_values, pixel_values_videos,
                image_grid_thw, video_grid_thw,
            )
            batch_size, seq_len = inputs_embeds.shape[:2]
            self.rope_deltas = rope_deltas

            logits = []
            for b_idx in range(batch_size):
                cache_pos = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                output = self.prefill_decoder(
                    inputs_embeds=inputs_embeds[b_idx: b_idx + 1],
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_pos,
                    batch_idx=b_idx,
                    position_embed=position_embed[:, b_idx: b_idx + 1],
                )
                logits.append(output.logits)

            # Transfer DeltaNet states from prefill to decode runtime
            if self.can_generate():
                self.decoder._conv_states = self.prefill_decoder._conv_states
                self.decoder._recurrent_states = self.prefill_decoder._recurrent_states

            logits = torch.cat(logits, dim=0)
        # Decode
        else:
            inputs_embeds, position_embed = self._preprocess_decoder(input_ids, cache_position)
            output = self.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_embed=position_embed,
            )
            logits = output.logits

        if not return_dict:
            return (logits, generate_idx)
        return RBLNDecoderOnlyOutput(logits=logits, generate_idx=generate_idx)
