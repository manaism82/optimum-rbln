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

from typing import Any

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig


def _validate_linear_state_snapshot_slots(config: RBLNDecoderOnlyModelConfig) -> None:
    """Validate `linear_state_snapshot_slots` (K) on a Qwen3.5 config.

    K > 0 appends K snapshot rows after the `batch_size` live rows of every GatedDeltaNet state cache. The decode
    graph reads and writes only the live rows, so it must run the full `batch_size` (a single decoder).
    """
    slots = config.linear_state_snapshot_slots
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 0:
        raise ValueError(f"linear_state_snapshot_slots must be an integer >= 0, got {slots!r}.")
    if slots > 0 and config.can_generate and config.decoder_batch_sizes != [config.batch_size]:
        raise ValueError(
            "linear_state_snapshot_slots > 0 requires a single decoder of batch_size "
            f"(decoder_batch_sizes == [{config.batch_size}]), got decoder_batch_sizes={config.decoder_batch_sizes}."
        )


class RBLNQwen3_5ForCausalLMConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """
    Configuration class for RBLN Qwen3.5 (text backbone) causal language models.

    Qwen3.5 is a hybrid decoder: most layers are `linear_attention` (GatedDeltaNet) and a
    minority are `full_attention` (gated softmax attention). Full-attention layers use the
    standard paged KV cache; linear-attention layers instead carry a `conv_state` and a
    `recurrent_state`. Which layers are linear is read from the HF `config.layer_types` into the internal
    `linear_attention_layers` field; this config extends `RBLNDecoderOnlyModelForCausalLMConfig` with
    `gdn_chunk_size`, `linear_attention_layers` and `linear_state_snapshot_slots` (GatedDeltaNet state rows
    for hybrid prefix caching).

    Example usage:
    ```python
    from optimum.rbln import RBLNQwen3_5ForCausalLM, RBLNQwen3_5ForCausalLMConfig

    config = RBLNQwen3_5ForCausalLMConfig(
        batch_size=1,
        max_seq_len=32768,
        tensor_parallel_size=4,
    )
    model = RBLNQwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-27B", export=True, rbln_config=config)
    ```
    """

    def __init__(
        self,
        gdn_chunk_size: int | None = None,
        linear_attention_layers: list[int] | None = None,
        linear_state_snapshot_slots: int = 0,
        **kwargs: Any,
    ):
        """
        Args:
            gdn_chunk_size (Optional[int]): GatedDeltaNet prefill sub-chunk size. Each prefill window
                is split into `prefill_chunk_size // gdn_chunk_size` sub-chunks processed by the chunked
                delta rule. Must divide `prefill_chunk_size`. `None` -> `prefill_chunk_size` (no split).
            linear_attention_layers (list[int] | None): The linear_attention (GatedDeltaNet)
                layer indices, populated automatically from `layer_types` at compile time (not user-set).
            linear_state_snapshot_slots (int): Number K of extra GatedDeltaNet state rows used as prefix
                snapshots for hybrid prefix caching. Defaults to 0 (off). With K > 0 every `conv_state_*` /
                `recurrent_state_*` cache has `batch_size + K` rows (snapshot `s` is row `batch_size + s`),
                the prefill graph takes an extra `state_src_idx` input (the row a window reads its carried
                state from; `batch_idx` stays the row it writes) and `decoder_batch_sizes` must be
                `[batch_size]`. With K = 0 the compiled graphs are unchanged.
            kwargs: Additional arguments passed to `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `linear_state_snapshot_slots` is not an integer >= 0, or is > 0 while
                `decoder_batch_sizes` != [batch_size].
        """
        super().__init__(**kwargs)
        self.gdn_chunk_size = gdn_chunk_size
        self.linear_attention_layers = linear_attention_layers or []
        self.linear_state_snapshot_slots = linear_state_snapshot_slots
        _validate_linear_state_snapshot_slots(self)


class RBLNQwen3_5TextModelConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration class for the bare RBLN Qwen3.5 text backbone (no LM head, text-only).

    Linear-attention layers are read from the HF `config.layer_types`; see
    `RBLNQwen3_5ForCausalLMConfig` for `gdn_chunk_size`.
    """

    def __init__(
        self,
        gdn_chunk_size: int | None = None,
        linear_attention_layers: list[int] | None = None,
        linear_state_snapshot_slots: int = 0,
        **kwargs: Any,
    ):
        """
        Args:
            gdn_chunk_size (Optional[int]): GatedDeltaNet prefill sub-chunk size. Each prefill window
                is split into `prefill_chunk_size // gdn_chunk_size` sub-chunks processed by the chunked
                delta rule. Must divide `prefill_chunk_size`. `None` -> `prefill_chunk_size` (no split).
            linear_attention_layers (list[int] | None): The linear_attention (GatedDeltaNet)
                layer indices, populated automatically from `layer_types` at compile time (not user-set).
            linear_state_snapshot_slots (int): Number K of extra GatedDeltaNet state rows used as prefix
                snapshots for hybrid prefix caching. Defaults to 0 (off). With K > 0 every `conv_state_*` /
                `recurrent_state_*` cache has `batch_size + K` rows (snapshot `s` is row `batch_size + s`),
                the prefill graph takes an extra `state_src_idx` input (the row a window reads its carried
                state from; `batch_idx` stays the row it writes) and `decoder_batch_sizes` must be
                `[batch_size]`. With K = 0 the compiled graphs are unchanged.
            kwargs: Additional arguments passed to `RBLNDecoderOnlyModelConfig`.

        Raises:
            ValueError: If `linear_state_snapshot_slots` is not an integer >= 0, or is > 0 while
                `decoder_batch_sizes` != [batch_size].
        """
        super().__init__(**kwargs)
        self.gdn_chunk_size = gdn_chunk_size
        self.linear_attention_layers = linear_attention_layers or []
        self.linear_state_snapshot_slots = linear_state_snapshot_slots
        _validate_linear_state_snapshot_slots(self)


class RBLNQwen3_5VisionModelConfig(RBLNModelConfig):
    """Vision encoder config for Qwen3.5: per-image `max_seq_len`."""

    # The position-embedding cache lives on the host and never reaches the compiled graph, so its size is a
    # load-time option: it is not written to rbln_config.json and can be changed when loading a compiled model.
    # It is kept in `_runtime_options` like `device` because that is what survives the reload of a submodule
    # config: the parent loads `visual` from its own rbln_config.json and carries over only `_runtime_options`
    # of the config it was given.
    subclass_non_save_attributes = ["pos_embed_cache_size"]

    def __init__(
        self,
        max_seq_len: int | list[int] = None,
        batch_size: int = 1,
        pos_embed_cache_size: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            max_seq_len (Optional[Union[int, List[int]]]): Vision Transformer attention max sequence
                length(s) = number of (merged) patches per image/video. RBLN runs inference per image, so
                set this to the max expected resolution to bound compute. Required.
            batch_size (int): the vision encoder runs one image at a time (the parent config forces this
                by default).
            pos_embed_cache_size (Optional[int]): Number of image sizes (patch grid height x width) whose
                interpolated position embeddings are kept on the host, so a repeated size skips the
                interpolation. The least recently used size is dropped first; 0 disables the cache. One entry
                holds height x width x vision hidden size float32 values. Defaults to 16.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `max_seq_len` is None or not provided, if `batch_size` is not 1, or if
                `pos_embed_cache_size` is not an integer >= 0.
        """
        super().__init__(**kwargs)

        if batch_size != 1:
            raise ValueError(f"The Qwen3.5 vision encoder only supports batch_size=1, got {batch_size}.")
        self.batch_size = batch_size

        self.pos_embed_cache_size = 16 if pos_embed_cache_size is None else pos_embed_cache_size

        if max_seq_len is not None:
            if isinstance(max_seq_len, int):
                max_seq_len = [max_seq_len]
            elif isinstance(max_seq_len, list):
                max_seq_len.sort(reverse=True)
        else:
            raise ValueError("'max_seq_len' must be specified.")

        self.max_seq_len = max_seq_len

    @property
    def pos_embed_cache_size(self) -> int:
        return self._runtime_options["pos_embed_cache_size"]

    @pos_embed_cache_size.setter
    def pos_embed_cache_size(self, pos_embed_cache_size: int):
        if (
            isinstance(pos_embed_cache_size, bool)
            or not isinstance(pos_embed_cache_size, int)
            or pos_embed_cache_size < 0
        ):
            raise ValueError(f"pos_embed_cache_size must be an integer >= 0, got {pos_embed_cache_size!r}.")
        self._runtime_options["pos_embed_cache_size"] = pos_embed_cache_size


class RBLNQwen3_5ModelConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration for the bare Qwen3.5 model (vision encoder + hybrid text, no LM head).

    Qwen3.5 is natively vision-language, so this is the multimodal model config. Independent of the
    Qwen3-VL config (inherits `RBLNDecoderOnlyModelConfig` directly), carrying its own `visual`
    submodule handling plus the Qwen3.5-specific `gdn_chunk_size`. Which layers are linear is read
    from the HF `config.text_config.layer_types`. The vision encoder output is injected into
    `inputs_embeds` (`use_inputs_embeds=True`).
    """

    submodules = ["visual"]
    subclass_non_save_attributes = ["_load_visual_runtime", "memory_budget"]

    def __init__(
        self,
        gdn_chunk_size: int | None = None,
        linear_attention_layers: list[int] | None = None,
        visual: RBLNModelConfig | None = None,
        _load_visual_runtime: bool = True,
        linear_state_snapshot_slots: int = 0,
        **kwargs: Any,
    ):
        """
        Args:
            gdn_chunk_size (Optional[int]): GatedDeltaNet prefill sub-chunk size. Each prefill window is
                split into `prefill_chunk_size // gdn_chunk_size` sub-chunks processed by the chunked
                delta rule. Must divide `prefill_chunk_size`. `None` -> `prefill_chunk_size` (no split).
            linear_attention_layers (list[int] | None): The linear_attention (GatedDeltaNet)
                layer indices, populated automatically from `layer_types` at compile time (not user-set).
            visual (Optional[RBLNModelConfig]): Configuration for the vision encoder submodule.
            _load_visual_runtime (bool): Whether to create the visual encoder runtime (False on
                decoder-only nodes in a disaggregated setup). Defaults to True.
            linear_state_snapshot_slots (int): Number K of extra GatedDeltaNet state rows used as prefix
                snapshots for hybrid prefix caching. Defaults to 0 (off). With K > 0 every `conv_state_*` /
                `recurrent_state_*` cache has `batch_size + K` rows (snapshot `s` is row `batch_size + s`),
                the prefill graph takes an extra `state_src_idx` input (the row a window reads its carried
                state from; `batch_idx` stays the row it writes) and `decoder_batch_sizes` must be
                `[batch_size]`. With K = 0 the compiled graphs are unchanged.
            kwargs: Additional arguments passed to `RBLNDecoderOnlyModelConfig`.

        Raises:
            ValueError: If `use_inputs_embeds` is False.
            ValueError: If `linear_state_snapshot_slots` is not an integer >= 0, or is > 0 while
                `decoder_batch_sizes` != [batch_size].
        """
        super().__init__(**kwargs)
        if not getattr(self, "use_inputs_embeds", True):
            raise ValueError(
                "RBLNQwen3_5ModelConfig requires use_inputs_embeds=True. "
                "The visual encoder output must be injected into inputs_embeds."
            )
        # The vision encoder runs one image at a time, so force batch_size=1 on the submodule.
        self.visual = self.initialize_submodule_config(submodule_config=visual, force_kwargs=True, batch_size=1)
        self._load_visual_runtime = _load_visual_runtime
        self.gdn_chunk_size = gdn_chunk_size
        self.linear_attention_layers = linear_attention_layers or []
        self.linear_state_snapshot_slots = linear_state_snapshot_slots
        _validate_linear_state_snapshot_slots(self)


class RBLNQwen3_5ForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """
    Configuration for `RBLNQwen3_5ForConditionalGeneration` (vision-language).

    Qwen3.5 pairs a Qwen3-VL-style vision encoder (no deepstack) with the hybrid Qwen3.5
    text backbone (`linear_attention` GatedDeltaNet layers + `full_attention` gated layers).
    The vision encoder output is injected into `inputs_embeds` (`use_inputs_embeds=True`).

    Independent of the Qwen3-VL config: inherits `RBLNDecoderOnlyModelForCausalLMConfig` directly
    (like the Qwen3-VL config does), carrying its own `visual` submodule handling plus the
    Qwen3.5-specific `gdn_chunk_size`.

    Example usage:
    ```python
    from optimum.rbln import RBLNQwen3_5ForConditionalGeneration

    model = RBLNQwen3_5ForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3.5-...", export=True,
        rbln_config={"max_seq_len": 32768, "tensor_parallel_size": 4, "visual": {"max_seq_len": 6400}},
    )
    ```
    """

    submodules = ["visual"]
    subclass_non_save_attributes = ["_load_visual_runtime", "memory_budget"]

    def __init__(
        self,
        gdn_chunk_size: int | None = None,
        linear_attention_layers: list[int] | None = None,
        use_inputs_embeds: bool = True,
        visual: RBLNModelConfig | None = None,
        _load_visual_runtime: bool = True,
        linear_state_snapshot_slots: int = 0,
        **kwargs: Any,
    ):
        """
        Args:
            gdn_chunk_size (Optional[int]): GatedDeltaNet prefill sub-chunk size. Each prefill window is
                split into `prefill_chunk_size // gdn_chunk_size` sub-chunks. Must divide
                `prefill_chunk_size`. `None` -> `prefill_chunk_size` (no split). See rbln_chunk_gated_delta_rule.
            linear_attention_layers (list[int] | None): The linear_attention (GatedDeltaNet)
                layer indices, populated automatically from `layer_types` at compile time (not user-set).
            use_inputs_embeds (bool): Must be True — the vision encoder output is injected into inputs_embeds.
            visual (Optional[RBLNModelConfig]): Configuration for the vision encoder submodule.
            _load_visual_runtime (bool): Whether to create the visual encoder runtime. Set False on
                decoder-only nodes in a disaggregated setup (then pre-computed image_embeds must be fed to
                forward()). Defaults to True.
            linear_state_snapshot_slots (int): Number K of extra GatedDeltaNet state rows used as prefix
                snapshots for hybrid prefix caching. Defaults to 0 (off). With K > 0 every `conv_state_*` /
                `recurrent_state_*` cache has `batch_size + K` rows (snapshot `s` is row `batch_size + s`),
                the prefill graph takes an extra `state_src_idx` input (the row a window reads its carried
                state from; `batch_idx` stays the row it writes) and `decoder_batch_sizes` must be
                `[batch_size]`. With K = 0 the compiled graphs are unchanged.
            kwargs: Additional arguments passed to `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `use_inputs_embeds` is False.
            ValueError: If `linear_state_snapshot_slots` is not an integer >= 0, or is > 0 while
                `decoder_batch_sizes` != [batch_size].
        """
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNQwen3_5ForConditionalGenerationConfig requires use_inputs_embeds=True. "
                "The visual encoder output must be injected into inputs_embeds."
            )
        # The vision encoder runs one image at a time, so force batch_size=1 on the submodule.
        self.visual = self.initialize_submodule_config(submodule_config=visual, force_kwargs=True, batch_size=1)
        self._load_visual_runtime = _load_visual_runtime
        self.gdn_chunk_size = gdn_chunk_size
        self.linear_attention_layers = linear_attention_layers or []
        self.linear_state_snapshot_slots = linear_state_snapshot_slots
        _validate_linear_state_snapshot_slots(self)
