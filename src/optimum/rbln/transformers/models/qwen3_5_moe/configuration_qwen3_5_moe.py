"""RBLN configuration classes for Qwen3.5-35B-A3B (MoE with hybrid DeltaNet + Full Attention).

Based on optimum-rbln's Qwen3-VL-MoE configuration, adapted for Qwen3.5's
hybrid attention architecture (30 DeltaNet linear attention + 10 full attention layers).
"""

from typing import Any, List, Optional, Union

from optimum.rbln.configuration_utils import RBLNModelConfig
from optimum.rbln.transformers.models.decoderonly.configuration_decoderonly import (
    RBLNDecoderOnlyModelConfig,
    RBLNDecoderOnlyModelForCausalLMConfig,
)


class RBLNQwen3_5MoeForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """Configuration for RBLNQwen3_5MoeForConditionalGeneration.

    Extends the decoder-only config with vision encoder support and
    hybrid attention state management (DeltaNet recurrent/conv states).
    """

    submodules = ["visual"]

    def __init__(
        self,
        use_inputs_embeds: bool = True,
        visual: Optional[RBLNModelConfig] = None,
        **kwargs: Any,
    ):
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNQwen3_5MoeForConditionalGenerationConfig does not allow `use_inputs_embeds` "
                "to be set to False, as the model accepts only `inputs_embeds` as input."
            )
        self.visual = visual


class RBLNQwen3_5MoeModelConfig(RBLNDecoderOnlyModelConfig):
    submodules = ["visual"]

    def __init__(self, visual: Optional[RBLNModelConfig] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.visual = self.initialize_submodule_config(submodule_config=visual)


class RBLNQwen3_5MoeVisionModelConfig(RBLNModelConfig):
    """Configuration for the Qwen3.5-MoE vision encoder.

    Identical structure to Qwen3-VL vision encoder (27 blocks, full attention only).
    """

    def __init__(self, max_seq_lens: Union[int, List[int]] = None, **kwargs: Any):
        super().__init__(**kwargs)

        if max_seq_lens is not None:
            if isinstance(max_seq_lens, int):
                max_seq_lens = [max_seq_lens]
            elif isinstance(max_seq_lens, list):
                max_seq_lens.sort(reverse=True)
        else:
            raise ValueError("'max_seq_lens' must be specified.")

        self.max_seq_lens = max_seq_lens
