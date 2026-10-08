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

import inspect
from collections.abc import Callable
from typing import Any

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeModel,
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeVisionModel,
)

from ..qwen3_5.modeling_qwen3_5 import (
    RBLNQwen3_5ForCausalLM,
    RBLNQwen3_5ForConditionalGeneration,
    RBLNQwen3_5Model,
    RBLNQwen3_5VisionModel,
)
from .configuration_qwen3_5_moe import (
    RBLNQwen3_5MoeForCausalLMConfig,  # noqa: F401
    RBLNQwen3_5MoeForConditionalGenerationConfig,  # noqa: F401
    RBLNQwen3_5MoeModelConfig,  # noqa: F401
    RBLNQwen3_5MoeVisionModelConfig,  # noqa: F401
)
from .qwen3_5_moe_architecture import Qwen3_5Moe_CausalLMWrapper, Qwen3_5Moe_LanguageModelWrapper


class RBLNQwen3_5MoeVisionModel(RBLNQwen3_5VisionModel):
    """Qwen3.5-MoE vision encoder. The vision tower is identical to dense Qwen3.5 (no deepstack)."""

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Qwen3_5MoeVisionModel, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val


class RBLNQwen3_5MoeForCausalLM(RBLNQwen3_5ForCausalLM):
    """
    RBLNQwen3_5MoeForCausalLM is the text-only variant of Qwen3.5-MoE (e.g. Qwen3.5-35B-A3B, Qwen3.6-35B-A3B),
    optimized for RBLN NPUs. It runs the hybrid Qwen3.5 decoder with sparse MoE MLPs (routed experts lowered to
    the RBLN `custom_moe_glu` op + a sigmoid-gated shared expert), without the vision encoder.

    Examples:
        ```python
        from optimum.rbln import RBLNQwen3_5MoeForCausalLM

        model = RBLNQwen3_5MoeForCausalLM.from_pretrained(
            "Qwen/Qwen3.6-35B-A3B",
            export=True,
            rbln_config={"num_devices": 16, "kvcache_partition_len": 16384, "max_seq_len": 262144},
        )
        model.save_pretrained("compiled-qwen3.6-35b-a3b-text")
        ```
    """

    _decoder_wrapper_cls = Qwen3_5Moe_CausalLMWrapper


class RBLNQwen3_5MoeModel(RBLNQwen3_5Model):
    _decoder_wrapper_cls = Qwen3_5Moe_LanguageModelWrapper
    _config_class = Qwen3_5MoeConfig
    _rotary_emb_class = Qwen3_5MoeTextRotaryEmbedding
    _get_rope_index_func = Qwen3_5MoeModel.get_rope_index
    get_vision_position_ids = Qwen3_5MoeModel.get_vision_position_ids


class RBLNQwen3_5MoeForConditionalGeneration(RBLNQwen3_5ForConditionalGeneration):
    """
    RBLNQwen3_5MoeForConditionalGeneration is the vision-language Qwen3.5-MoE model (e.g. Qwen3.5-35B-A3B,
    Qwen3.6-35B-A3B), optimized for RBLN NPUs. It pairs the Qwen3.5 vision encoder with the hybrid Qwen3.5 text
    backbone whose MLPs are sparse MoE blocks (routed experts + sigmoid-gated shared expert).

    Everything except the MoE block (vision encoder, GatedDeltaNet / gated attention layers, linear-state runtime,
    mRoPE handling) is shared with [`RBLNQwen3_5ForConditionalGeneration`].

    Examples:
        ```python
        from optimum.rbln import RBLNQwen3_5MoeForConditionalGeneration

        model = RBLNQwen3_5MoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3.6-35B-A3B",
            export=True,
            rbln_config={
                "visual": {"num_devices": 16, "max_seq_len": 16384, "create_runtimes": False},
                "num_devices": 16,
                "kvcache_partition_len": 16384,
                "max_seq_len": 262144,
                "create_runtimes": False,
            },
        )
        model.save_pretrained("qwen3.6-35b-a3b")
        ```
    """

    _decoder_wrapper_cls = Qwen3_5Moe_LanguageModelWrapper
    _config_class = Qwen3_5MoeConfig
    _rotary_emb_class = Qwen3_5MoeTextRotaryEmbedding
    _get_rope_index_func = Qwen3_5MoeModel.get_rope_index
    get_vision_position_ids = Qwen3_5MoeModel.get_vision_position_ids
