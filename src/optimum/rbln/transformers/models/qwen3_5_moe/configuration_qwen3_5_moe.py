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

from ..qwen3_5.configuration_qwen3_5 import (
    RBLNQwen3_5ForCausalLMConfig,
    RBLNQwen3_5ForConditionalGenerationConfig,
    RBLNQwen3_5ModelConfig,
    RBLNQwen3_5VisionModelConfig,
)


class RBLNQwen3_5MoeForCausalLMConfig(RBLNQwen3_5ForCausalLMConfig):
    """
    Configuration class for the text-only RBLN Qwen3.5-MoE (e.g. Qwen3.5-35B-A3B, Qwen3.6-35B-A3B) causal LM.

    Qwen3.5-MoE shares the hybrid Qwen3.5 decoder (GatedDeltaNet `linear_attention` layers interleaved with
    gated `full_attention` layers) and only replaces the dense MLP with a sparse MoE block (routed experts +
    sigmoid-gated shared expert). All options are therefore identical to `RBLNQwen3_5ForCausalLMConfig`.
    """


class RBLNQwen3_5MoeVisionModelConfig(RBLNQwen3_5VisionModelConfig):
    """Configuration for the Qwen3.5-MoE vision encoder. Identical to `RBLNQwen3_5VisionModelConfig`."""


class RBLNQwen3_5MoeModelConfig(RBLNQwen3_5ModelConfig):
    """Configuration for `RBLNQwen3_5MoeModel`. Identical to `RBLNQwen3_5ModelConfig`."""


class RBLNQwen3_5MoeForConditionalGenerationConfig(RBLNQwen3_5ForConditionalGenerationConfig):
    """
    Configuration for `RBLNQwen3_5MoeForConditionalGeneration` (vision-language, e.g. Qwen3.6-35B-A3B).

    Identical to `RBLNQwen3_5ForConditionalGenerationConfig`; the MoE block is lowered to the RBLN
    `custom_moe_glu` op inside the compiled language model graph, so no MoE-specific options are needed.

    Example usage:
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
    ```
    """
