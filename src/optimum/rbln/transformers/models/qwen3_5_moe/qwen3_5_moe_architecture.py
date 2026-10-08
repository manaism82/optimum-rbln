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

import torch
from torch import nn

from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import DecoderOnlyAttention, DecoderOnlyLayer
from ..qwen2_moe.qwen2_moe_architecture import Qwen2MoeMLP
from ..qwen3_5.qwen3_5_architecture import (
    Qwen3_5_CausalLMWrapper,
    Qwen3_5_LanguageModelWrapper,
    Qwen3_5GatedDeltaNet,
    Qwen3_5LinearDecoderLayer,
)


_HF_SPARSE_MOE_BLOCK_NAME = "Qwen3_5MoeSparseMoeBlock"


def _wrap_moe_block(mlp: nn.Module) -> nn.Module:
    return Qwen3_5MoeSparseMoeBlock(mlp) if mlp.__class__.__name__ == _HF_SPARSE_MOE_BLOCK_NAME else mlp


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Routed experts (`custom_moe_glu`) + sigmoid-gated shared expert.

    HF `Qwen3_5MoeTopKRouter` always renormalizes the top-k probabilities (softmax -> topk -> /sum), which is
    mathematically identical to topk -> softmax-of-topk (`renormalize=True`). The router has no
    `norm_topk_prob` attribute, so `Qwen2MoeSparseMoeBlock` cannot be reused directly.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.num_experts = model.gate.num_experts
        self.top_k = model.gate.top_k
        gate_weight = model.gate.weight
        gate = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
        gate.weight = nn.Parameter(gate_weight.detach().clone())
        self.gate = gate
        self.experts = Qwen2MoeMLP(model.experts, self.top_k, norm_topk_prob=True)
        self.shared_expert = model.shared_expert
        self.shared_expert_gate = model.shared_expert_gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            torch.nn.functional.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )
        final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class Qwen3_5MoeLayer(DecoderOnlyLayer):
    """`full_attention` decoder layer with the sparse MoE block."""

    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: RBLNLoRAConfig | None = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = _wrap_moe_block(layer.mlp)

    def get_mlp(self) -> nn.Module:
        return self.mlp


class Qwen3_5MoeLinearDecoderLayer(Qwen3_5LinearDecoderLayer):
    """`linear_attention` (GatedDeltaNet) decoder layer with the sparse MoE block."""

    def __init__(self, layer: nn.Module, linear_attn: Qwen3_5GatedDeltaNet):
        super().__init__(layer, linear_attn)
        self.mlp = _wrap_moe_block(layer.mlp)


class _Qwen3_5MoeConvertMixin:
    """Builds the hybrid layer stack with MoE-aware layer classes.

    Same as `Qwen3_5_CausalLMWrapper.convert_to_rbln_class` except that linear-attention layers are built with
    `get_rbln_linear_layer_class()` (the qwen3_5 implementation hard-codes `Qwen3_5LinearDecoderLayer`).
    """

    def get_rbln_layer_class(self):
        return Qwen3_5MoeLayer

    def get_rbln_linear_layer_class(self):
        return Qwen3_5MoeLinearDecoderLayer

    def convert_to_rbln_class(self, model, max_seq_len: int, use_rotary_emb: bool):
        layer_types = self.config.layer_types
        new_layers = []
        for layer_idx, layer in enumerate(self.get_decoder_layers(model)):
            if layer_types[layer_idx] == "linear_attention":
                rbln_deltanet = Qwen3_5GatedDeltaNet(layer.linear_attn, self.rbln_config, layer_idx)
                new_layers.append(self.get_rbln_linear_layer_class()(layer, rbln_deltanet))
            else:
                new_self_attn = self.get_rbln_attn_class()(layer.self_attn, self.rbln_config, is_sliding=False)
                new_layers.append(
                    self.get_rbln_layer_class()(layer, new_self_attn, lora_config=self.rbln_config.lora_config)
                )

        new_model = self.get_rbln_model_class()(
            self.get_model_layer(model),
            new_layers,
            self.rbln_config,
            use_learned_pos_emb=self.__class__._use_learned_pos_emb,
            use_rotary_emb=use_rotary_emb,
        )
        if self.is_causal_lm:
            return self.get_rbln_causal_lm_class()(model, new_model)
        return new_model


class Qwen3_5Moe_CausalLMWrapper(_Qwen3_5MoeConvertMixin, Qwen3_5_CausalLMWrapper):
    pass


class Qwen3_5Moe_LanguageModelWrapper(_Qwen3_5MoeConvertMixin, Qwen3_5_LanguageModelWrapper):
    pass
