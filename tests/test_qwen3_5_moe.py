"""CPU-only checks for the Qwen3.5-MoE (qwen3_5_moe) RBLN wrappers.

The routed experts go through `torch.ops.rbln_custom_ops.custom_moe_glu`, whose CPU implementation is registered
by rebel-compiler, so the wrapper can be compared against the HF eager MoE block without an NPU.
"""

import pytest
import rebel  # noqa: F401  (registers rbln_custom_ops)
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeSparseMoeBlock

from optimum.rbln.transformers.models.qwen3_5_moe.qwen3_5_moe_architecture import (
    Qwen3_5MoeSparseMoeBlock as RBLNQwen3_5MoeSparseMoeBlock,
)
from optimum.rbln.transformers.models.qwen3_5_moe.qwen3_5_moe_architecture import _wrap_moe_block


def _tiny_text_config():
    return Qwen3_5MoeTextConfig(
        hidden_size=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=48,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=128,
    )


@pytest.mark.parametrize("num_tokens", [1, 7, 128])
def test_sparse_moe_block_matches_hf(num_tokens):
    torch.manual_seed(0)
    config = _tiny_text_config()
    hf_block = Qwen3_5MoeSparseMoeBlock(config).float().eval()
    for param in hf_block.parameters():
        torch.nn.init.normal_(param, 0.0, 0.1)

    rbln_block = RBLNQwen3_5MoeSparseMoeBlock(hf_block).eval()
    hidden_states = torch.randn(1, num_tokens, config.hidden_size)
    with torch.no_grad():
        expected = hf_block(hidden_states)
        actual = rbln_block(hidden_states)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_wrap_moe_block_replaces_only_sparse_block():
    config = _tiny_text_config()
    layer = Qwen3_5MoeDecoderLayer(config, layer_idx=0)
    wrapped = _wrap_moe_block(layer.mlp)
    assert isinstance(wrapped, RBLNQwen3_5MoeSparseMoeBlock)
    assert wrapped.top_k == config.num_experts_per_tok
    assert wrapped.experts.gate_proj.weight.shape == (
        config.num_experts,
        config.moe_intermediate_size,
        config.hidden_size,
    )
    # a non-MoE module is passed through untouched
    dense = torch.nn.Linear(4, 4)
    assert _wrap_moe_block(dense) is dense
