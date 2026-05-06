"""Qwen3.5-35B-A3B architecture wrappers for RBLN compilation.

Hybrid architecture:
- Vision encoder: 27 blocks, full attention (identical to Qwen3-VL, no DeepStack)
- Language model: 40 layers with hybrid attention
  - 30 DeltaNet linear attention layers (layer_types="linear_attention")
  - 10 Full attention layers (layer_types="full_attention")
  - All layers use MoE (256 experts + 1 shared expert)

Based on:
- Qwen3-VL-30B-A3B-Instruct/rbln_qwen3_vl_moe/qwen3_vl_moe_architecture.py
- optimum.rbln DecoderOnly framework (for full attention layers)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from optimum.rbln.transformers.models.decoderonly.configuration_decoderonly import RBLNLoRAConfig
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_partial,
)
from optimum.rbln.transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    RBLNQwen2_5_VisionTransformerPretrainedModelConfig,
)


# DeltaNet chunk-wise gated delta-rule chunk size.
# Must divide `prefill_chunk_size` (RBLN framework requires prefill_chunk_size to
# be a positive multiple of 64; the FLA reference also uses 64). Increasing it
# reduces the outer-loop unroll count but enlarges per-chunk matmuls; decreasing
# it does the inverse. 64 is the same default the transformers reference uses.
DELTANET_CHUNK_SIZE = 64


# ==============================================================================
# Vision Encoder (identical to Qwen3-VL, no DeepStack)
# ==============================================================================


class Qwen3_5VisionTransformerWrapper(nn.Module):
    """Qwen3.5 Vision Transformer for RBLN compilation.

    Same architecture as Qwen3-VL vision encoder (27 blocks, full attention).
    No DeepStack (deepstack_visual_indexes=[]).
    """

    def __init__(self, model: nn.Module, rbln_config):
        super().__init__()
        self.merger = model.merger
        self.rbln_config = rbln_config
        self.blocks = nn.ModuleList([
            Qwen3_5VisionBlock(block, rbln_config) for block in model.blocks
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        attn_masks = (1.0 - attn_masks) * torch.finfo(hidden_states.dtype).min

        for block in self.blocks:
            hidden_states = block(hidden_states, attn_masks, [cos, sin])

        hidden_states = self.merger(hidden_states)
        return hidden_states


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, model: nn.Module, rbln_config):
        super().__init__()
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.attn = Qwen3_5VisionFullAttention(model.attn, rbln_config)
        self.mlp = model.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), attn_masks, position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionFullAttention(nn.Module):
    """Full attention for vision encoder."""

    def __init__(self, model: nn.Module, rbln_config) -> None:
        super().__init__()
        self.num_heads = model.num_heads
        self.head_dim = getattr(model, "head_dim", model.proj.in_features // model.num_heads)
        self.qkv = model.qkv
        self.proj = model.proj
        self.scale = torch.tensor(1 / math.sqrt(self.head_dim), dtype=rbln_config.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        hidden_states = hidden_states.unsqueeze(0)
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(1, seq_length, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn_weights = attn_weights + attn_masks
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=hidden_states.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)

        return attn_output


# ==============================================================================
# Language Model - Full Attention (10 layers)
# ==============================================================================


class Qwen3_5MoeFullAttention(DecoderOnlyAttention):
    """Full attention with QK-norm and per-head sigmoid output gate (Qwen3.5).

    `Qwen3_5MoeAttention.q_proj` outputs `num_heads * head_dim * 2` features —
    the first half is the query, the second half is a sigmoid-applied gate that
    multiplies the attention output BEFORE `o_proj`. Reference:
    `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeAttention.forward`.

    The framework's `DecoderOnlyAttention.forward` doesn't know about this gate,
    so we override `forward` to slice q_proj output and apply the gate.
    """

    def __post_init__(self, self_attn=None):
        super().__post_init__(self_attn)
        # QK-norm (RMSNorm on head_dim)
        if hasattr(self_attn, "q_norm"):
            self.q_norm = self_attn.q_norm
        if hasattr(self_attn, "k_norm"):
            self.k_norm = self_attn.k_norm

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        # Qwen3.5 uses partial_rotary_factor=0.25 (64 of head_dim=256 dims).
        # cos/sin are already shaped to the partial dim by Qwen3_5MoeTextRotaryEmbedding.
        return apply_rotary_pos_emb_partial(
            query_states, key_states, cos, sin, ndim=cos.shape[-1]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: torch.LongTensor,
        past_key_values,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
    ):
        batch_size, query_length, _ = hidden_states.size()

        # q_proj outputs num_heads * head_dim * 2; split into [query | gate].
        if self.lora_config:
            qkv_q = self.q_proj(hidden_states, lora_int_id)
            key_states = self.k_proj(hidden_states, lora_int_id)
            value_states = self.v_proj(hidden_states, lora_int_id)
        else:
            qkv_q = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        qkv_q = qkv_q.view(batch_size, query_length, self.num_heads, self.head_dim * 2)
        query_states, gate = torch.chunk(qkv_q, 2, dim=-1)
        gate = gate.reshape(batch_size, query_length, self.num_heads * self.head_dim)

        query_states = query_states.transpose(1, 2)  # [B, H, S, D]
        key_states = key_states.view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if cos is not None and sin is not None:
            query_states, key_states = self.apply_rotary_pos_embed(
                query_states, key_states, cos, sin
            )

        if batch_size > 1 and "prefill" in self.phase:
            raise NotImplementedError(
                f"batch size should be 1 if prefill phase, but got {batch_size}."
            )

        k_scale, v_scale = self.maybe_get_kvcache_scale()

        attn_output = self.get_attention_op()(
            query_states,
            key_states,
            value_states,
            attention_mask,
            past_key_state=past_key_values[self.layer_idx][0],
            past_value_state=past_key_values[self.layer_idx][1],
            seq_position=seq_positions,
            scale=self.scale,
            block_tables=block_tables,
            block_size=self.kvcache_block_size,
            k_scale=k_scale,
            v_scale=v_scale,
            s_aux=getattr(self, "sinks", None),
        )

        # Apply per-head sigmoid gate (Qwen3.5 specific).
        attn_output = attn_output * torch.sigmoid(gate)

        if self.lora_config:
            attn_outputs = self.o_proj(attn_output, lora_int_id)
        else:
            attn_outputs = self.o_proj(attn_output)

        return attn_outputs


# ==============================================================================
# Language Model - DeltaNet Linear Attention (30 layers)
# ==============================================================================


class ManualRMSNorm(nn.Module):
    """RMSNorm without aten::rms_norm (not supported by rebel-compiler)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm).to(x.dtype) * self.weight


class ManualRMSNormGated(nn.Module):
    """Gated RMSNorm: output = RMSNorm(x) * SiLU(gate)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def l2norm(x, dim=-1, eps=1e-6):
    """L2 normalization (FLA-compatible).

    Avoid `torch.norm` / `linalg.vector_norm` because the TVM PyTorch frontend
    asserts the dtype is float32/float64; DeltaNet calls this on bf16 tensors.
    Equivalent to `x / (x.norm(dim=dim, keepdim=True) + eps)` but expressed as
    a multiply by `rsqrt(sum(x*x) + eps)`.
    """
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def chunk_gated_delta_rule(
    query, key, value, g, beta, initial_state, chunk_size: int = DELTANET_CHUNK_SIZE
):
    """Chunk-wise gated delta-rule attention (FLA torch fallback).

    Mirrors `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.torch_chunk_gated_delta_rule`.
    Inputs are [B, S, H, D] (or [B, S, H] for `g`/`beta`); the function transposes
    internally and returns `(core_attn_out: [B, S, H, V], last_recurrent_state)`.
    The outer state-update loop runs `seq_len // chunk_size` iterations — for the
    default `prefill_chunk_size=128` and `chunk_size=64`, that's 2 outer iterations
    per layer instead of 128 sequential single-token updates, which keeps the
    unrolled graph small enough for the RBLN compiler.
    """
    initial_dtype = query.dtype
    # Transpose to [B, H, S, D] / [B, H, S] in fp32 for accumulation stability.
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # Reshape to chunks: [B, H, num_chunks, chunk_size, D]
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # Per-chunk decay solve (size-CHUNK_SIZE constants, independent of seq_len).
    upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    # Strictly lower-triangular `L` from `-K_beta @ K^T * decay_mask`.
    L = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(upper, 0)

    # Triangular solve `T = (I - L)^{-1}` via Neumann series:
    #   T = I + L + L^2 + ... + L^{cs-1}    (since L is strictly lower triangular,
    #                                         L^cs = 0 so the series terminates).
    # Reference's in-place slice update was the same triangular solve via row-fill.
    # We replace it with iterated matmul (chunk_size×chunk_size mats) so the graph
    # has no `aten::index_put_` / `aten::slice_scatter` nodes — those are what we
    # suspect the RBLN partition stage cannot split.
    _idx = torch.arange(chunk_size, device=L.device)
    eye = (_idx.unsqueeze(0) == _idx.unsqueeze(1)).to(L.dtype)
    power = L
    series = L
    for _ in range(chunk_size - 2):
        power = power @ L
        series = series + power
    attn = series + eye

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = initial_state.to(value)
    upper_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    num_chunks = total_sequence_length // chunk_size
    # Outer chunk loop: accumulate per-chunk outputs into a list, stack at end —
    # avoids the `core_attn_out[:, :, i] = ...` in-place slice assignment.
    chunk_outputs = []
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_intra = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(
            upper_strict, 0
        )
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        chunk_out = attn_inter + attn_intra @ v_new
        chunk_outputs.append(chunk_out)
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (
                k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )

    # Stack chunk outputs back into a [B, H, num_chunks, chunk_size, V] tensor,
    # then merge the chunk dim with the chunk_size dim → [B, H, total, V].
    core_attn_out = torch.stack(chunk_outputs, dim=2)
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5MoeDeltaNetAttention(nn.Module):
    """Gated DeltaNet linear attention for RBLN compilation.

    Two compile paths, selected at trace time by `seq_len`:
    - decode (seq_len=1): a single recurrent step (1 unrolled iteration / layer).
    - prefill (seq_len > 1, typically `prefill_chunk_size=128`): chunk-wise via
      `chunk_gated_delta_rule` (chunk_size=64 → seq_len/64 outer iterations and
      a constant chunk-size×chunk_size triangular solve, both far smaller than
      a single-token recurrent unroll over the full prefill chunk).

    State management:
    - conv_state: [batch, conv_dim, kernel_size-1] — causal conv history
    - recurrent_state: [batch, num_v_heads, key_dim, value_dim] — delta-rule state
    """

    def __init__(self, hf_layer, layer_idx, deltanet_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.deltanet_idx = deltanet_idx
        attn = hf_layer.linear_attn

        # Dimensions
        self.hidden_size = attn.hidden_size
        self.num_k_heads = attn.num_k_heads
        self.num_v_heads = attn.num_v_heads
        self.head_k_dim = attn.head_k_dim
        self.head_v_dim = attn.head_v_dim
        self.key_dim = attn.key_dim
        self.value_dim = attn.value_dim
        self.conv_dim = attn.conv_dim
        self.conv_kernel_size = attn.conv_kernel_size

        # Projections
        self.in_proj_qkv = attn.in_proj_qkv
        self.in_proj_z = attn.in_proj_z
        self.in_proj_b = attn.in_proj_b
        self.in_proj_a = attn.in_proj_a
        self.out_proj = attn.out_proj

        # Conv weights — clone to avoid inference tensor issues
        self.conv_weight = nn.Parameter(attn.conv1d.weight.data.squeeze(1).clone())  # [conv_dim, kernel_size]
        self.conv_bias = nn.Parameter(attn.conv1d.bias.data.clone()) if attn.conv1d.bias is not None else None

        # Decay parameters — clone to detach from inference mode
        self.A_log = nn.Parameter(attn.A_log.data.clone())
        self.dt_bias = nn.Parameter(attn.dt_bias.data.clone())

        # Gated normalization (use manual impl for RBLN compatibility)
        self.norm = ManualRMSNormGated(self.head_v_dim, eps=attn.layer_norm_epsilon)
        if hasattr(attn.norm, 'weight'):
            self.norm.weight.data.copy_(attn.norm.weight.data)

        # Ratio for repeating k heads to match v heads
        self.kv_head_ratio = self.num_v_heads // self.num_k_heads

        # Chunk size for the chunk-wise gated delta-rule path used during prefill.
        # Pulled from the module-level constant so a single edit point governs it.
        self.delta_chunk_size = DELTANET_CHUNK_SIZE

    def forward(self, hidden_states, conv_state, recurrent_state):
        """DeltaNet attention. Handles both decode (seq_len=1) and prefill (seq_len>1).

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            conv_state: [batch, conv_dim, kernel_size-1]
            recurrent_state: [batch, num_v_heads, key_dim, value_dim]

        Returns:
            output: [batch, seq_len, hidden_size]
            new_conv_state: [batch, conv_dim, kernel_size-1]
            new_recurrent_state: [batch, num_v_heads, key_dim, value_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        conv_dim = self.conv_dim
        kernel_size = self.conv_kernel_size
        state_len = kernel_size - 1

        # ---- Projections --------------------------------------------------
        mixed_qkv = self.in_proj_qkv(hidden_states)            # [B, S, conv_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2)                  # [B, conv_dim, S]

        z = self.in_proj_z(hidden_states)                      # [B, S, value_dim]
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)                      # [B, S, num_v_heads]
        a = self.in_proj_a(hidden_states)                      # [B, S, num_v_heads]

        # ---- Multi-step causal conv1d -------------------------------------
        # Equivalent to F.conv1d(cat(state, x), w, b, groups=conv_dim) but expressed
        # as gather-and-multiply on a sliding window. Replaces `groups=conv_dim` (i.e.
        # depthwise conv) which the RBLN partition stage appears unable to split
        # across devices when conv_dim is large (8192 here). The rewrite uses only
        # cat / gather / multiply / sum-reduce, which are partition-friendly.
        # New conv_state = last (kernel_size-1) frames of the concatenated input.
        conv_input = torch.cat(
            [conv_state.to(mixed_qkv.dtype), mixed_qkv], dim=-1
        )                                                       # [B, C, S+state_len]
        new_conv_state = conv_input[:, :, -state_len:]

        # Build a sliding window manually: stack `kernel_size` shifted views of the
        # input along a new last axis, giving [B, C, S, K]. We avoid `Tensor.unfold`
        # in case it lowers to an op the RBLN frontend does not support — `narrow`
        # + `stack` decomposes into standard slice + cat ops.
        windows = torch.stack(
            [
                conv_input[:, :, k : k + seq_len]
                for k in range(kernel_size)
            ],
            dim=-1,
        )                                                       # [B, C, S, K]
        # `conv_weight` has shape [C, K]; broadcast along batch and seq dims.
        mixed_qkv_conv = (windows * self.conv_weight.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        if self.conv_bias is not None:
            mixed_qkv_conv = mixed_qkv_conv + self.conv_bias.view(1, -1, 1)
        mixed_qkv_conv = F.silu(mixed_qkv_conv)                  # [B, C, S]
        mixed_qkv_conv = mixed_qkv_conv.transpose(1, 2)          # [B, S, C]

        # ---- Split QKV ----------------------------------------------------
        query, key, value = torch.split(
            mixed_qkv_conv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()                                      # [B, S, num_v_heads]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.kv_head_ratio > 1:
            query = query.repeat_interleave(self.kv_head_ratio, dim=2)
            key = key.repeat_interleave(self.kv_head_ratio, dim=2)

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        # ---- Gated delta rule -------------------------------------------------
        # Branch on the trace-time constant `seq_len`:
        # - decode (seq_len == 1): one recurrent step, tiny graph
        # - prefill (seq_len > 1): chunk-wise (jit.trace unrolls the chunk loop
        #   `seq_len // chunk_size` times instead of `seq_len` times, which keeps
        #   the device-graph small enough for tensor parallelism splitting)
        if seq_len == 1:
            # query/key/value: [B, S, H, D] → [B, H, S=1, D] in fp32
            query_t = query.transpose(1, 2).contiguous().to(torch.float32)
            key_t = key.transpose(1, 2).contiguous().to(torch.float32)
            value_t = value.transpose(1, 2).contiguous().to(torch.float32)
            beta_t = beta.transpose(1, 2).contiguous().to(torch.float32)  # [B, H, 1]
            g_t = g.transpose(1, 2).contiguous()                          # [B, H, 1]

            scale = 1.0 / math.sqrt(self.head_k_dim)
            q_step = query_t[:, :, 0] * scale       # [B, H, key_dim]
            k_step = key_t[:, :, 0]
            v_step = value_t[:, :, 0]
            g_step = g_t[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            b_step = beta_t[:, :, 0].unsqueeze(-1)                   # [B, H, 1]

            new_recurrent_state = recurrent_state.to(torch.float32) * g_step
            kv_mem = (new_recurrent_state * k_step.unsqueeze(-1)).sum(dim=-2)
            delta = (v_step - kv_mem) * b_step
            new_recurrent_state = (
                new_recurrent_state + k_step.unsqueeze(-1) * delta.unsqueeze(-2)
            )
            o_step = (new_recurrent_state * q_step.unsqueeze(-1)).sum(dim=-2)
            core_attn_out = o_step.unsqueeze(2)  # [B, H, 1, value_dim]
            core_attn_out = core_attn_out.transpose(1, 2).contiguous()  # [B, 1, H, V]
            core_attn_out = core_attn_out.to(hidden_states.dtype)
        else:
            core_attn_out, new_recurrent_state = chunk_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=recurrent_state.to(torch.float32),
                chunk_size=self.delta_chunk_size,
            )
            core_attn_out = core_attn_out.to(hidden_states.dtype)

        # ---- Gated RMSNorm + output projection --------------------------------
        # core_attn_out: [B, S, H, head_v_dim] → flatten for norm
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)

        return output, new_conv_state, new_recurrent_state.to(hidden_states.dtype)


# ==============================================================================
# Language Model - MoE with Shared Expert
# ==============================================================================


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """MoE block with 256 routed experts + 1 shared expert + sigmoid gate."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.gate = model.gate  # TopKRouter
        self.shared_expert = model.shared_expert  # MLP
        self.shared_expert_gate = model.shared_expert_gate  # Linear(hidden→1)

        # Routed experts: convert to custom_moe_glu format
        self.routed_experts = Qwen3_5MoeRoutedExperts(model.experts, model.gate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_2d = hidden_states.view(-1, hidden_dim)

        # `custom_moe_glu` performs softmax + topk + (optional) norm internally,
        # so we forward the raw router logits and discard the precomputed weights
        # / indices that `Qwen3_5MoeTopKRouter` returns.
        router_logits, _, _ = self.gate(hidden_states_2d)
        routed_output = self.routed_experts(hidden_states_2d, router_logits)

        # Shared expert with sigmoid gate
        shared_output = self.shared_expert(hidden_states_2d)
        shared_gate = F.sigmoid(self.shared_expert_gate(hidden_states_2d))
        shared_output = shared_gate * shared_output

        output = routed_output + shared_output
        return output.reshape(batch_size, sequence_length, hidden_dim)


class Qwen3_5MoeRoutedExperts(nn.Module):
    """Routed experts using RBLN custom_moe_glu op.

    Qwen3.5 expert weight format:
    - gate_up_proj: [num_experts, 2*intermediate_dim, hidden_dim] (nn.Parameter)
    - down_proj: [num_experts, hidden_dim, intermediate_dim] (nn.Parameter)

    custom_moe_glu expects Linear.weight format:
    - gate_proj: [num_experts, intermediate_dim, hidden_dim]
    - up_proj:   [num_experts, intermediate_dim, hidden_dim]
    - down_proj:  [num_experts, hidden_dim, intermediate_dim]
    """

    def __init__(self, experts, gate):
        super().__init__()
        self.num_experts = experts.num_experts
        self.top_k = gate.top_k
        self.norm_topk_prob = True  # Qwen3.5 normalizes topk prob in the router

        hidden_dim = experts.hidden_dim
        intermediate_dim = experts.intermediate_dim

        # Split fused gate_up_proj: [num_experts, 2*intermediate_dim, hidden_dim]
        gate_up = experts.gate_up_proj.data
        gate_weight, up_weight = gate_up.chunk(2, dim=1)
        # gate_weight/up_weight: [num_experts, intermediate_dim, hidden_dim]
        # Already in Linear.weight format (out_features, in_features) — no transpose needed

        self.gate_proj = nn.Linear(hidden_dim, self.num_experts * intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, self.num_experts * intermediate_dim, bias=False)
        self.down_proj = nn.Linear(self.num_experts * intermediate_dim, hidden_dim, bias=False)

        self.gate_proj.weight.data = gate_weight.contiguous()
        self.up_proj.weight.data = up_weight.contiguous()
        # down_proj: [num_experts, hidden_dim, intermediate_dim] — already correct
        self.down_proj.weight.data = experts.down_proj.data.contiguous()

    def forward(self, x, router_logits):
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            router_logits,
            self.top_k,
            self.norm_topk_prob,
        )


# ==============================================================================
# Language Model - Hybrid Decoder Layers
# ==============================================================================


class Qwen3_5MoeFullAttnLayer(DecoderOnlyLayer):
    """Decoder layer with full attention + MoE."""

    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config=None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = Qwen3_5MoeSparseMoeBlock(layer.mlp)

    def get_mlp(self) -> nn.Module:
        return self.mlp


class Qwen3_5MoeDeltaNetLayer(nn.Module):
    """Decoder layer with DeltaNet linear attention + MoE.

    This layer does NOT use KV-cache. Instead it uses recurrent_state and conv_state.
    """

    def __init__(self, layer, deltanet_attn: Qwen3_5MoeDeltaNetAttention):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.deltanet_attn = deltanet_attn
        self.mlp = Qwen3_5MoeSparseMoeBlock(layer.mlp)

    @property
    def phase(self):
        return "decode"

    @phase.setter
    def phase(self, value):
        pass  # DeltaNet layers don't distinguish phases in the same way

    def forward(
        self,
        hidden_states,
        conv_state,
        recurrent_state,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, new_conv_state, new_recurrent_state = self.deltanet_attn(
            hidden_states, conv_state, recurrent_state
        )

        hidden_states = residual + hidden_states

        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_conv_state, new_recurrent_state


# ==============================================================================
# Hybrid Decoder Model
# ==============================================================================


class Qwen3_5MoeHybridDecoderModel(DecoderOnlyModel):
    """Hybrid decoder model with both full attention and DeltaNet layers.

    Full attention layers (10) use KV-cache via past_key_values.
    DeltaNet layers (30) use recurrent/conv state passed as extra inputs.
    """

    def __init__(self, *args, layer_types=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_types = layer_types or []

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        cache_position=None,
        position_ids=None,
        query_position=None,
        past_key_values=None,
        rotary_emb=None,
        global_block_tables=None,
        local_block_tables=None,
        lora_int_id=None,
        output_hidden_states=None,
        # DeltaNet states
        all_conv_states=None,
        all_recurrent_states=None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_embedding()(input_ids)

        hidden_states = inputs_embeds * self.hidden_multiplier

        position_ids = position_ids if position_ids is not None else cache_position
        if rotary_emb is not None:
            if isinstance(rotary_emb, torch.Tensor):
                cos = rotary_emb[0]
                sin = rotary_emb[1]
            else:
                cos, sin = rotary_emb(hidden_states, self.max_seq_len)
                from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
                    slice_and_unsqueeze_cos_sin,
                )
                cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, position_ids)
        else:
            cos, sin = None, None

        if self.attn_impl == "flash_attn":
            seq_positions = cache_position[:, 0]
            seq_positions = self.convert_sequence_positions_for_flash_attn(
                seq_positions=seq_positions, max_seq_len=self.max_seq_len
            )
        else:
            seq_positions = cache_position[:, :1]

        all_hidden_states = () if output_hidden_states else None
        new_conv_states = []
        new_recurrent_states = []
        deltanet_idx = 0

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.layer_types[layer_idx] == "full_attention":
                # Full attention layer: use KV-cache
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    seq_positions=seq_positions,
                    past_key_values=past_key_values,
                    cos=cos,
                    sin=sin,
                    block_tables=global_block_tables,
                    lora_int_id=lora_int_id,
                )
            else:
                # DeltaNet layer: use recurrent/conv state
                conv_state = all_conv_states[deltanet_idx]
                recurrent_state = all_recurrent_states[deltanet_idx]

                hidden_states, new_conv, new_rec = layer(
                    hidden_states=hidden_states,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                )
                new_conv_states.append(new_conv)
                new_recurrent_states.append(new_rec)
                deltanet_idx += 1

        hidden_states = self.get_last_layernorm()(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states, new_conv_states, new_recurrent_states


class Qwen3_5MoeHybridForCausalLM(DecoderOnlyForCausalLM):
    """CausalLM wrapper for hybrid decoder."""

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        cache_position=None,
        position_ids=None,
        query_position=None,
        past_key_values=None,
        rotary_emb=None,
        global_block_tables=None,
        local_block_tables=None,
        lora_int_id=None,
        output_hidden_states=None,
        all_conv_states=None,
        all_recurrent_states=None,
    ):
        hidden_states, all_hidden_states, new_conv_states, new_recurrent_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=output_hidden_states,
            all_conv_states=all_conv_states,
            all_recurrent_states=all_recurrent_states,
        )

        if "prefill" in self.phase and query_position is not None:
            hidden_states = hidden_states[:, query_position.to(torch.int).unsqueeze(0)]

        logits = self.lm_head(hidden_states)

        return logits, all_hidden_states, new_conv_states, new_recurrent_states


# ==============================================================================
# Language Model Wrapper
# ==============================================================================


class Qwen3_5Moe_LanguageModelWrapper(DecoderOnlyWrapper):
    """Language model wrapper for Qwen3.5-MoE hybrid architecture.

    Manages both KV-cache (for full attention) and DeltaNet state.
    """

    def __init__(self, model: PreTrainedModel, rbln_config, use_rotary_emb: bool):
        # Qwen3_5MoeConfig doesn't proxy text_config attributes.
        original_config = model.config
        model.config = original_config.text_config
        self._layer_types = list(original_config.text_config.layer_types)
        self._num_deltanet_layers = self._layer_types.count("linear_attention")
        self._text_config = original_config.text_config
        super().__init__(model, rbln_config, use_rotary_emb)
        model.config = original_config

        # Fix inference tensors: rebel.compile_from_torch requires version-tracked tensors
        self._fix_inference_tensors()

    def _fix_inference_tensors(self):
        """Replace inference tensors with regular tensors in all parameters.

        This runs inside torch.inference_mode() context (from get_compiled_model),
        so we must explicitly exit inference mode to create non-inference clones.
        """
        with torch.inference_mode(False):
            for name, param in list(self.named_parameters()):
                if param.data.is_inference():
                    param.data = param.data.clone()

    def get_rbln_attn_class(self):
        return Qwen3_5MoeFullAttention

    def get_rbln_layer_class(self):
        return Qwen3_5MoeFullAttnLayer

    def get_rbln_model_class(self):
        return Qwen3_5MoeHybridDecoderModel

    def get_rbln_causal_lm_class(self):
        return Qwen3_5MoeHybridForCausalLM

    def convert_to_rbln_class(self, model, max_seq_len, use_rotary_emb):
        new_layers = []
        attn_kv_idx = 0  # KV-cache index (only for full attention layers)

        for layer_idx, layer in enumerate(self.get_decoder_layers(model)):
            if self._layer_types[layer_idx] == "full_attention":
                # Full attention layer: use existing framework
                # Remap layer_idx for KV-cache access
                original_layer_idx = layer.self_attn.layer_idx
                layer.self_attn.layer_idx = attn_kv_idx
                attn_kv_idx += 1

                new_self_attn = self.get_rbln_attn_class()(
                    layer.self_attn, self.rbln_config, is_sliding=False
                )
                new_layer = self.get_rbln_layer_class()(layer, new_self_attn)
                new_layers.append(new_layer)
            else:
                # DeltaNet layer
                deltanet_idx = sum(
                    1 for t in self._layer_types[:layer_idx] if t == "linear_attention"
                )
                deltanet_attn = Qwen3_5MoeDeltaNetAttention(layer, layer_idx, deltanet_idx)
                new_layer = Qwen3_5MoeDeltaNetLayer(layer, deltanet_attn)
                new_layers.append(new_layer)

        new_model = self.get_rbln_model_class()(
            self.get_model_layer(model),
            new_layers,
            self.rbln_config,
            use_learned_pos_emb=self.__class__._use_learned_pos_emb,
            use_rotary_emb=use_rotary_emb,
            layer_types=self._layer_types,
        )

        if self.is_causal_lm:
            new_model = self.get_rbln_causal_lm_class()(model, new_model)
            return new_model
        else:
            return new_model

    def get_decoder_layers(self, model):
        return model.model.language_model.layers if hasattr(model, "model") else model.language_model.layers

    def get_model_layer(self, model):
        return model.model.language_model if hasattr(model, "model") else model.language_model

    def prepare_forward_args(self, *args):
        args = list(args)
        input_ids = None if self.rbln_config.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.rbln_config.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0)
        local_block_tables = None
        position_embeds = args.pop(0)

        # DeltaNet states: conv_states and recurrent_states
        all_conv_states = [args.pop(0) for _ in range(self._num_deltanet_layers)]
        all_recurrent_states = [args.pop(0) for _ in range(self._num_deltanet_layers)]

        query_position = args.pop(0) if self.phase == "prefill" and self.rbln_config.logits_to_keep > 0 else None
        attention_mask = args.pop(0) if self.rbln_config.use_attention_mask else None
        lora_int_id = args.pop(0) if self.rbln_config.lora_config else None

        # Remaining args are past_key_values (only for full attention layers)
        past_key_values = args
        num_attn_layers = self._layer_types.count("full_attention")
        if len(past_key_values) != 2 * num_attn_layers:
            raise ValueError(
                f"Expected {2 * num_attn_layers} past_key_values, got {len(past_key_values)}"
            )

        _past_key_values = []
        for i in range(num_attn_layers):
            key_states = past_key_values[i * 2]
            value_states = past_key_values[i * 2 + 1]
            _past_key_values.append([key_states, value_states])
        past_key_values = _past_key_values

        return (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            None,  # position_ids
            lora_int_id,
            past_key_values,
            position_embeds,
            all_conv_states,
            all_recurrent_states,
        )

    def forward(self, *args):
        (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            rotary_emb,
            all_conv_states,
            all_recurrent_states,
        ) = self.prepare_forward_args(*args)

        logits, all_hidden_states, new_conv_states, new_recurrent_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=self.rbln_config.output_hidden_states,
            all_conv_states=all_conv_states,
            all_recurrent_states=all_recurrent_states,
        )

        # Return: logits + updated DeltaNet states
        outputs = [logits]
        for conv_s in new_conv_states:
            outputs.append(conv_s)
        for rec_s in new_recurrent_states:
            outputs.append(rec_s)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
