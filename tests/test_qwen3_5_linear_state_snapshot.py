"""CPU-only checks for Qwen3.5 GatedDeltaNet state snapshots (hybrid prefix caching, `linear_state_snapshot_slots`).

The prefill/decode graphs are not compiled here. Instead `_CpuGraph` stands in for a compiled `rebel.Runtime`: it
keeps the static `conv_state_*` / `recurrent_state_*` caches as host tensors and runs the exact wrapped torch module
optimum-rbln would hand to the compiler (Qwen3_5_LanguageModelWrapper -> Qwen3_5Model -> Qwen3_5GatedDeltaNet), with
`rbln_cache_update` executed as the in-place row update it is on the device. The runtime side
(`RBLNQwen3_5RuntimeModel`, built by `_qwen3_5_setup_hybrid_runtime`) is the real one, so these tests cover the
per-window schedule, the graph input order, the read/write row split and the decode slice together.

The numeric harness model has linear-attention layers only: the full-attention custom ops have no numeric CPU
implementation, and the cached-prefix KV is handled outside optimum-rbln. The input_info and torch.export checks use
a hybrid (linear + full attention) model.
"""

import contextlib
import json
from types import SimpleNamespace

import pytest
import rebel  # noqa: F401  (registers rbln_custom_ops / rbln ops)
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

from optimum.rbln import (
    RBLNQwen3_5ForCausalLM,
    RBLNQwen3_5ForCausalLMConfig,
    RBLNQwen3_5ForConditionalGeneration,
    RBLNQwen3_5ForConditionalGenerationConfig,
    RBLNQwen3_5MoeForConditionalGeneration,
    RBLNQwen3_5MoeForConditionalGenerationConfig,
)
from optimum.rbln.transformers.models.qwen3_5 import qwen3_5_runtime_utils
from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import _qwen3_5_setup_hybrid_runtime
from optimum.rbln.transformers.models.qwen3_5.qwen3_5_runtime_utils import plan_linear_state_windows


CHUNK = 128
BATCH_SIZE = 2
SNAPSHOT_SLOTS = 2
POISON = 7.0  # every state row starts as 7.0, so a read of an unintended row shows up in the outputs

TEXT_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "layer_types": ["linear_attention", "linear_attention"],
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "vocab_size": 256,
    "max_position_embeddings": 4096,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [3, 3, 2],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
}
VISION_CONFIG = {
    "depth": 1,
    "hidden_size": 64,
    "num_heads": 2,
    "intermediate_size": 128,
    "out_hidden_size": 128,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "num_position_embeddings": 64,
    "in_channels": 3,
}
# Pinned NPU: config resolution never queries a device.
COMMON_RBLN = {
    "batch_size": BATCH_SIZE,
    "max_seq_len": 2048,
    "kvcache_partition_len": 1024,
    "kvcache_num_blocks": 4,
    "prefill_chunk_size": CHUNK,
    "npu": "RBLN-CA25",
    "create_runtimes": False,
}


_UNSET = object()


def _rbln_config(cls, snapshot_slots=_UNSET, **kwargs):
    kw = {**COMMON_RBLN, **kwargs}
    if snapshot_slots is not _UNSET:
        kw["linear_state_snapshot_slots"] = snapshot_slots
    if cls is not RBLNQwen3_5ForCausalLMConfig:
        kw.setdefault("visual", {"max_seq_len": 256, "npu": "RBLN-CA25"})
    return cls(**kw)


def _is_state(name):
    return name.startswith(("conv_state_", "recurrent_state_")) and not name.endswith("_mask")


@contextlib.contextmanager
def _device_cache_update():
    """Run `rbln_cache_update` as the in-place update it is on the device (its CPU stub returns empty_like)."""

    def cache_update(cache, state, position, axis):
        axis = int(axis)
        cache.narrow(axis, int(position), state.shape[axis]).copy_(state)
        return cache

    namespace = torch.ops.rbln_custom_ops
    original = namespace.rbln_cache_update
    namespace.rbln_cache_update = cache_update
    try:
        yield
    finally:
        namespace.rbln_cache_update = original


class _CpuGraph:
    """Stand-in for a compiled graph: same positional input order, static state caches kept on the host."""

    # Inputs the decode graph does not consume (the compiler drops them; the decode runtime never feeds them).
    _DECODE_UNUSED = ("conv_state_mask", "recurrent_state_mask", "valid_mask")

    def __init__(self, harness, phase, input_info):
        self.harness = harness
        self.phase = phase
        self.input_info = input_info
        unused = self._DECODE_UNUSED if phase == "decode" else ()
        names = [n for n, _, _ in input_info if not _is_state(n) and n not in unused]
        self._index_to_input_name = dict(enumerate(names))
        self.calls = []

    def __call__(self, *args):
        named = dict(zip(self._index_to_input_name.values(), args, strict=True))
        self.calls.append({k: v.clone() for k, v in named.items()})
        full = []
        for name, shape, dtype in self.input_info:
            if _is_state(name):
                full.append(self.harness.states[name])
            elif name in named:
                full.append(named[name])
            else:
                full.append(torch.zeros(shape, dtype=getattr(torch, dtype)))
        self.harness.wrapped.phase = self.phase
        with torch.no_grad(), _device_cache_update():
            return self.harness.wrapped(*full)


class _Qwen3_5CpuHarness:
    def __init__(self, dtype, snapshot_slots, gdn_chunk_size, output_hidden_states):
        torch.manual_seed(0)
        self.dtype = dtype
        hf_config = Qwen3_5Config(text_config=TEXT_CONFIG, vision_config=VISION_CONFIG)
        hf_model = Qwen3_5ForConditionalGeneration(hf_config).eval()
        with torch.no_grad():
            for p in hf_model.model.language_model.parameters():
                p.copy_(torch.randn_like(p) * 0.2)
            for layer in hf_model.model.language_model.layers:
                gdn = layer.linear_attn
                # Per-token decay exp(-A * softplus(a + dt_bias)): heads from long (~1e-3) to short (~0.5) memory,
                # so the carried state still shapes the last tokens of a several-hundred-token suffix.
                gdn.A_log.copy_(torch.log(torch.logspace(-2, 1, gdn.A_log.numel())))
                gdn.dt_bias.fill_(-2.0)
                gdn.in_proj_a.weight.mul_(0.1)
        hf_model = hf_model.to(dtype)

        cls = RBLNQwen3_5ForConditionalGeneration
        rbln_config = _rbln_config(
            RBLNQwen3_5ForConditionalGenerationConfig,
            snapshot_slots,
            gdn_chunk_size=gdn_chunk_size,
            dtype=str(dtype).removeprefix("torch."),
            output_hidden_states=output_hidden_states,
        )
        hf_model = cls._reconstruct_model_if_needed(hf_model)
        self.rbln_config = cls.update_rbln_config(
            preprocessors=None, model=hf_model, model_config=hf_config, rbln_config=rbln_config
        )
        self.wrapped = cls._wrap_model_if_needed(hf_model, self.rbln_config)
        prefill_cfg, decode_cfg = self.rbln_config.compile_cfgs
        self.prefill_info, self.decode_info = prefill_cfg.input_info, decode_cfg.input_info
        self.state_names = [n for n, _, _ in self.prefill_info if _is_state(n)]
        self.state_shapes = {n: (s, d) for n, s, d in self.prefill_info if _is_state(n)}
        self.reset_states()

        self.prefill_graph = _CpuGraph(self, "prefill", self.prefill_info)
        self.decode_graph = _CpuGraph(self, "decode", self.decode_info)
        model = SimpleNamespace(
            rbln_config=self.rbln_config,
            config=hf_config,
            embed_tokens=None,
            dtype=dtype,
            model=[self.prefill_graph, self.decode_graph],
            logits_last_dim=hf_config.text_config.vocab_size,
            can_generate=lambda: True,
        )
        _qwen3_5_setup_hybrid_runtime(model)
        self.prefill_decoder, self.decoder = model.prefill_decoder, model.decoder
        shapes = {n: s for n, s, _ in self.prefill_info}
        self.hidden_size = shapes["inputs_embeds"][-1]
        self.rotary_ndims = shapes["position_emb"][-1]
        self.num_blocks = shapes["block_tables"][0]

    def reset_states(self):
        self.states = {n: torch.full(s, POISON, dtype=getattr(torch, d)) for n, (s, d) in self.state_shapes.items()}

    def snapshot(self, row):
        return {n: t[row].clone() for n, t in self.states.items()}

    def tokens(self, length, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, length, self.hidden_size, generator=g).to(self.dtype)

    def prefill(self, inputs_embeds, batch_idx, start=0, **state_kwargs):
        """Returns the runtime output (last-token logits, and per-layer hidden states of every fed token)."""
        length = inputs_embeds.shape[1]
        return self.prefill_decoder(
            inputs_embeds=inputs_embeds,
            cache_position=torch.arange(start, start + length, dtype=torch.int32).unsqueeze(0),
            batch_idx=batch_idx,
            position_embed=torch.zeros(2, 1, 1, length, self.rotary_ndims, dtype=self.dtype),
            block_tables=torch.zeros(self.num_blocks, dtype=torch.int16),
            **state_kwargs,
        )

    def decode(self, inputs_embeds, positions):
        batch = inputs_embeds.shape[0]
        return self.decoder(
            inputs_embeds=inputs_embeds,
            cache_position=torch.tensor(positions, dtype=torch.int32).view(batch, 1),
            position_embed=torch.zeros(2, batch, 1, 1, self.rotary_ndims, dtype=self.dtype),
            block_tables=torch.zeros(batch, self.num_blocks, dtype=torch.int16),
        ).logits


def _rows_equal(a, b):
    return all(torch.equal(a[n], b[n]) for n in a)


def _max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


_HARNESSES = {}


def _harness(dtype=torch.float32, snapshot_slots=SNAPSHOT_SLOTS, gdn_chunk_size=CHUNK, output_hidden_states=False):
    key = (dtype, snapshot_slots, gdn_chunk_size, output_hidden_states)
    if key not in _HARNESSES:
        _HARNESSES[key] = _Qwen3_5CpuHarness(dtype, snapshot_slots, gdn_chunk_size, output_hidden_states)
    h = _HARNESSES[key]
    h.reset_states()
    h.prefill_graph.calls.clear()
    h.decode_graph.calls.clear()
    return h


# ---------------------------------------------------------------------------------------------------------------
# (1) Bit-exact capture / restore against an uninterrupted prefill
# ---------------------------------------------------------------------------------------------------------------

PREFIX = 2 * CHUNK  # cached boundary b (a multiple of prefill_chunk_size)
LIVE_X, LIVE_Y = 0, 1  # batch rows of the capturing and the restoring request
SNAP_A, SNAP_B = BATCH_SIZE, BATCH_SIZE + 1  # snapshot rows batch_size + s


def _assert_same_outputs(out, ref, start=0):
    """Bitwise: the next-token logits and every layer's hidden state of every fed token (`ref` from `start`)."""
    assert torch.equal(out.logits, ref.logits)
    assert len(out.hidden_states) == len(ref.hidden_states)
    for layer, (o, r) in enumerate(zip(out.hidden_states, ref.hidden_states, strict=True)):
        assert torch.equal(o, r[:, start:]), f"hidden state {layer} differs"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("gdn_chunk_size", [CHUNK, CHUNK // 2], ids=["gdn128", "gdn64"])
@pytest.mark.parametrize("suffix_len", [1, 127, 128, 129, 300])
def test_capture_and_restore_are_bit_exact(dtype, gdn_chunk_size, suffix_len):
    """Prompt lengths b + 1 .. b + 300 around window boundaries (b = 256 is the cached prefix)."""
    h = _harness(dtype, gdn_chunk_size=gdn_chunk_size, output_hidden_states=True)
    prefix = h.tokens(PREFIX, seed=1)
    x = torch.cat([prefix, h.tokens(200, seed=2)], dim=1)  # request that captures the prefix
    y = torch.cat([prefix, h.tokens(suffix_len, seed=3)], dim=1)  # request that hits it
    y_len = y.shape[1]
    steps = torch.cat([h.tokens(1, seed=4), h.tokens(1, seed=5)])  # one decode token per live row

    # References: uninterrupted prefills, then one decode step.
    x_ref = h.prefill(x, LIVE_X)
    state_x_ref = h.snapshot(LIVE_X)
    h.reset_states()
    h.prefill(prefix, LIVE_X)
    state_prefix_ref = h.snapshot(LIVE_X)
    h.reset_states()
    y_ref = h.prefill(y, LIVE_Y)
    state_y_ref = h.snapshot(LIVE_Y)
    decode_y_ref = h.decode(steps, [x.shape[1], y_len])[LIVE_Y]
    state_y_decoded_ref = h.snapshot(LIVE_Y)
    if y_len > PREFIX + CHUNK:
        h.reset_states()
        h.prefill(y[:, : PREFIX + CHUNK], LIVE_Y)
        state_y_boundary_ref = h.snapshot(LIVE_Y)

    # Capture: the window ending at the boundary writes the snapshot row; the request's own outputs are unchanged.
    h.reset_states()
    _assert_same_outputs(h.prefill(x, LIVE_X, state_capture={PREFIX: SNAP_A}), x_ref)
    assert _rows_equal(h.snapshot(LIVE_X), state_x_ref)
    assert _rows_equal(h.snapshot(SNAP_A), state_prefix_ref)

    # Restore: the second request feeds only its suffix and resumes from the snapshot row.
    _assert_same_outputs(h.prefill(y[:, PREFIX:], LIVE_Y, start=PREFIX, state_restore_row=SNAP_A), y_ref, PREFIX)
    assert _rows_equal(h.snapshot(LIVE_Y), state_y_ref)
    assert _rows_equal(h.snapshot(SNAP_A), state_prefix_ref)  # a restore only reads the snapshot

    # Decode continues from the restored row exactly as after a full prefill, and leaves snapshot rows alone.
    assert torch.equal(h.decode(steps, [x.shape[1], y_len])[LIVE_Y], decode_y_ref)
    assert _rows_equal(h.snapshot(LIVE_Y), state_y_decoded_ref)
    assert _rows_equal(h.snapshot(SNAP_A), state_prefix_ref)

    # Restore + capture: resume from one snapshot and capture a longer boundary of the same request.
    if y_len > PREFIX + CHUNK:
        h.reset_states()
        for name in h.state_names:
            h.states[name][SNAP_A] = state_prefix_ref[name]
        out = h.prefill(
            y[:, PREFIX:], LIVE_Y, start=PREFIX, state_restore_row=SNAP_A, state_capture={PREFIX + CHUNK: SNAP_B}
        )
        _assert_same_outputs(out, y_ref, PREFIX)
        assert _rows_equal(h.snapshot(LIVE_Y), state_y_ref)
        assert _rows_equal(h.snapshot(SNAP_B), state_y_boundary_ref)
        assert _rows_equal(h.snapshot(SNAP_A), state_prefix_ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_snapshot_rows_do_not_change_uncached_outputs(dtype):
    """K > 0 without cache arguments computes exactly what the K = 0 model computes."""
    results = []
    for snapshot_slots in (SNAPSHOT_SLOTS, 0):
        h = _harness(dtype, snapshot_slots=snapshot_slots, output_hidden_states=True)
        x = h.tokens(PREFIX + 77, seed=1)
        steps = torch.cat([h.tokens(1, seed=2), h.tokens(1, seed=3)])
        h.prefill(x, LIVE_Y)
        out = h.prefill(x, LIVE_X)
        decode_logits = h.decode(steps, [x.shape[1]] * BATCH_SIZE)
        results.append((out, decode_logits, h.snapshot(LIVE_X), h.snapshot(LIVE_Y)))
    (with_rows, without_rows) = results
    _assert_same_outputs(with_rows[0], without_rows[0])
    assert torch.equal(with_rows[1], without_rows[1])
    assert _rows_equal(with_rows[2], without_rows[2])
    assert _rows_equal(with_rows[3], without_rows[3])


def _patched_plan(monkeypatch, edit_first_window):
    original = qwen3_5_runtime_utils.plan_linear_state_windows

    def plan(*args, **kwargs):
        schedule = original(*args, **kwargs)
        return [edit_first_window(schedule[0])] + schedule[1:]

    monkeypatch.setattr(qwen3_5_runtime_utils, "plan_linear_state_windows", plan)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_negative_controls_diverge(dtype, monkeypatch):
    """The restore needs both the snapshot row AND carry masks of ones; the alternatives available without
    snapshot rows are wrong."""
    h = _harness(dtype, output_hidden_states=True)
    prefix = h.tokens(PREFIX, seed=1)
    x = torch.cat([prefix, h.tokens(200, seed=2)], dim=1)
    y = torch.cat([prefix, h.tokens(150, seed=3)], dim=1)
    y_ref = h.prefill(y, LIVE_Y)
    state_y_ref = h.snapshot(LIVE_Y)

    def run_restore():
        h.reset_states()
        h.prefill(x, LIVE_X, state_capture={PREFIX: SNAP_A})
        h.decode(torch.cat([h.tokens(1, seed=4), h.tokens(1, seed=5)]), [x.shape[1], 0])
        return h.prefill(y[:, PREFIX:], LIVE_Y, start=PREFIX, state_restore_row=SNAP_A), h.snapshot(LIVE_Y)

    # Sanity: the correct schedule reproduces the reference.
    out, state = run_restore()
    _assert_same_outputs(out, y_ref, PREFIX)
    assert _rows_equal(state, state_y_ref)

    controls = {}
    with monkeypatch.context() as m:
        # Today's first-window behaviour: zero masks (the snapshot is read but discarded).
        _patched_plan(m, lambda w: (w[0], w[1], False))
        controls["zero_mask"] = run_restore()
    with monkeypatch.context() as m:
        # No snapshot row: resume from a row that served the same prefix (it holds END-of-request state).
        _patched_plan(m, lambda w: (LIVE_X, w[1], True))
        controls["stale_row"] = run_restore()

    for name, (out, state) in controls.items():
        suffix_hidden = _max_abs_diff(out.hidden_states[-1], y_ref.hidden_states[-1][:, PREFIX:])
        last_logits = _max_abs_diff(out.logits, y_ref.logits)
        final_state = max(_max_abs_diff(state[n], state_y_ref[n]) for n in state)
        assert min(suffix_hidden, last_logits, final_state) > 1e-2, (name, suffix_hidden, last_logits, final_state)


# ---------------------------------------------------------------------------------------------------------------
# (2) Per-window graph inputs fed by the runtime
# ---------------------------------------------------------------------------------------------------------------


def _expected_window(cache_start, batch_idx, read_row, carry, valid):
    return {
        "cache_start": cache_start,
        "batch_idx": batch_idx,
        "state_src_idx": read_row,
        "mask": "ones" if carry else "zeros",
        "valid": valid,
    }


def _observed_windows(h, snapshot_slots):
    expected_names = {n for n, _, _ in h.prefill_info if not _is_state(n)}
    windows = []
    for named in h.prefill_graph.calls:
        assert set(named) == expected_names
        for name in ("batch_idx", "state_src_idx", "query_position"):
            if name in named:
                assert named[name].dtype == torch.int16 and named[name].dim() == 0
        assert ("state_src_idx" in named) == (snapshot_slots > 0)
        conv_mask, recurrent_mask = named["conv_state_mask"], named["recurrent_state_mask"]
        assert conv_mask.unique().numel() == 1 and recurrent_mask.unique().numel() == 1
        assert conv_mask.flatten()[0].item() == recurrent_mask.flatten()[0].item()
        windows.append(
            {
                "cache_start": named["cache_position"][0, 0].item(),
                "batch_idx": named["batch_idx"].item(),
                "state_src_idx": named["state_src_idx"].item() if "state_src_idx" in named else None,
                "mask": "ones" if conv_mask.flatten()[0].item() == 1 else "zeros",
                "valid": int(named["valid_mask"].sum().item()),
            }
        )
    return windows


B = 1  # live row used by the schedule tests
S0, S1 = BATCH_SIZE, BATCH_SIZE + 1

SCHEDULES = {
    # name: (snapshot_slots, start, num_tokens, state kwargs, expected per-window (start, write, read, carry, valid))
    "no_cache_K0": (0, 0, 300, {}, [(0, B, None, False, 128), (128, B, None, True, 128), (256, B, None, True, 44)]),
    "no_cache_K2": (2, 0, 300, {}, [(0, B, B, False, 128), (128, B, B, True, 128), (256, B, B, True, 44)]),
    "capture_at_256": (
        2,
        0,
        300,
        {"state_capture": {256: S0}},
        [(0, B, B, False, 128), (128, S0, B, True, 128), (256, B, S0, True, 44)],
    ),
    "capture_two_boundaries": (
        2,
        0,
        500,
        {"state_capture": {384: S1, 256: S0}},
        [(0, B, B, False, 128), (128, S0, B, True, 128), (256, S1, S0, True, 128), (384, B, S1, True, 116)],
    ),
    "restore_at_256": (
        2,
        256,
        200,
        {"state_restore_row": S1},
        [(256, B, S1, True, 128), (384, B, B, True, 72)],
    ),
    "restore_at_256_capture_at_384": (
        2,
        256,
        300,
        {"state_restore_row": S0, "state_capture": {384: S1}},
        [(256, S1, S0, True, 128), (384, B, S1, True, 128), (512, B, B, True, 44)],
    ),
}


@pytest.mark.parametrize("name", list(SCHEDULES))
def test_prefill_window_inputs(name):
    snapshot_slots, start, num_tokens, state_kwargs, expected = SCHEDULES[name]
    h = _harness(snapshot_slots=snapshot_slots)
    h.prefill(h.tokens(num_tokens, seed=0), B, start=start, **state_kwargs)
    observed = _observed_windows(h, snapshot_slots)
    assert observed == [_expected_window(s, w, r, c, v) for s, w, r, c, v in expected]
    assert plan_linear_state_windows(start, num_tokens, CHUNK, B, BATCH_SIZE, snapshot_slots, **state_kwargs) == [
        (r if snapshot_slots else B, w, c) for _, w, r, c, _ in expected
    ]


INVALID_PLANS = [
    # (snapshot_slots, prefix_cached_len, query_length, batch_idx, state kwargs, error match)
    (0, 256, 100, B, {}, "state_restore_row"),
    (2, 256, 100, B, {}, "state_restore_row"),
    (0, 0, 300, B, {"state_capture": {128: S0}}, "linear_state_snapshot_slots=0"),
    (0, 256, 100, B, {"state_restore_row": S0}, "linear_state_snapshot_slots=0"),
    (2, 200, 100, B, {"state_restore_row": S0}, "multiple of prefill_chunk_size"),
    (2, 0, 300, B, {"state_restore_row": S0}, "cache position 0"),
    (2, 256, 100, B, {"state_restore_row": B}, "snapshot row"),
    (2, 256, 100, B, {"state_restore_row": BATCH_SIZE + SNAPSHOT_SLOTS}, "snapshot row"),
    (2, 0, 300, B, {"state_capture": {256: B}}, "snapshot rows"),
    (2, 0, 300, B, {"state_capture": {200: S0}}, "boundary 200"),
    (2, 0, 256, B, {"state_capture": {256: S0}}, "boundary 256"),  # would be the last window
    (2, 0, 300, B, {"state_capture": {384: S0}}, "boundary 384"),  # beyond the prompt
    (2, 256, 300, B, {"state_restore_row": S0, "state_capture": {256: S1}}, "boundary 256"),  # not after P
    (2, 0, 500, B, {"state_capture": {128: S0, 256: S0}}, "more than once"),
    (2, 0, 300, None, {}, "batch_idx"),
    (2, 0, 300, BATCH_SIZE, {}, "batch_idx"),  # a live row id that is a snapshot row
    (2, 0, 300, B, {"state_capture": {256.0: S0}}, "must be an integer"),
]


@pytest.mark.parametrize("snapshot_slots, prefix, query_length, batch_idx, kwargs, match", INVALID_PLANS)
def test_invalid_state_arguments_raise(snapshot_slots, prefix, query_length, batch_idx, kwargs, match):
    with pytest.raises(ValueError, match=match):
        plan_linear_state_windows(prefix, query_length, CHUNK, batch_idx, BATCH_SIZE, snapshot_slots, **kwargs)


def test_runtime_rejects_cached_prefix_without_restore_row():
    h = _harness()
    with pytest.raises(ValueError, match="state_restore_row"):
        h.prefill(h.tokens(100, seed=0), B, start=256)
    assert h.prefill_graph.calls == []


def test_runtime_rejects_state_arguments_on_decode():
    h = _harness()
    with pytest.raises(ValueError, match="only apply to prefill"):
        h.decoder(
            inputs_embeds=h.tokens(BATCH_SIZE, seed=0).view(BATCH_SIZE, 1, -1),
            cache_position=torch.zeros(BATCH_SIZE, 1, dtype=torch.int32),
            block_tables=torch.zeros(BATCH_SIZE, h.num_blocks, dtype=torch.int16),
            state_restore_row=S0,
        )


# ---------------------------------------------------------------------------------------------------------------
# (3) Config round trip; K = 0 input_info / compile configs identical to the pre-snapshot layout
# ---------------------------------------------------------------------------------------------------------------

# Prefill / decode input_info of the mixed tiny model (3 linear + 1 full attention layer, batch_size 2) before
# linear_state_snapshot_slots existed (optimum-rbln 375f90d3). K = 0 must reproduce it exactly.
MIXED_TEXT_CONFIG = {
    **TEXT_CONFIG,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
}
_LINEAR_STATES = [
    entry
    for i in range(3)
    for entry in ([f"conv_state_{i}", [2, 3, 256], "float32"], [f"recurrent_state_{i}", [2, 128, 32], "float32"])
]
_KV = [["past_key_values_6", [4, 1, 1024, 64], "float32"], ["past_key_values_7", [4, 1, 1024, 64], "float32"]]
PRE_SNAPSHOT_INPUT_INFO = {
    "prefill": [
        ["inputs_embeds", [1, 128, 128], "float32"],
        ["cache_position", [1, 128], "int32"],
        ["block_tables", [2], "int16"],
        ["position_emb", [2, 1, 1, 128, 16], "float32"],
        ["query_position", [], "int16"],
        *_LINEAR_STATES,
        *_KV,
        ["conv_state_mask", [1, 3, 256], "float32"],
        ["recurrent_state_mask", [1, 128, 32], "float32"],
        ["valid_mask", [1, 128, 1], "float32"],
        ["batch_idx", [], "int16"],
    ],
    "decoder_batch_2": [
        ["inputs_embeds", [2, 1, 128], "float32"],
        ["cache_position", [2, 1], "int32"],
        ["block_tables", [2, 2], "int16"],
        ["position_emb", [2, 2, 1, 1, 16], "float32"],
        *_LINEAR_STATES,
        *_KV,
        ["conv_state_mask", [2, 3, 256], "float32"],
        ["recurrent_state_mask", [2, 128, 32], "float32"],
        ["valid_mask", [2, 1, 1], "float32"],
    ],
}


def _updated_mixed_config(snapshot_slots=_UNSET):
    torch.manual_seed(0)
    hf_config = Qwen3_5Config(text_config=MIXED_TEXT_CONFIG, vision_config=VISION_CONFIG)
    hf_model = RBLNQwen3_5ForConditionalGeneration._reconstruct_model_if_needed(
        Qwen3_5ForConditionalGeneration(hf_config).eval()
    )
    rbln_config = _rbln_config(
        RBLNQwen3_5ForConditionalGenerationConfig,
        snapshot_slots,
        visual={"cls_name": "RBLNQwen3_5VisionModelConfig", "max_seq_len": 256, "npu": "RBLN-CA25"},
    )
    return hf_config, RBLNQwen3_5ForConditionalGeneration.update_rbln_config(
        preprocessors=None, model=hf_model, model_config=hf_config, rbln_config=rbln_config
    )


def _input_info_by_graph(rbln_config):
    return {c.compiled_model_name: [list(map(_jsonable, e)) for e in c.input_info] for c in rbln_config.compile_cfgs}


def _jsonable(x):
    return json.loads(json.dumps(x))


@pytest.mark.parametrize("snapshot_slots", [_UNSET, 0], ids=["default", "explicit_0"])
def test_k0_input_info_is_unchanged(snapshot_slots):
    _, rbln_config = _updated_mixed_config(snapshot_slots)
    assert rbln_config.linear_state_snapshot_slots == 0
    assert _input_info_by_graph(rbln_config) == PRE_SNAPSHOT_INPUT_INFO


def test_k0_compile_configs_match_default():
    _, default = _updated_mixed_config()
    _, explicit = _updated_mixed_config(0)
    assert [c.asdict() for c in default.compile_cfgs] == [c.asdict() for c in explicit.compile_cfgs]
    assert default._prepare_for_serialization() == explicit._prepare_for_serialization()


def test_snapshot_rows_input_info():
    _, rbln_config = _updated_mixed_config(SNAPSHOT_SLOTS)
    rows = BATCH_SIZE + SNAPSHOT_SLOTS
    expected = {}
    for graph, info in PRE_SNAPSHOT_INPUT_INFO.items():
        expected[graph] = [[n, [rows, *s[1:]] if _is_state(n) else s, d] for n, s, d in info]
    expected["prefill"].append(["state_src_idx", [], "int16"])
    assert _input_info_by_graph(rbln_config) == expected
    linear_metas = [m for m in rbln_config.cache_metas if m.layer_type == "linear_attention"]
    assert len(linear_metas) == 6 and all(m.compile_shape[0] == rows for m in linear_metas)


@pytest.mark.parametrize("snapshot_slots", [0, SNAPSHOT_SLOTS])
def test_config_round_trip(tmp_path, snapshot_slots):
    _, rbln_config = _updated_mixed_config(snapshot_slots)
    rbln_config.save(str(tmp_path))
    saved = json.loads((tmp_path / "rbln_config.json").read_text())
    assert saved["linear_state_snapshot_slots"] == snapshot_slots

    rows = BATCH_SIZE + snapshot_slots
    linear_metas = [m for m in saved["cache_metas"] if m["layer_type"] == "linear_attention"]
    assert len(linear_metas) == 6 and all(m["shape"][0] == rows for m in linear_metas)

    reloaded = RBLNQwen3_5ForConditionalGenerationConfig.from_pretrained(str(tmp_path))
    assert reloaded.linear_state_snapshot_slots == snapshot_slots
    assert _input_info_by_graph(reloaded) == _input_info_by_graph(rbln_config)


def test_config_without_the_field_loads_as_k0(tmp_path):
    _, rbln_config = _updated_mixed_config()
    rbln_config.save(str(tmp_path))
    path = tmp_path / "rbln_config.json"
    saved = json.loads(path.read_text())
    del saved["linear_state_snapshot_slots"]  # an artifact compiled before the field existed
    path.write_text(json.dumps(saved))
    reloaded = RBLNQwen3_5ForConditionalGenerationConfig.from_pretrained(str(tmp_path))
    assert reloaded.linear_state_snapshot_slots == 0
    assert _input_info_by_graph(reloaded) == PRE_SNAPSHOT_INPUT_INFO


# ---------------------------------------------------------------------------------------------------------------
# (4) Decode slice and config validation
# ---------------------------------------------------------------------------------------------------------------


def test_decode_reads_and_writes_only_live_rows():
    h = _harness()
    for row in range(BATCH_SIZE + SNAPSHOT_SLOTS):
        for name in h.state_names:
            h.states[name][row].normal_(generator=torch.Generator().manual_seed(row))
    before = {n: t.clone() for n, t in h.states.items()}
    steps = torch.cat([h.tokens(1, seed=1), h.tokens(1, seed=2)])
    logits = h.decode(steps, [10, 20])
    for name in h.state_names:
        assert torch.equal(h.states[name][BATCH_SIZE:], before[name][BATCH_SIZE:])
        assert not torch.equal(h.states[name][:BATCH_SIZE], before[name][:BATCH_SIZE])

    # The live rows decode exactly as in a K = 0 model whose cache holds the same live rows.
    k0 = _harness(snapshot_slots=0)
    for name in k0.state_names:
        k0.states[name].copy_(before[name][:BATCH_SIZE])
    assert torch.equal(k0.decode(steps, [10, 20]), logits)
    for name in k0.state_names:
        assert torch.equal(k0.states[name], h.states[name][:BATCH_SIZE])


CONFIG_CLASSES = [RBLNQwen3_5ForConditionalGenerationConfig, RBLNQwen3_5MoeForConditionalGenerationConfig]


@pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
@pytest.mark.parametrize("decoder_batch_sizes", [[2, 1], [1]])
def test_snapshot_rows_require_single_full_batch_decoder(config_cls, decoder_batch_sizes):
    with pytest.raises(ValueError, match="decoder_batch_sizes"):
        _rbln_config(config_cls, SNAPSHOT_SLOTS, decoder_batch_sizes=list(decoder_batch_sizes))
    # Unchanged without snapshot rows.
    assert _rbln_config(config_cls, 0, decoder_batch_sizes=list(decoder_batch_sizes)).linear_state_snapshot_slots == 0


@pytest.mark.parametrize("config_cls", CONFIG_CLASSES + [RBLNQwen3_5ForCausalLMConfig])
@pytest.mark.parametrize("bad", [-1, 1.5, True, "2", None])
def test_invalid_snapshot_slots_raise(config_cls, bad):
    with pytest.raises(ValueError, match="linear_state_snapshot_slots"):
        _rbln_config(config_cls, bad)


@pytest.mark.parametrize("config_cls", CONFIG_CLASSES + [RBLNQwen3_5ForCausalLMConfig])
def test_snapshot_slots_accepted(config_cls):
    config = _rbln_config(config_cls, 3, decoder_batch_sizes=[BATCH_SIZE])
    assert config.linear_state_snapshot_slots == 3
    assert _rbln_config(config_cls).linear_state_snapshot_slots == 0


# ---------------------------------------------------------------------------------------------------------------
# (5) The graphs handed to the compiler trace (torch.export, as rebel.compile_from_torch does first)
# ---------------------------------------------------------------------------------------------------------------

MOE_TEXT_CONFIG = {
    **MIXED_TEXT_CONFIG,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
}
EXPORT_MODELS = {
    "qwen3_5_vl": lambda: (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Config(text_config=MIXED_TEXT_CONFIG, vision_config=VISION_CONFIG),
        RBLNQwen3_5ForConditionalGeneration,
        RBLNQwen3_5ForConditionalGenerationConfig,
    ),
    "qwen3_5_moe_vl": lambda: (
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeConfig(text_config=MOE_TEXT_CONFIG, vision_config=VISION_CONFIG),
        RBLNQwen3_5MoeForConditionalGeneration,
        RBLNQwen3_5MoeForConditionalGenerationConfig,
    ),
    "qwen3_5_text": lambda: (
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig(**MIXED_TEXT_CONFIG),
        RBLNQwen3_5ForCausalLM,
        RBLNQwen3_5ForCausalLMConfig,
    ),
}


@pytest.mark.parametrize("snapshot_slots", [0, SNAPSHOT_SLOTS])
@pytest.mark.parametrize("model_name", list(EXPORT_MODELS))
def test_graphs_export(model_name, snapshot_slots):
    torch.manual_seed(0)
    hf_cls, hf_config, rbln_cls, config_cls = EXPORT_MODELS[model_name]()
    hf_model = rbln_cls._reconstruct_model_if_needed(hf_cls(hf_config).eval())
    rbln_config = rbln_cls.update_rbln_config(
        preprocessors=None,
        model=hf_model,
        model_config=hf_config,
        rbln_config=_rbln_config(config_cls, snapshot_slots),
    )
    wrapped = rbln_cls._wrap_model_if_needed(hf_model, rbln_config)
    static_tensors = {}
    for compile_cfg in rbln_config.compile_cfgs:
        phase = "prefill" if compile_cfg.compiled_model_name == "prefill" else "decode"
        example_inputs = compile_cfg.get_dummy_inputs(fill=0, static_tensors=static_tensors)
        if phase == "prefill":
            _, static_tensors = rbln_cls._get_compile_context(compile_cfg, example_inputs)
        wrapped.phase = phase
        linear = torch.nn.functional.linear
        torch.nn.functional.linear = torch.ops.rbln_custom_ops.linear  # as RBLNDecoderOnlyModel._compile_model
        try:
            program = torch.export.export(wrapped, tuple(example_inputs), strict=False)
        finally:
            torch.nn.functional.linear = linear

        placeholders = {node.name: node for node in program.graph.nodes if node.op == "placeholder"}
        user_inputs = program.graph_signature.user_inputs
        assert len(user_inputs) == len(compile_cfg.input_info)
        used = {
            name
            for (name, _, _), node in zip(compile_cfg.input_info, user_inputs, strict=True)
            if placeholders[node].users
        }
        if phase == "prefill":
            assert "batch_idx" in used
            assert ("state_src_idx" in used) == (snapshot_slots > 0)
        assert all(n in used for n, _, _ in compile_cfg.input_info if _is_state(n))
