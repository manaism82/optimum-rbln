"""CPU-only checks for the Qwen3.5 vision position-embedding interpolation and its per-size cache.

`RBLNQwen3_5VisionModel.fast_pos_embed_interpolate` runs on the host, so the methods are bound to a stand-in that
carries only the attributes they read; no vision runtime is created.
"""

import types
from collections import OrderedDict

import pytest
import torch

from optimum.rbln.transformers.models.qwen3_5.configuration_qwen3_5 import RBLNQwen3_5VisionModelConfig
from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import RBLNQwen3_5VisionModel


NUM_GRID_PER_SIDE = 8
HIDDEN = 16
MERGE = 2


def _vision(cache_size):
    torch.manual_seed(0)
    model = types.SimpleNamespace(
        rbln_config=types.SimpleNamespace(pos_embed_cache_size=cache_size),
        pos_embed=torch.nn.Embedding(NUM_GRID_PER_SIDE**2, HIDDEN),
        num_grid_per_side=NUM_GRID_PER_SIDE,
        spatial_merge_size=MERGE,
        _pos_embed_cache=OrderedDict(),
    )
    for name in ("fast_pos_embed_interpolate", "_interpolate_frame_pos_embed"):
        setattr(model, name, types.MethodType(getattr(RBLNQwen3_5VisionModel, name), model))
    return model


def _reference(model, grid_thw):
    """The interpolation as it was before the cache: all images at once through Python lists."""
    grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]
    for t, h, w in zip(grid_ts, grid_hs, grid_ws, strict=True):  # noqa: B007
        h_idxs = torch.linspace(0, model.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, model.num_grid_per_side - 1, w)
        h_idxs_floor, w_idxs_floor = h_idxs.int(), w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=model.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=model.num_grid_per_side - 1)
        dh, dw = h_idxs - h_idxs_floor, w_idxs - w_idxs_floor
        base_h, base_h_ceil = h_idxs_floor * model.num_grid_per_side, h_idxs_ceil * model.num_grid_per_side
        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
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
    idx_tensor = torch.tensor(idx_list, dtype=torch.long)
    weight_tensor = torch.tensor(weight_list, dtype=model.pos_embed.weight.dtype)
    pos_embeds = model.pos_embed(idx_tensor) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws, strict=True)])
    out = []
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=True):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = pos_embed.view(t, h // MERGE, MERGE, w // MERGE, MERGE, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
        out.append(pos_embed)
    return torch.cat(out)


GRIDS = [
    [[1, 6, 10]],
    [[1, 6, 10], [1, 4, 4], [1, 6, 10], [1, 12, 8]],  # repeated size within one request
    [[3, 4, 6], [1, 4, 6]],  # video frames share the per-frame embeddings of an image of the same size
    [[1, 14, 22], [2, 2, 2]],
]


@pytest.mark.parametrize("cache_size", [0, 1, 16])
def test_matches_reference_bit_for_bit(cache_size):
    model = _vision(cache_size)
    with torch.no_grad():
        for _ in range(2):  # the second pass is served from the cache when it is enabled
            for grid in GRIDS:
                grid_thw = torch.tensor(grid)
                assert torch.equal(model.fast_pos_embed_interpolate(grid_thw), _reference(model, grid_thw))


def test_cache_keeps_the_most_recently_used_sizes():
    model = _vision(2)
    with torch.no_grad():
        for grid in ([[1, 4, 4]], [[1, 6, 6]], [[1, 4, 4]], [[1, 8, 8]]):
            model.fast_pos_embed_interpolate(torch.tensor(grid))
    assert list(model._pos_embed_cache) == [(4, 4), (8, 8)]


def test_zero_disables_the_cache():
    model = _vision(0)
    with torch.no_grad():
        model.fast_pos_embed_interpolate(torch.tensor(GRIDS[1]))
    assert len(model._pos_embed_cache) == 0


def test_cached_tensor_is_not_returned_to_the_caller():
    model = _vision(16)
    grid_thw = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        model.fast_pos_embed_interpolate(grid_thw).add_(1.0)
        assert torch.equal(model.fast_pos_embed_interpolate(grid_thw), _reference(model, grid_thw))


def test_config_cache_size_is_a_load_time_option(tmp_path):
    assert RBLNQwen3_5VisionModelConfig(max_seq_len=[256]).pos_embed_cache_size == 16
    config = RBLNQwen3_5VisionModelConfig(max_seq_len=[256], pos_embed_cache_size=4)
    assert config.pos_embed_cache_size == 4
    config.save(tmp_path)
    assert "pos_embed_cache_size" not in (tmp_path / "rbln_config.json").read_text()

    # A parent model reloads its `visual` config from the submodule's own rbln_config.json, passing the config it
    # built from the user's overrides; the cache size must survive that reload.
    assert RBLNQwen3_5VisionModelConfig.from_pretrained(tmp_path).pos_embed_cache_size == 16
    assert RBLNQwen3_5VisionModelConfig.from_pretrained(tmp_path, rbln_config=config).pos_embed_cache_size == 4
    reloaded = RBLNQwen3_5VisionModelConfig.from_pretrained(tmp_path, rbln_config={"pos_embed_cache_size": 2})
    assert reloaded.pos_embed_cache_size == 2


@pytest.mark.parametrize("value", [-1, True, 2.0, "4"])
def test_config_rejects_invalid_cache_size(value):
    with pytest.raises(ValueError, match="pos_embed_cache_size"):
        RBLNQwen3_5VisionModelConfig(max_seq_len=[256], pos_embed_cache_size=value)
