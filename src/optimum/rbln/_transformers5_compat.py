# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility shim for transformers 5.x + optimum-rbln.

optimum-rbln was originally built for transformers 4.57.x. Several names were
removed or renamed in transformers 5.x. This module patches them back so
optimum-rbln can be imported without errors on transformers 5.5.1+.

Imported unconditionally from optimum/rbln/__init__.py. On transformers 4.x it
is a no-op.

IMPORTANT: transformers.processing_utils calls `direct_transformers_import`
which re-executes transformers/__init__.py and wipes out module attributes.
So we must trigger that first, THEN apply our patches.
"""

import contextlib

import transformers


def _transformers_major() -> int:
    try:
        return int(transformers.__version__.split(".")[0])
    except (ValueError, AttributeError):
        return 0


if _transformers_major() >= 5:
    # Force early loading of processing_utils so it runs direct_transformers_import
    # before we apply our patches. Otherwise subsequent imports of processing_utils
    # (e.g. by auto_processing_auto during optimum-rbln import) will wipe our patches.
    import transformers.processing_utils  # noqa: F401

    import transformers.modeling_utils as _tm

    # 1) AutoModelForVision2Seq → AutoModelForImageTextToText
    if "AutoModelForVision2Seq" not in transformers.__dict__:
        from transformers import AutoModelForImageTextToText
        transformers.AutoModelForVision2Seq = AutoModelForImageTextToText
        if hasattr(transformers, "_objects"):
            transformers._objects["AutoModelForVision2Seq"] = AutoModelForImageTextToText

    # 1b) MODEL_FOR_VISION_2_SEQ_MAPPING → MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
    import transformers.models.auto.modeling_auto as _auto_mod
    if not hasattr(_auto_mod, "MODEL_FOR_VISION_2_SEQ_MAPPING"):
        _auto_mod.MODEL_FOR_VISION_2_SEQ_MAPPING = _auto_mod.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
        _auto_mod.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = _auto_mod.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    # 1c) Patch from_pretrained to convert use_auth_token → token
    # optimum-rbln passes use_auth_token which transformers 5.x no longer accepts
    _orig_from_pretrained = _tm.PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _patched_from_pretrained(cls, *args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_from_pretrained(cls, *args, **kwargs)

    _tm.PreTrainedModel.from_pretrained = _patched_from_pretrained

    # 1d) PretrainedConfig.torchscript was removed in transformers 5.x
    from transformers.configuration_utils import PretrainedConfig as _PretrainedConfig
    if not hasattr(_PretrainedConfig, "torchscript"):
        _PretrainedConfig.torchscript = False

    # 2) no_init_weights context manager
    if not hasattr(_tm, "no_init_weights"):
        @contextlib.contextmanager
        def _no_init_weights(_enable=True):
            import torch
            old_linear = torch.nn.Linear.reset_parameters
            old_emb = torch.nn.Embedding.reset_parameters
            torch.nn.Linear.reset_parameters = lambda self: None
            torch.nn.Embedding.reset_parameters = lambda self: None
            try:
                yield
            finally:
                torch.nn.Linear.reset_parameters = old_linear
                torch.nn.Embedding.reset_parameters = old_emb
        _tm.no_init_weights = _no_init_weights

    # 2b) Replace @torch.inference_mode() with @torch.no_grad() on get_compiled_model
    # transformers 5.x loads weights as inference tensors, and rebel's get_torch_hash
    # cannot handle them. torch.no_grad() gives the same perf benefit without creating
    # inference tensors.
    import torch as _torch
    from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import (
        RBLNDecoderOnlyModelForCausalLM as _RBLNDecoderOnlyModelForCausalLM,
    )
    _orig_get_compiled = _RBLNDecoderOnlyModelForCausalLM.get_compiled_model
    if hasattr(_orig_get_compiled, "__func__"):
        _unwrapped = _orig_get_compiled.__func__
    else:
        _unwrapped = _orig_get_compiled

    if hasattr(_unwrapped, "__wrapped__"):
        _core_fn = _unwrapped.__wrapped__
    else:
        _core_fn = _unwrapped

    @classmethod
    @_torch.no_grad()
    def _new_get_compiled_model(cls, model, rbln_config):
        return _core_fn(cls, model, rbln_config)

    _RBLNDecoderOnlyModelForCausalLM.get_compiled_model = _new_get_compiled_model

    # 2c) Patch rebel.compile_from_torch to handle inference tensors (fallback)
    # transformers 5.x loads weights in inference mode, and rebel's get_torch_hash
    # fails on inference tensors. Fix: clone all inference params before compiling.
    import rebel as _rebel
    _orig_compile_from_torch = _rebel.compile_from_torch

    def _patched_compile_from_torch(model, *args, **kwargs):
        import torch
        with torch.inference_mode(False):
            for param in model.parameters():
                if param.data.is_inference():
                    param.data = param.data.clone()
            for buf_name, buf in model.named_buffers():
                if buf.is_inference():
                    parts = buf_name.rsplit(".", 1)
                    if len(parts) == 2:
                        parent = model
                        for p in parts[0].split("."):
                            parent = getattr(parent, p)
                        parent.register_buffer(parts[1], buf.clone())
                    else:
                        setattr(model, buf_name, buf.clone())
            for mod in model.modules():
                for attr_name in list(mod.__dict__.keys()):
                    val = mod.__dict__[attr_name]
                    if isinstance(val, _torch.Tensor) and val.is_inference():
                        mod.__dict__[attr_name] = val.clone()
        return _orig_compile_from_torch(model, *args, **kwargs)

    _rebel.compile_from_torch = _patched_compile_from_torch

    # 3) get_state_dict_dtype
    if not hasattr(_tm, "get_state_dict_dtype"):
        import torch
        def _get_state_dict_dtype(state_dict):
            for v in state_dict.values():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    return v.dtype
            return torch.float32
        _tm.get_state_dict_dtype = _get_state_dict_dtype
