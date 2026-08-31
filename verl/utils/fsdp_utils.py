# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import itertools
import json
import math
import os
import threading
from abc import ABC
from collections import OrderedDict
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from packaging import version
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name

from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.model import check_exclude_modules, check_target_modules


_FSDP1_MIXED_GRAD_CLIP_LOCK = threading.Lock()


def _get_grad_norm_allow_mixed_dtype(params, norm_type, zero, device):
    """Equivalent to PyTorch FSDP's private helper without its dtype guard.

    PyTorch already evaluates each per-gradient norm in FP32, so gradients of
    different storage dtypes can be combined safely after converting each norm
    to FP32. This preserves the public FSDP global sharded-norm semantics.
    """
    grads = [param.grad for param in params if param.grad is not None]
    if not grads:
        return zero
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32) for grad in grads]
        ),
        norm_type,
        dtype=torch.float32,
    )
    return grad_norm.to(device=device)


def fsdp1_clip_grad_norm_(module: FSDP, max_norm, norm_type=2.0):
    """FSDP1 global grad clipping that supports mixed BF16/FP32 gradients.

    Torch 2.7's public ``FSDP.clip_grad_norm_`` implements the correct global
    reduction and dtype promotion, but its private local helper rejects mixed
    gradient dtypes before doing the documented FP32 norm calculation. Patch
    only that helper for the duration of the collective and retain all public
    FSDP sharded/non-sharded accounting and in-place clipping behavior.
    """
    grad_dtypes = {param.grad.dtype for param in module.parameters() if param.grad is not None}
    if len(grad_dtypes) <= 1:
        return module.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)

    import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_impl

    with _FSDP1_MIXED_GRAD_CLIP_LOCK:
        original_get_grad_norm = fsdp_impl._get_grad_norm
        fsdp_impl._get_grad_norm = _get_grad_norm_allow_mixed_dtype
        try:
            return module.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)
        finally:
            fsdp_impl._get_grad_norm = original_get_grad_norm


@torch.no_grad()
def repair_unreduced_fsdp1_grad_shards_(module: FSDP):
    """Reduce-scatter any FSDP1 flat gradient left unexpectedly unsharded.

    A flat parameter managed by a sharded FSDP handle must present a local
    gradient shard to both ``clip_grad_norm_`` and the optimizer. A missed
    post-backward hook can instead leave the full low-precision gradient on a
    local parameter shard. Detect that invariant violation by shape and run
    the same padded reduce-scatter, predivide, postdivide, and parameter-dtype
    cast used by PyTorch's FSDP1 post-backward path.

    The repair is intentionally narrow: any shape that cannot be a padded full
    gradient for an FSDP-managed sharded flat parameter raises immediately.
    """
    if not isinstance(module, FSDP):
        return {"scanned": 0, "repaired": 0, "examples": []}

    _lazy_init(module, module)
    world_size = int(module.world_size)
    process_group = module.process_group
    predivide = float(module._gradient_predivide_factor)
    postdivide = float(module._gradient_postdivide_factor)
    repaired = []
    scanned = 0

    for handle_idx, handle in enumerate(module._all_handles):
        flat_param = handle.flat_param
        grad = flat_param.grad
        if grad is None:
            continue
        scanned += 1
        if grad.numel() == flat_param.numel():
            continue
        if not handle.uses_sharded_strategy:
            raise RuntimeError(
                "Non-sharded FSDP flat parameter has mismatched gradient shape: "
                f"param={tuple(flat_param.shape)} grad={tuple(grad.shape)}"
            )

        local_numel = int(flat_param.numel())
        full_numel = int(grad.numel())
        padded_numel = local_numel * world_size
        if full_numel <= local_numel or full_numel > padded_numel:
            raise RuntimeError(
                "Unsupported FSDP flat gradient shape mismatch: "
                f"local_numel={local_numel} grad_numel={full_numel} world_size={world_size}"
            )

        original_grad_dtype = grad.dtype
        full_grad = grad.detach().reshape(-1)
        reduce_dtype = handle._reduce_dtype
        if full_grad.dtype != reduce_dtype:
            full_grad = full_grad.to(dtype=reduce_dtype)
        if full_numel < padded_numel:
            full_grad = torch.nn.functional.pad(full_grad, (0, padded_numel - full_numel))
        if predivide > 1:
            full_grad.div_(predivide)

        local_grad = torch.empty(local_numel, dtype=full_grad.dtype, device=full_grad.device)
        dist.reduce_scatter_tensor(local_grad, full_grad, group=process_group)
        if postdivide > 1:
            local_grad.div_(postdivide)
        if local_grad.dtype != flat_param.dtype:
            local_grad = local_grad.to(dtype=flat_param.dtype)

        # The mismatched tensor was installed internally by FSDP, bypassing
        # Parameter.grad's public shape guard. Replace its storage in place.
        grad.data = local_grad.view_as(flat_param)
        repaired.append(
            {
                "handle": handle_idx,
                "local_numel": local_numel,
                "full_grad_numel": full_numel,
                "param_dtype": str(flat_param.dtype),
                "original_grad_dtype": str(original_grad_dtype),
                "reduce_dtype": str(reduce_dtype),
                "fqns": list(getattr(flat_param, "_fqns", ()))[:4],
            }
        )

    return {"scanned": scanned, "repaired": len(repaired), "examples": repaired[:8]}


@torch.no_grad()
def align_optimizer_grad_dtype_device_(optimizer):
    """Align FSDP-produced gradients with their optimizer parameters.

    FSDP may attach a low-precision gradient shard to a full-precision flat
    parameter (or vice versa). PyTorch's foreach Adam implementation groups
    parameters, gradients, and states by a common device/dtype and rejects
    such a pair before performing any update. Preserve the already-clipped
    gradient values while converting only mismatched gradient storage to the
    parameter's optimizer dtype/device.

    Returns a compact audit dictionary suitable for rank-local logging.
    """
    aligned = []
    scanned = 0
    for group_idx, param_group in enumerate(optimizer.param_groups):
        for param_idx, param in enumerate(param_group["params"]):
            grad = param.grad
            if grad is None:
                continue
            scanned += 1
            if grad.shape != param.shape:
                raise RuntimeError(
                    "Optimizer parameter/gradient shape mismatch remains after FSDP repair: "
                    f"group={group_idx} index={param_idx} param={tuple(param.shape)} grad={tuple(grad.shape)}"
                )
            if grad.dtype == param.dtype and grad.device == param.device:
                continue
            aligned.append(
                {
                    "group": group_idx,
                    "index": param_idx,
                    "param_dtype": str(param.dtype),
                    "grad_dtype": str(grad.dtype),
                    "param_device": str(param.device),
                    "grad_device": str(grad.device),
                    "numel": int(grad.numel()),
                }
            )
            # FSDP installs the mismatched shard internally, bypassing the
            # Parameter.grad assignment guard. Mutating the grad tensor's data
            # is therefore required before a type-correct reassignment would
            # be possible.
            grad.data = grad.data.to(device=param.device, dtype=param.dtype)

    return {
        "scanned": scanned,
        "aligned": len(aligned),
        "examples": aligned[:8],
    }

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import Shard

    fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard

    fully_shard_module = torch.distributed._composable.fsdp.fully_shard
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy, fully_shard_module = None, None, None, None, None


def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=get_device_id(), recurse=False)
        get_torch_device().empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    from accelerate import init_empty_weights

    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        if mesh is None:
            init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        else:
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context


# Copyright 2020-present the HuggingFace Inc. team.
# Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    """Get FSDP wrap policy for the module.

    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    # NOTE: This is a temporary workaround to be compatible with the OmegaConf & dataclass. We will remove this
    # once we have make all config in verl from OmegaConf to data class.
    def _get_attr(attr_name, default_value=None):
        if hasattr(config, "get"):
            return config.get(attr_name, default_value)
        else:
            return config.__getattribute__(attr_name)

    if _get_attr("disable", False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = _get_attr(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )
    min_num_params = _get_attr("min_num_params", 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:

        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy


@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    if fsdp_version(model) == 2:
        offload_fsdp2_model_to_cpu(model, empty_cache)
        return

    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    if fsdp_version(model) == 2:
        load_fsdp2_model_to_gpu(model)
        return

    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"
    device_id = get_device_id()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device(f"{get_device_name()}:{device_id}"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data


@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    device = get_device_id()
    model.to(device)


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


@contextmanager
def meta_device_init():
    """
    Create model parameters with meta device.

    Note buffers in model will still be initialized in default device (e.g., CPU),
    since the buffers can be non-persistent and filled with expected values that can
    NOT be captured in meta device.
    """
    device = torch.device("meta")
    old_register_parameter = nn.Module.register_parameter
    registered = set()

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        # we will skip register shared parameters as it
        # is already registered previously
        if param is not None and param not in registered:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            registered.add(module._parameters[name])

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        registered.clear()
        nn.Module.register_parameter = old_register_parameter


def parallel_load_safetensors(filepath):
    """
    Parallel load safetensors from huggingface checkpoint

    Huggingface checkpoint contains:

    - config.json: a json file for model configuration
    - model.safetensor.index.json: a json file for safetensors (parameters & buffers) index
    - model-000x-of-ooxx.safetensors: a binary file for safetensors (parameters & buffers) chunks

    Or (when model is small),

    - model.safetensors: a binary file for all parameters and buffers

    Each rank will own a part of model chunks and load them directly into GPU memory.
    """
    from safetensors.torch import load_file

    safetensors2param = {}

    index_file = os.path.join(filepath, "model.safetensors.index.json")
    if os.path.exists(index_file):
        index = json.load(open(index_file, "rb"))
        for param_name, filename in index["weight_map"].items():
            safetensors2param.setdefault(filename, []).append(param_name)
    else:
        # in this case, the model is small and we can load it all at once
        param_file = os.path.join(filepath, "model.safetensors")
        assert os.path.exists(param_file), f"Cannot find {param_file}"
        states = load_file(param_file)
        for param_name in states:
            safetensors2param.setdefault("model.safetensors", []).append(param_name)
        del states

    total_files = len(safetensors2param)
    ckpt_chunks = sorted(safetensors2param.keys())
    world_size = dist.get_world_size()
    size = int(math.ceil(total_files / world_size))
    ckpt_chunks = [ckpt_chunks[rank * size : rank * size + size] for rank in range(world_size)]

    shard_states = {}
    device = get_device_id()
    for rank, files in enumerate(ckpt_chunks):
        if rank == dist.get_rank():
            for file in files:
                file = os.path.join(filepath, file)
                states = load_file(file, device=device)
                # print(f"rank {rank} loading {file}...")
                shard_states.update(states)
        else:
            for file in files:
                for param_name in safetensors2param[file]:
                    shard_states[param_name] = rank
    return shard_states


def parallel_init_module_fn(module: torch.nn.Module, shard_states: dict[str, torch.nn.Parameter]):
    """
    Generate a function to initialize sub-modules in the `module` with `shard_states`
    from huggingface checkpoint.

    Args:
        module (torch.nn.Module): the global module to be initialized
        shard_states (Dict[str, torch.nn.Parameter]): the shard states from huggingface checkpoint

    Returns:
        init_fn (Callable): a function to initialize sub-modules in the `module` with `shard_states`
    """

    state2fqn = {}
    for name, state in itertools.chain(
        module.named_parameters(remove_duplicate=False), module.named_buffers(remove_duplicate=False)
    ):
        state2fqn.setdefault(state, []).append(name)
    # remove standalone parameters and buffers
    shared = {s for s, names in state2fqn.items() if len(names) > 1}
    materialized_states = {}

    @torch.no_grad()
    def create_and_sync_state(param_name, state, is_param):
        assert param_name in shard_states, f"{param_name} not loaded"
        device = get_device_id()
        if is_param:
            param = torch.nn.Parameter(torch.empty_like(state.data, device=device), requires_grad=state.requires_grad)
        else:  # buffer
            param = torch.empty_like(state.data, device=device)
        loaded = shard_states[param_name]
        if isinstance(loaded, torch.nn.Parameter | torch.Tensor):
            # NOTE: loaded.dtype can be different with param.dtype
            param.data.copy_(loaded.data)
            dist.broadcast(param.data, src=dist.get_rank())
        else:
            assert isinstance(loaded, int)  # the rank that holds the state
            dist.broadcast(param.data, src=loaded)
        shard_states.pop(param_name)
        del loaded
        return param

    def init_fn(sub_mod: torch.nn.Module, recurse: bool = True):
        param_and_buffers = tuple(sub_mod.named_parameters(recurse=False)) + tuple(sub_mod.named_buffers(recurse=False))
        # param_and_buffers = sorted(sub_mod.named_parameters(recurse=False), key=lambda x: x[0])
        for name, state in param_and_buffers:
            if not state.is_meta:
                continue
            is_param = name in sub_mod._parameters
            fqn = state2fqn[state].pop(0)
            # non-persistent buffers will not be saved in state dict, we can safely skip it
            if (not is_param) and fqn not in shard_states:
                if state.is_meta:
                    raise RuntimeError(
                        f"find a non-persistent buffer ({fqn}) initiated with device meta. Such buffer is not saved "
                        f"in checkpoint and user should guarantee to init in CPU / GPU device."
                    )
                continue
            # for shared parameter, we get it from the first time it is created
            if state in shared:
                if state not in materialized_states:
                    materialized_states[state] = create_and_sync_state(fqn, state, is_param)
                else:
                    if fqn in shard_states:
                        shard_states.pop(fqn)
                materialize_state = materialized_states[state]
            # for not shared parameter, we create it directly
            else:
                materialize_state = create_and_sync_state(fqn, state, is_param)
            if is_param:
                sub_mod._parameters[name] = materialize_state
            else:
                sub_mod._buffers[name] = materialize_state
        if recurse:
            for module in sub_mod.children():
                init_fn(module, recurse=True)

        # for debug
        # if len(shard_states) == 0: print("clear")
        return sub_mod

    return init_fn


def fsdp_version(model):
    if isinstance(model, FSDP):
        return 1
    elif isinstance(model, FSDPModule):
        return 2
    else:
        return 0


def get_fsdp_state_ctx(model, state_type, state_cfg, optim_cfg):
    if fsdp_version(model) == 1:
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)
    else:
        return nullcontext()


def get_fsdp_full_state_dict(model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True):
    """
    Get the full state dict from an FSDP model.

    Args:
        model (torch.nn.Module): The FSDP model to get state dict from
        offload_to_cpu (bool, optional): Whether to offload the state dict to CPU. Defaults to True.
        rank0_only (bool, optional): Whether to only get state dict on rank 0. Defaults to True.

    Returns:
        dict: The full state dict of the model

    Raises:
        NotImplementedError: If the FSDP version is unknown
    """
    if fsdp_version(model) == 1:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        state_dict_config = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)
        with get_fsdp_state_ctx(
            model, state_type=StateDictType.FULL_STATE_DICT, state_cfg=state_dict_config, optim_cfg=None
        ):
            state_dict = model.state_dict()
        return state_dict
    elif fsdp_version(model) == 2:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state_dict_config = StateDictOptions(
            full_state_dict=True, cpu_offload=offload_to_cpu, broadcast_from_rank0=not rank0_only
        )
        state_dict = get_model_state_dict(model, options=state_dict_config)
        return state_dict
    else:
        raise NotImplementedError(f"Unknown FSDP version {fsdp_version}")


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """

    if version.parse(torch.__version__) >= version.parse("2.7.0"):
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    else:
        # official torch 2.6.0 set_model_state_dict API leads to OOM
        # use torch 2.7.0 copy from verl/third_party/torch/distributed/checkpoint
        from verl.third_party.torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # To broadcast, it needs to be instantiated in the GPU.
    if dist.get_rank() == 0:
        model = model.to(device=get_device_id(), non_blocking=True)
    else:
        model = model.to_empty(device=get_device_id())

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True)
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(get_device_id())


@contextmanager
def maybe_patch_fsdp_module(model):
    if fully_shard_module is None:
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module


def apply_fsdp2(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f"wrap module {module.__class__.__name__}")
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #     print(f"wrap module {model.__class__.__name__}")
    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


def get_shard_placement_fn(fsdp_size):
    """Choose the dimension that can divide fsdp_size to avoid padding"""

    def shard_placement_fn(param):
        shape = list(param.shape)
        for i in range(len(shape)):
            if shape[i] % fsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return shard_placement_fn


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """torch.nn.utils.clip_grad_norm_ cann't run on cpu parameter DTensor"""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(get_device_id(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def layered_summon_lora_params(fsdp_module) -> OrderedDict:
    from peft.utils.save_and_load import get_peft_model_state_dict

    def __prefix_submodules(module, prefix):
        for name, submodule in module.named_modules():
            if name.startswith(prefix) and "." not in name[len(prefix) :]:
                yield name, submodule

    lora_params = OrderedDict()
    prefix_list = [
        # fsdp
        "_fsdp_wrapped_module.base_model.model.",
        "_fsdp_wrapped_module.base_model.model.model.",
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
        # fsdp2
        "base_model.model.",
        "base_model.model.model.",
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
    ]
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            prefix = name.replace("_fsdp_wrapped_module.base_model.model.", "base_model.model.")
            if name.endswith(".model") or name.endswith(".layers"):
                continue
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=submodule.state_dict())
                    sub_lora_params = {
                        f"{prefix}.{name}": param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in sub_lora_params.items()
                    }
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                get_torch_device().empty_cache()
    return lora_params


def collect_lora_params(module: FSDP, layered_summon: bool, base_sync_done: bool) -> OrderedDict:
    """
    collect lora params or full params if base model is not ready in vllm
    work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
    """
    from peft.utils.save_and_load import get_peft_model_state_dict

    lora_params = OrderedDict()
    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    if fsdp_version(module) > 0:
        if layered_summon:
            if not base_sync_done:
                raise ValueError(
                    "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                    "rollout.load_format=safetensors"
                )
            lora_params = layered_summon_lora_params(module)
        else:
            with FSDP.summon_full_params(module, writeback=False):
                if base_sync_done:
                    lora_params = get_peft_model_state_dict(peft_model)
                    lora_params = {
                        name: param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in lora_params.items()
                    }
                else:
                    model = peft_model.base_model.model
                    orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                        lora_params[name] = (
                            param.full_tensor().detach().cpu()
                            if hasattr(param, "full_tensor")
                            else param.detach().cpu()
                        )
                    model = model.to(orig_dev)
            get_torch_device().empty_cache()
    else:
        if base_sync_done:
            lora_params = get_peft_model_state_dict(peft_model)
        else:
            model = peft_model.base_model.model
            orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
            model = model.to("cpu")
            for name, param in model.state_dict().items():
                if any(x in name for x in ["_flat_param", "lora_"]):
                    continue
                name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                lora_params[name] = param.detach().cpu()
            model = model.to(orig_dev)
    return lora_params


def replace_lora_wrapper(k, peft_config):
    """Replace LoRA parameter keys with base layer equivalents.

    Transforms LoRA parameter names to their corresponding base layer
    names for proper weight loading in vLLM when base model sync is not done.

    Args:
        k (str): Original parameter key name.

    Returns:
        str: Transformed parameter key for base layer.
    """
    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if k.endswith(".weight"):
        module_k = k[: -len(".weight")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.weight"
    if k.endswith(".bias"):
        module_k = k[: -len(".bias")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.bias"
    return k
