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

import inspect
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Mapping

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, VLLM_SLEEP_LEVEL
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.device import get_device_id, get_device_name, get_torch_device, set_expandable_segments
from verl.utils.fsdp_utils import (
    fsdp_version,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.utils.import_utils import deprecated
from verl.utils.model import check_exclude_modules, check_target_modules, convert_weight_keys
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.utils.torch_functional import check_device_is_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _d0_6_tensor(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.detach()


def _d0_6_checksum(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().float().contiguous().cpu()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def _d0_6_sample_values(tensor: torch.Tensor) -> list[float]:
    flat = tensor.detach().flatten()
    if flat.numel() == 0:
        return []
    idxs = sorted(set([0, flat.numel() // 2, flat.numel() - 1]))
    return [float(flat[i].detach().float().cpu().item()) for i in idxs]


def _d0_6_pair_stats(source: torch.Tensor, target: torch.Tensor) -> dict:
    src = _d0_6_tensor(source)
    tgt = _d0_6_tensor(target)
    base = {
        "source_shape": list(src.shape),
        "target_shape": list(tgt.shape),
        "source_dtype": str(src.dtype),
        "target_dtype": str(tgt.dtype),
        "source_checksum": _d0_6_checksum(src),
        "target_checksum": _d0_6_checksum(tgt),
        "source_sample_values": _d0_6_sample_values(src),
        "target_sample_values": _d0_6_sample_values(tgt),
    }
    if tuple(src.shape) != tuple(tgt.shape):
        base.update(
            {
                "shape_mismatch": True,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "l2": None,
                "cosine": None,
            }
        )
        return base
    src_f = src.detach().float().cpu()
    tgt_f = tgt.detach().float().cpu()
    diff = (src_f - tgt_f).abs()
    src_cast = src.detach().to(dtype=tgt.dtype).cpu()
    tgt_cast_ref = tgt.detach().cpu()
    cast_diff = (src_cast.float() - tgt_cast_ref.float()).abs()
    src_flat = src_f.flatten()
    tgt_flat = tgt_f.flatten()
    denom = torch.linalg.vector_norm(src_flat) * torch.linalg.vector_norm(tgt_flat)
    cosine = None if float(denom.detach().cpu().item()) == 0.0 else float(torch.dot(src_flat, tgt_flat).div(denom).detach().cpu().item())
    base.update({
        "shape_mismatch": False,
        "strict_exact_equal": str(src.dtype) == str(tgt.dtype) and torch.equal(src.cpu(), tgt.cpu()),
        "max_abs_diff": float(diff.max().detach().cpu().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().detach().cpu().item()) if diff.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(diff.flatten()).detach().cpu().item()) if diff.numel() else 0.0,
        "cosine": cosine,
        "source_cast_to_target_dtype": str(tgt.dtype),
        "source_cast_to_target_checksum": _d0_6_checksum(src_cast),
        "source_cast_to_target_sample_values": _d0_6_sample_values(src_cast),
        "source_cast_equals_target": torch.equal(src_cast, tgt_cast_ref),
        "max_abs_diff_after_source_cast": float(cast_diff.max().detach().cpu().item()) if cast_diff.numel() else 0.0,
        "mean_abs_diff_after_source_cast": float(cast_diff.mean().detach().cpu().item()) if cast_diff.numel() else 0.0,
    })
    return base


def _d0_6_slice_rows(tensor: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    n = tensor.shape[0]
    per = n // tp_size
    return tensor[tp_rank * per:(tp_rank + 1) * per]


def _d0_6_slice_cols(tensor: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    n = tensor.shape[1]
    per = n // tp_size
    return tensor[:, tp_rank * per:(tp_rank + 1) * per]


def _d0_6_slice_rows_unless_already_sharded(tensor: torch.Tensor, target: torch.Tensor | None, tp_rank: int, tp_size: int) -> torch.Tensor:
    if target is not None and tuple(tensor.shape) == tuple(target.shape):
        return tensor
    return _d0_6_slice_rows(tensor, tp_rank, tp_size)


def _d0_6_qkv_expected(updated_params: Mapping[str, torch.Tensor], layer: int, tp_rank: int, tp_size: int, suffix: str) -> torch.Tensor:
    q = _d0_6_tensor(updated_params[f"model.layers.{layer}.self_attn.q_proj.{suffix}"])
    k = _d0_6_tensor(updated_params[f"model.layers.{layer}.self_attn.k_proj.{suffix}"])
    v = _d0_6_tensor(updated_params[f"model.layers.{layer}.self_attn.v_proj.{suffix}"])
    return torch.cat([
        _d0_6_slice_rows(q, tp_rank, tp_size),
        _d0_6_slice_rows(k, tp_rank, tp_size),
        _d0_6_slice_rows(v, tp_rank, tp_size),
    ], dim=0)


def _d0_6_gate_up_expected(updated_params: Mapping[str, torch.Tensor], layer: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    gate = _d0_6_tensor(updated_params[f"model.layers.{layer}.mlp.gate_proj.weight"])
    up = _d0_6_tensor(updated_params[f"model.layers.{layer}.mlp.up_proj.weight"])
    return torch.cat([
        _d0_6_slice_rows(gate, tp_rank, tp_size),
        _d0_6_slice_rows(up, tp_rank, tp_size),
    ], dim=0)


def d0_6_required_source_weight_names() -> set[str]:
    names = {
        "model.vision_tower_aux_list.0.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "model.vision_tower_aux_list.0.vision_tower.vision_model.encoder.layers.13.self_attn.q_proj.weight",
        "model.vision_tower_aux_list.0.vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.weight",
        "model.vision_tower_aux_list.0.vision_tower.vision_model.post_layernorm.weight",
        "model.mm_projector.0.weight",
        "model.mm_projector.0.bias",
        "model.mm_projector.2.weight",
        "model.mm_projector.2.bias",
        "model.image_newline",
        "model.norm.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    }
    for layer in (0, 14, 27):
        names.update(
            {
                f"model.layers.{layer}.self_attn.q_proj.weight",
                f"model.layers.{layer}.self_attn.k_proj.weight",
                f"model.layers.{layer}.self_attn.v_proj.weight",
                f"model.layers.{layer}.self_attn.q_proj.bias",
                f"model.layers.{layer}.self_attn.k_proj.bias",
                f"model.layers.{layer}.self_attn.v_proj.bias",
                f"model.layers.{layer}.self_attn.o_proj.weight",
                f"model.layers.{layer}.mlp.gate_proj.weight",
                f"model.layers.{layer}.mlp.up_proj.weight",
                f"model.layers.{layer}.mlp.down_proj.weight",
            }
        )
    return names


def _d0_6_compare_weight_sync(model, updated_params, *, out_dir: str, tp_rank: int, tp_size: int) -> dict:
    params = dict(model.named_parameters())
    records = {}
    missing = []

    def add(name: str, source, target_name: str):
        target = params.get(target_name)
        if source is None or target is None:
            missing.append({"name": name, "target_name": target_name, "source_missing": source is None, "target_missing": target is None})
            return
        records[name] = {"target_name": target_name, **_d0_6_pair_stats(source, target)}

    # Full replicated Cambrian multimodal tensors.
    direct = {
        "vision_layer0_q": (
            "model.vision_tower_aux_list.0.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        ),
        "vision_layer13_q": (
            "model.vision_tower_aux_list.0.vision_tower.vision_model.encoder.layers.13.self_attn.q_proj.weight",
            "vision_tower.vision_model.encoder.layers.13.self_attn.q_proj.weight",
        ),
        "vision_layer25_q": (
            "model.vision_tower_aux_list.0.vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.weight",
            "vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.weight",
        ),
        "vision_post_layernorm": (
            "model.vision_tower_aux_list.0.vision_tower.vision_model.post_layernorm.weight",
            "vision_tower.vision_model.post_layernorm.weight",
        ),
        "mm_projector_0_weight": ("model.mm_projector.0.weight", "mm_projector.0.weight"),
        "mm_projector_0_bias": ("model.mm_projector.0.bias", "mm_projector.0.bias"),
        "mm_projector_2_weight": ("model.mm_projector.2.weight", "mm_projector.2.weight"),
        "mm_projector_2_bias": ("model.mm_projector.2.bias", "mm_projector.2.bias"),
        "image_newline": ("model.image_newline", "image_newline"),
        "llm_final_norm": ("model.norm.weight", "language_model.model.norm.weight"),
    }
    for label, (src_name, tgt_name) in direct.items():
        add(label, updated_params.get(src_name), tgt_name)

    # Tensor-parallel LLM tensors.
    embed_src = updated_params.get("model.embed_tokens.weight")
    lm_head_src = updated_params.get("lm_head.weight")
    embed_tgt = params.get("language_model.model.embed_tokens.weight")
    lm_head_tgt = params.get("language_model.lm_head.weight")
    add("embed_tokens_shard", None if embed_src is None else _d0_6_slice_rows_unless_already_sharded(_d0_6_tensor(embed_src), embed_tgt, tp_rank, tp_size), "language_model.model.embed_tokens.weight")
    add("lm_head_shard", None if lm_head_src is None else _d0_6_slice_rows_unless_already_sharded(_d0_6_tensor(lm_head_src), lm_head_tgt, tp_rank, tp_size), "language_model.lm_head.weight")

    for layer in (0, 14, 27):
        pfx = f"language_model.model.layers.{layer}"
        try:
            add(f"layer{layer}_qkv_weight_shard", _d0_6_qkv_expected(updated_params, layer, tp_rank, tp_size, "weight"), f"{pfx}.self_attn.qkv_proj.weight")
            add(f"layer{layer}_qkv_bias_shard", _d0_6_qkv_expected(updated_params, layer, tp_rank, tp_size, "bias"), f"{pfx}.self_attn.qkv_proj.bias")
            add(f"layer{layer}_o_weight_col_shard", _d0_6_slice_cols(_d0_6_tensor(updated_params[f"model.layers.{layer}.self_attn.o_proj.weight"]), tp_rank, tp_size), f"{pfx}.self_attn.o_proj.weight")
            add(f"layer{layer}_gate_up_weight_shard", _d0_6_gate_up_expected(updated_params, layer, tp_rank, tp_size), f"{pfx}.mlp.gate_up_proj.weight")
            add(f"layer{layer}_down_weight_col_shard", _d0_6_slice_cols(_d0_6_tensor(updated_params[f"model.layers.{layer}.mlp.down_proj.weight"]), tp_rank, tp_size), f"{pfx}.mlp.down_proj.weight")
        except Exception as exc:
            missing.append({"name": f"layer{layer}_packed_exception", "error": repr(exc)})

    strict_all_equal = all((not r.get("shape_mismatch")) and r.get("strict_exact_equal") for r in records.values()) and not missing
    cast_all_equal = all((not r.get("shape_mismatch")) and r.get("source_cast_equals_target") for r in records.values()) and not missing
    payload = {
        "tag": "d0_6_weight_equality_after_fsdp_to_vllm_sync",
        "rank": int(os.environ.get("RANK", "-1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "tp_rank": int(tp_rank),
        "tp_size": int(tp_size),
        "compared_tensor_count": len(records),
        "missing": missing,
        "strict_all_compared_equal": strict_all_equal,
        "all_compared_equal": cast_all_equal,
        "all_compared_equal_after_source_cast_to_target_dtype": cast_all_equal,
        "tensors": records,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"d0_6_weight_equality_rank{payload['rank']}_tp{tp_rank}_pid{os.getpid()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    if os.environ.get("VAGEN_D0_6_ABORT_ON_WEIGHT_MISMATCH") == "1" and not cast_all_equal:
        raise RuntimeError(f"D0.6 weight equality failed; wrote {path}")
    return payload


@deprecated()
class FSDPVLLMShardingManager(BaseShardingManager):
    """Sharding manager for FSDP models with vLLM inference engine integration.

    Manages parameter synchronization between FSDP training models and vLLM
    inference engines, handling both full parameters and LoRA adapters with
    efficient memory management and device placement.
    """

    @check_device_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        rollout_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        load_format: str = "dummy_hf",
        layered_summon: bool = True,
    ):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer initialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine
        # self.model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if
        # inference_engine else None

        self.model_runner = (
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
            if self.inference_engine
            else None
        )

        self.model_config = model_config
        self.rollout_config = rollout_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig()
            )
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = "dummy" not in load_format
        if is_version_ge(pkg="vllm", minver="0.7.3"):
            VLLMHijack.hijack()

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        def __collect_lora_params() -> OrderedDict:
            """
            collect lora params or full params if base model is not ready in vllm
            work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
            """
            from peft.utils.save_and_load import get_peft_model_state_dict

            lora_params = OrderedDict()
            peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
            if fsdp_version(self.module) > 0:
                if self.layered_summon:
                    if not self.base_sync_done:
                        raise ValueError(
                            "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                            "rollout.load_format=safetensors"
                        )
                    lora_params = layered_summon_lora_params(self.module)
                else:
                    with FSDP.summon_full_params(self.module, writeback=False):
                        if self.base_sync_done:
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
                if self.base_sync_done:
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

        # NOTE: Basically, we only need `get_torch_device().empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        self.timing = {}
        with simple_timer("reshard", self.timing):
            get_torch_device().empty_cache()

            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            if self.offload_param:
                load_fsdp_model_to_gpu(self.module)

            peft_config = None
            peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
            if hasattr(peft_model, "peft_config"):
                peft_config = peft_model.peft_config.get("default", None)
                params = __collect_lora_params()
            else:
                params = self.module.state_dict()
            params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

            # vllm need to set _set_allocator_settings to False
            logger.debug("fsdp vllm sharding_manager _set_allocator_settings to False")
            set_expandable_segments(False)

            if self.rollout_config.free_cache_engine:
                if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                    self.inference_engine.wake_up(tags=["weights"])
                else:
                    self.inference_engine.wake_up()

            # update model params
            self.update_params(params, peft_config=peft_config)
            d0_6_dir = os.environ.get("VAGEN_D0_6_DIR")
            if d0_6_dir and peft_config is None:
                _d0_6_compare_weight_sync(
                    self.model_runner.model,
                    params,
                    out_dir=d0_6_dir,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
            get_torch_device().empty_cache()

            if (
                self.rollout_config.free_cache_engine
                and "tags" in inspect.signature(self.inference_engine.wake_up).parameters
            ):
                self.inference_engine.wake_up(tags=["kv_cache"])

            log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if self.rollout_config.free_cache_engine:
            self.inference_engine.sleep(level=VLLM_SLEEP_LEVEL)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # _set_allocator_settings to True is required by fsdp2 to avoid oom
        logger.debug("fsdp vllm sharding_manager _set_allocator_settings to True")
        set_expandable_segments(True)

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config=None):
        """Update model parameters in the vLLM inference engine.

        Synchronizes parameters from the FSDP training model to the vLLM inference
        engine, handling both full model parameters and LoRA adapters with proper
        device placement and memory management.

        Args:
            updated_params (dict): Dictionary of parameter names to tensor values.
            peft_config (optional): PEFT configuration for LoRA adapters.
        """
        model = self.model_runner.model
        if peft_config:
            if self.base_sync_done:
                lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
                lora_reqest = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                self.inference_engine.llm_engine.add_lora(lora_reqest)
                logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}")
                return
            else:

                def replace_lora_wrapper(k):
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
                        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(
                            peft_config, module_k
                        ):
                            return f"{module_k}.base_layer.weight"
                    if k.endswith(".bias"):
                        module_k = k[: -len(".bias")]
                        if check_exclude_modules(peft_config, module_k):
                            return k
                        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(
                            peft_config, module_k
                        ):
                            return f"{module_k}.base_layer.bias"
                    return k

                updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        patch_vllm_moe_model_weight_loader(model)
        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        loaded_params = model.load_weights(
            (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in updated_params.items()
            )
        )

        self.base_sync_done = True
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")
