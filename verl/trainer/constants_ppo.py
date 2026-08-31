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

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # symmetric memory allreduce not work properly in spmd mode
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
        # Force eager attention: flash_attn 2.8.1 requires GLIBC_2.32 but nodes run 2.31
        "TRANSFORMERS_ATTN_IMPLEMENTATION": "eager",
        # Offline mode: nodes cannot reach huggingface.co
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # Cambrian-S external implementation path; Ray workers need this to
        # import vagen.models.cambrian_register and register cambrian_qwen.
        "CAMBRIAN_SRC": "/mnt/umm/users/yinbaiqiao/cambrian-s",
            # Triton/vllm JIT compilation needs a C compiler.
            # These pods do not have /usr/bin/gcc, so use the shared conda env.
            "CC": "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/x86_64-conda-linux-gnu-gcc",
            "CXX": "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/x86_64-conda-linux-gnu-g++",
            # gsplat local rendering JITs a CUDA extension; use conda CUDA toolkit.
            "CUDA_HOME": "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "TORCH_EXTENSIONS_DIR": "/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_yinbaiqiao",
            "CPATH": "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/targets/x86_64-linux/include",
            "CPLUS_INCLUDE_PATH": "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/targets/x86_64-linux/include",
            "LIBRARY_PATH": "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/targets/x86_64-linux/lib",
            # Keep vllm torch.compile disabled to avoid extra JIT memory pressure.
            "VLLM_TORCH_COMPILE_LEVEL": "0",
            "TORCH_COMPILE_DISABLE": "1",
            # Disable optional vLLM CUDA JIT paths; gsplat is the only CUDA JIT we need.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_USE_DEEP_GEMM": "0",
            "VLLM_SKIP_DEEP_GEMM_WARMUP": "1",
            # WandB offline mode (nodes have no network access / no API key)
            "WANDB_MODE": "offline",
            # Qwen2.5-VL: FLASH_ATTN/XFORMERS/FLASHINFER unavailable here → TORCH_SDPA
            "VLLM_ATTENTION_BACKEND": "TORCH_SDPA",
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    # Prefer parent-process values when present, but NEVER drop critical vars
    # that Ray workers must see explicitly (inheritance is unreliable for actors).
    _ALWAYS_PROPAGATE = {
        "WANDB_MODE",
        "WANDB_DIR",
        "WANDB_API_KEY",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TRANSFORMERS_ATTN_IMPLEMENTATION",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_TORCH_COMPILE_LEVEL",
        "TORCH_COMPILE_DISABLE",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_USE_DEEP_GEMM",
        "VLLM_SKIP_DEEP_GEMM_WARMUP",
        "CC",
        "CXX",
        "CUDA_HOME",
        "TORCH_CUDA_ARCH_LIST",
        "TORCH_EXTENSIONS_DIR",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "CAMBRIAN_SRC",
        "VAGEN_D0_3_SMOKE_DIR",
        "VAGEN_D0_4_DIR",
        "VAGEN_D0_6_DIR",
        "VAGEN_D0_6_ABORT_ON_WEIGHT_MISMATCH",
        "VAGEN_D0_6_MODEL_CHECKPOINT",
        "VAGEN_VISION_SANITY_DIR",
        "VAGEN_VISION_SANITY_MAX",
    }
    for key in list(runtime_env["env_vars"].keys()):
        if key in _ALWAYS_PROPAGATE:
            # Keep (and refresh from parent if set)
            if os.environ.get(key) is not None:
                runtime_env["env_vars"][key] = os.environ[key]
            continue
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    for key in _ALWAYS_PROPAGATE:
        if os.environ.get(key) is not None:
            runtime_env["env_vars"][key] = os.environ[key]
    return runtime_env
