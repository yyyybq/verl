# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import json
import os
import time

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    FSDPModule,
    align_optimizer_grad_dtype_device_,
    fsdp1_clip_grad_norm_,
    fsdp2_clip_grad_norm_,
    repair_unreduced_fsdp1_grad_shards_,
)
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _d0_3_group_name(name: str) -> str:
    if "flat_param" in name or "_flat_param" in name:
        return "fsdp_flat_mixed"
    if "vision_tower" in name:
        return "vision_tower"
    if "mm_projector" in name:
        return "mm_projector"
    if "nfp_head" in name:
        return "nfp_head"
    if "image_newline" in name:
        return "miv_related"
    if name.startswith("lm_head") or ".lm_head" in name:
        return "lm_head"
    marker = "model.layers."
    if marker in name:
        try:
            idx = int(name.split(marker, 1)[1].split(".", 1)[0])
        except Exception:
            return "llm_other"
        if idx < 9:
            return "llm_early"
        if idx < 19:
            return "llm_middle"
        return "llm_late"
    if "embed_tokens" in name or "model.norm" in name:
        return "llm_other"
    return "other"


def _d0_3_empty_group() -> dict:
    return {
        "requires_grad_params": 0,
        "optimizer_member_params": 0,
        "params_with_grad": 0,
        "grad_norm_sq": 0.0,
        "update_norm_sq": 0.0,
    }


def _d0_13_marker(stage: str, **extra) -> None:
    out_dir = os.environ.get("VAGEN_D0_13_TIMELINE_DIR")
    if not out_dir:
        return
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:
        rank = -1
    rec = {
        "stage": stage,
        "rank": int(rank),
        "pid": int(os.getpid()),
        "time": time.time(),
        **extra,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"actor_rank{rank}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    @staticmethod
    def _extract_auxiliary_loss(output) -> torch.Tensor | None:
        """Return a scalar auxiliary loss from model outputs, if one is present."""
        aux_loss = getattr(output, "loss", None)
        if aux_loss is None and isinstance(output, (tuple, list)) and len(output) > 0:
            first = output[0]
            if torch.is_tensor(first) and first.dim() == 0:
                aux_loss = first
        if aux_loss is None:
            return None
        if isinstance(aux_loss, (tuple, list)):
            aux_loss = aux_loss[0] if aux_loss else None
        if aux_loss is None:
            return None
        if not torch.is_tensor(aux_loss):
            aux_loss = torch.as_tensor(aux_loss)
        return aux_loss.mean()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, return_aux_loss=False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            aux_loss: optional scalar auxiliary model loss
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            if return_aux_loss:
                # the model line's NFP loss (2026-11-03): its statistics for the metrics (vagen/models/cambrian_nfp.py);
                # FSDP rebuilds the output and drops that attribute, so they come from the adapter's kept copy (11-04)
                from vagen.models.cambrian_nfp import take_stats

                self._last_nfp_stats = take_stats(output)
                return entropy, log_probs, self._extract_auxiliary_loss(output)
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        _d0_13_marker("before_grad_clip")
        if isinstance(self.actor_module, FSDP):
            fsdp_grad_repair = repair_unreduced_fsdp1_grad_shards_(self.actor_module)
            if fsdp_grad_repair["repaired"]:
                _d0_13_marker("fsdp_unreduced_grad_repair", **fsdp_grad_repair)
                if torch.distributed.get_rank() == 0:
                    logger.warning(
                        "Reduce-scattered %s/%s unexpectedly unsharded FSDP gradients; examples=%s",
                        fsdp_grad_repair["repaired"],
                        fsdp_grad_repair["scanned"],
                        fsdp_grad_repair["examples"],
                    )
            grad_dtypes = sorted(
                {str(param.grad.dtype) for param in self.actor_module.parameters() if param.grad is not None}
            )
            if len(grad_dtypes) > 1:
                _d0_13_marker("mixed_dtype_grad_clip", grad_dtypes=grad_dtypes)
                if torch.distributed.get_rank() == 0:
                    logger.warning("Using mixed-dtype FSDP grad clipping for gradients: %s", grad_dtypes)
            grad_norm = fsdp1_clip_grad_norm_(self.actor_module, max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        _d0_13_marker(
            "after_grad_clip",
            grad_norm=float(grad_norm.detach().float().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
        )

        optimizer_grad_alignment = align_optimizer_grad_dtype_device_(self.actor_optimizer)
        if optimizer_grad_alignment["aligned"]:
            _d0_13_marker("optimizer_grad_alignment", **optimizer_grad_alignment)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.warning(
                    "Aligned %s/%s optimizer gradients to parameter dtype/device; examples=%s",
                    optimizer_grad_alignment["aligned"],
                    optimizer_grad_alignment["scanned"],
                    optimizer_grad_alignment["examples"],
                )

        d0_3_dir = os.environ.get("VAGEN_D0_3_SMOKE_DIR")
        d0_3_before = None
        d0_3_stats = None
        if d0_3_dir and self.actor_optimizer is not None:
            opt_ids = {id(p) for group in self.actor_optimizer.param_groups for p in group.get("params", [])}
            d0_3_before = {}
            d0_3_stats = {}
            for name, p in self.actor_module.named_parameters():
                group_name = _d0_3_group_name(name)
                rec = d0_3_stats.setdefault(group_name, _d0_3_empty_group())
                n = int(p.numel())
                if p.requires_grad:
                    rec["requires_grad_params"] += n
                if id(p) in opt_ids:
                    rec["optimizer_member_params"] += n
                if p.grad is not None:
                    rec["params_with_grad"] += n
                    g = p.grad.detach().float()
                    rec["grad_norm_sq"] += float(torch.sum(g * g).detach().cpu().item())
                if p.requires_grad and id(p) in opt_ids:
                    d0_3_before[name] = p.detach().float().cpu().clone()

        # if grad_norm is not finite, skip the update
        _d0_13_marker("before_optimizer_step")
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        _d0_13_marker("after_optimizer_step")
        if d0_3_dir and d0_3_before is not None and d0_3_stats is not None:
            for name, p in self.actor_module.named_parameters():
                before = d0_3_before.get(name)
                if before is None:
                    continue
                group_name = _d0_3_group_name(name)
                diff = p.detach().float().cpu() - before
                d0_3_stats[group_name]["update_norm_sq"] += float(torch.sum(diff * diff).item())
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            out_stats = {}
            for group_name, rec in d0_3_stats.items():
                out_stats[group_name] = {
                    "requires_grad_params": rec["requires_grad_params"],
                    "optimizer_member_params": rec["optimizer_member_params"],
                    "params_with_grad": rec["params_with_grad"],
                    "grad_norm": rec["grad_norm_sq"] ** 0.5,
                    "update_norm": rec["update_norm_sq"] ** 0.5,
                }
            os.makedirs(d0_3_dir, exist_ok=True)
            with open(os.path.join(d0_3_dir, f"optimizer_audit_rank{rank}.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "rank": int(rank),
                        "grad_clip_norm": float(grad_norm.detach().float().cpu().item()),
                        "note": "FSDP use_orig_params=false may expose flat/sharded parameters as fsdp_flat_mixed.",
                        "groups": out_stats,
                    },
                f,
                indent=2,
                ensure_ascii=False,
            )
            _d0_13_marker("after_optimizer_audit_write")
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        _d0_13_marker(
            "update_policy_start",
            minibatches=len(mini_batches),
            ppo_epochs=int(self.config.ppo_epochs),
        )
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob, aux_loss = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_aux_loss=True,
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if aux_loss is not None:
                        # Prefer image_gen_loss_coef (U1 FM) when set; else nfp_loss_coef (Cambrian).
                        image_gen_coef = self.config.get("image_gen_loss_coef", None)
                        nfp_loss_coef = (
                            float(image_gen_coef)
                            if image_gen_coef is not None
                            else self.config.get("nfp_loss_coef", 1.0)
                        )
                        aux_loss = aux_loss.to(policy_loss.device)
                        policy_loss = policy_loss + aux_loss * nfp_loss_coef
                        # with the model line's statistics (an episode without a pair is not counted); else as before
                        from vagen.models.cambrian_nfp import nfp_metrics

                        micro_batch_metrics.update(nfp_metrics(aux_loss.detach().item(),
                                                               getattr(self, "_last_nfp_stats", None),
                                                               nfp_loss_coef, loss_scale_factor))
                    else:
                        micro_batch_metrics["actor/nfp_loss_active"] = 0.0

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        _d0_13_marker("before_backward", batch_idx=int(batch_idx))
                        self.scaler.scale(loss).backward()
                    else:
                        _d0_13_marker("before_backward", batch_idx=int(batch_idx))
                        loss.backward()
                    _d0_13_marker("after_backward", batch_idx=int(batch_idx))

                    # U1 Plan B two-pass FM: PG backward above frees und activations;
                    # then gen-only FM backward (stashed during forward when U1_FM_BACKPROP=1).
                    def _unwrap_actor_module(m):
                        cur = m
                        for _ in range(6):
                            if hasattr(cur, "compute_pending_fm_aux_loss"):
                                return cur
                            if hasattr(cur, "module"):
                                cur = cur.module
                                continue
                            break
                        return m

                    u1_mod = _unwrap_actor_module(self.actor_module)
                    if isinstance(self.actor_module, FSDP) and hasattr(u1_mod, "compute_pending_fm_aux_loss"):
                        # Keep the pending-only FM pass inside the root FSDP
                        # forward lifecycle. Directly calling the unwrapped
                        # method after the PG forward/backward accesses
                        # resharded (zero-storage) root parameters.
                        fm_loss = self.actor_module(_u1_compute_pending_only=True)
                    elif hasattr(u1_mod, "compute_pending_fm_aux_loss"):
                        fm_loss = u1_mod.compute_pending_fm_aux_loss()
                    else:
                        fm_loss = None
                    if fm_loss is not None:
                        image_gen_coef = self.config.get("image_gen_loss_coef", None)
                        nfp_loss_coef = (
                            float(image_gen_coef)
                            if image_gen_coef is not None
                            else float(self.config.get("nfp_loss_coef", 1.0))
                        )
                        fm_scaled = fm_loss.to(loss.device) * nfp_loss_coef * loss_scale_factor
                        if self.scaler is not None:
                            self.scaler.scale(fm_scaled).backward()
                        else:
                            fm_scaled.backward()
                        micro_batch_metrics["actor/nfp_loss"] = float(fm_loss.detach().item())
                        micro_batch_metrics["actor/nfp_loss_scaled"] = float(
                            fm_loss.detach().item() * nfp_loss_coef * loss_scale_factor
                        )
                        micro_batch_metrics["actor/nfp_loss_coef"] = nfp_loss_coef
                        micro_batch_metrics["actor/image_gen_loss_coef"] = nfp_loss_coef
                        micro_batch_metrics["actor/nfp_loss_active"] = 1.0

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    # Explicit aliases for gated diagnostics / smoke dashboards
                    micro_batch_metrics["actor/pg_loss_raw"] = float(pg_loss.detach().item())
                    micro_batch_metrics["actor/pg_loss_weighted"] = float(pg_loss.detach().item() * loss_scale_factor)
                    micro_batch_metrics["actor/total_loss"] = float(loss.detach().item())
                    if entropy_coeff != 0:
                        micro_batch_metrics["actor/entropy_loss"] = float(entropy_loss.detach().item())
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                _d0_13_marker("before_metrics_reduce", batch_idx=int(batch_idx))
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
                _d0_13_marker("after_metrics_reduce", metric_keys=len(metrics))
        self.actor_optimizer.zero_grad()
        _d0_13_marker("before_worker_return", metric_keys=len(metrics))
        return metrics
