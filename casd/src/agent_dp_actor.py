import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

from casd.src.core_algos import agg_loss, compute_policy_loss, kl_penalty

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):

        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)

                logits_rmpad.div_(temperature)

                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                if self.use_ulysses_sp:

                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)

            return entropy, log_probs

    def _cat_log(self, text):

        import os
        try:
            import torch.distributed as _d
            _rk = _d.get_rank() if (_d.is_available() and _d.is_initialized()) else 0
        except Exception:
            _rk = 0
        line = f"[rank{_rk}] {text}"
        print(line, flush=True)
        f = os.environ.get("TRAIN_LOG_FILE", "train_log.txt")
        if f and f.lower() not in ("off", "none", ""):
            try:
                with open(f, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:
                pass

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:

        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:

            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):

        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'action_mask']
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        use_catalyst = "has_y1" in data.batch.keys()
        if use_catalyst:
            select_keys += ['y1_input_ids', 'y1_attention_mask', 'y1_position_ids',
                            'y1_responses', 'y1_action_mask', 'has_y1']

            if "y1aug_input_ids" in data.batch.keys():
                select_keys += ['y1aug_input_ids', 'y1aug_attention_mask', 'y1aug_position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        _l_int_samples = []

        _li_ce_sum, _li_n, _li_rows, _dlogp_wsum = 0.0, 0, 0, 0.0

        _li_wce_sum = 0.0

        _cat_n_sum = 0.0

        self._bar_dbg_n = 0
        self._bar_warn_n = 0
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):

                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu

                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                _cat_coef_mb = float(self.config.get("catalyst_coef", 0.0) or 0.0)
                _cat_n_global, _cat_world = 1.0, 1
                if use_catalyst and _cat_coef_mb > 0:
                    import torch.distributed as _dist

                    _mb_t = mini_batch.batch if hasattr(mini_batch, "batch") else mini_batch
                    _nv = torch.tensor([float(_mb_t["has_y1"].sum().item())],
                                       dtype=torch.float64, device=torch.cuda.current_device())
                    if _dist.is_available() and _dist.is_initialized():
                        _dist.all_reduce(_nv, op=_dist.ReduceOp.SUM)
                        _cat_world = _dist.get_world_size()
                    _cat_n_raw = float(_nv.item())
                    _cat_n_sum += _cat_n_raw

                    _cat_n_global = max(_cat_n_raw, 1.0)

                for data in micro_batches:

                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())

                    action_mask = data['action_mask']
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        action_mask=action_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=action_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]

                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=action_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    catalyst_coef = self.config.get("catalyst_coef", 0.0)
                    _cat_term = 0.0
                    if use_catalyst and catalyst_coef > 0:

                        y1_attention_mask = data["y1_attention_mask"]
                        if y1_attention_mask.sum() == 0:
                            y1_attention_mask = y1_attention_mask.clone()
                            y1_attention_mask[0, 0] = 1
                        y1_micro = {
                            "responses": data["y1_responses"],
                            "input_ids": data["y1_input_ids"],
                            "attention_mask": y1_attention_mask,
                            "position_ids": data["y1_position_ids"],
                        }
                        _, y1_log_prob = self._forward_micro_batch(
                            micro_batch=y1_micro, temperature=temperature, calculate_entropy=False)

                        y1_mask = data["y1_action_mask"].to(y1_log_prob.dtype) * data["has_y1"].unsqueeze(-1).to(y1_log_prob.dtype)

                        _cmode = self.config.get("catalyst_mode", "sft")
                        _dlogp_log = 0.0
                        if _cmode == "barrier" and "y1aug_input_ids" in data:
                            y1aug_att = data["y1aug_attention_mask"]
                            if y1aug_att.sum() == 0:
                                y1aug_att = y1aug_att.clone(); y1aug_att[0, 0] = 1
                            y1aug_micro = {"responses": data["y1_responses"],
                                           "input_ids": data["y1aug_input_ids"],
                                           "attention_mask": y1aug_att,
                                           "position_ids": data["y1aug_position_ids"]}
                            with torch.no_grad():
                                _, y1aug_log_prob = self._forward_micro_batch(
                                    micro_batch=y1aug_micro, temperature=temperature, calculate_entropy=False)
                            _clip = self.config.get("catalyst_barrier_clip", 4.0)
                            _beta = self.config.get("catalyst_barrier_beta", 1.0)
                            dlogp = (y1aug_log_prob - y1_log_prob.detach()).clamp(min=0.0, max=_clip)
                            w = 1.0 + _beta * dlogp

                            _wm = y1_mask.sum(dim=-1, keepdim=True)
                            _ww = (w * y1_mask).sum(dim=-1, keepdim=True)
                            w = w * torch.where(_wm > 0, _wm / _ww.clamp(min=1e-6), torch.ones_like(_wm))

                            _wce_per = ((-y1_log_prob * y1_mask * w).sum(dim=-1)
                                        / y1_mask.sum(dim=-1).clamp(min=1.0))
                            L_int = _wce_per.sum() * (_cat_world / _cat_n_global)
                            _dlogp_log = (dlogp * y1_mask).sum().item() / max(y1_mask.sum().item(), 1.0)

                            _bdn = getattr(self, "_bar_dbg_n", 0)
                            if _bdn < 1 and float(data["has_y1"].sum().item()) > 0:
                                self._bar_dbg_n = _bdn + 1
                                _msum = y1_mask.sum().clamp(min=1.0)
                                _w_mean = (w * y1_mask).sum().item() / _msum.item()
                                self._cat_log(
                                    f"[BARRIER-DBG #{_bdn}] barrier分支(v0) has_y1={int(data['has_y1'].sum().item())} "
                                    f"coef={catalyst_coef}\n"
                                    f"  引导: y1_真实tok={int(y1_attention_mask.sum().item())} "
                                    f"y1aug_真实tok={int(y1aug_att.sum().item())} (aug应>y1=老师引导已注入) "
                                    f"内化mask_tok={int(y1_mask.sum().item())}(应>0)\n"
                                    f"  梯度: y1_logp.grad={y1_log_prob.requires_grad}(应True=学生侧建图) "
                                    f"y1aug_logp.grad={y1aug_log_prob.requires_grad}(应False=Δlogp只当detach权重) "
                                    f"L_int.grad={L_int.requires_grad}(应True=loss连在图上)\n"
                                    f"  barrier: Δlogp均值(masked)={_dlogp_log:.4f}(应≥0;>0=引导抬高y1的logp/存在barrier;≈0→权重退化为均匀SFT) "
                                    f"clip={_clip} beta={_beta}\n"
                                    f"  权重w: mean(masked)={_w_mean:.4f}(归一化后应≈1.0) max={w.max().item():.4f}(应≥1)\n"
                                    f"  数值: L_int={L_int.item():.4f} L_int_finite={bool(torch.isfinite(L_int).item())}(应True) "
                                    f"dlogp_NaN={bool(torch.isnan(dlogp).any().item())}(应False)")
                        else:

                            if _cmode == "barrier":
                                _bwn = getattr(self, "_bar_warn_n", 0)
                                if _bwn < 1:
                                    self._bar_warn_n = _bwn + 1
                                    self._cat_log(f"[BARRIER-WARN] catalyst_mode=barrier 但落到 else! "
                                                  f"'y1aug_input_ids' in data = {'y1aug_input_ids' in data} → 退化为均匀SFT(无 barrier 加权),需检查 trainer 是否建了 y1aug")

                            _wce_per = ((-y1_log_prob * y1_mask).sum(dim=-1)
                                        / y1_mask.sum(dim=-1).clamp(min=1.0))
                            L_int = _wce_per.sum() * (_cat_world / _cat_n_global)

                        _per = (-y1_log_prob.detach() * y1_mask).sum(dim=-1) / y1_mask.sum(dim=-1).clamp(min=1.0)
                        _hy = data["has_y1"] > 0
                        _n_hy = int(_hy.sum().item())
                        if _hy.any():
                            _l_int_samples.extend(_per[_hy].detach().float().cpu().tolist())
                            _li_ce_sum += float(_per[_hy].sum().item())
                        _li_n += _n_hy
                        _li_rows += int(data["has_y1"].shape[0])
                        _dlogp_wsum += float(_dlogp_log) * _n_hy
                        _li_wce_sum += float(_wce_per.detach().sum().item())

                        _cat_term = catalyst_coef * L_int

                    if self.config.use_dynamic_bsz:

                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size) + _cat_term
                    else:
                        loss = policy_loss / self.gradient_accumulation + _cat_term
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)

        if use_catalyst and catalyst_coef > 0:
            import torch.distributed as _dist
            _vec = torch.tensor([_li_ce_sum, float(_li_n), float(_li_rows), _dlogp_wsum, _li_wce_sum],
                                dtype=torch.float64, device=torch.cuda.current_device())
            if _dist.is_available() and _dist.is_initialized():
                _dist.all_reduce(_vec, op=_dist.ReduceOp.SUM)
            _ce, _n, _rows, _dl, _wce = _vec.tolist()
            metrics["actor/L_int"] = [(_ce / _n) if _n > 0 else 0.0]
            metrics["actor/has_y1_frac"] = [(_n / _rows) if _rows > 0 else 0.0]
            metrics["actor/catalyst_dlogp"] = [(_dl / _n) if _n > 0 else 0.0]

            metrics["actor/L_int_applied"] = [(catalyst_coef * _wce / _n) if _n > 0 else 0.0]

            metrics["actor/y1_n_consistency"] = [float(_cat_n_sum - _n)]
        if _l_int_samples:
            metrics["actor/L_int_max"] = [max(_l_int_samples)]
            metrics["actor/L_int_min"] = [min(_l_int_samples)]
        self.actor_optimizer.zero_grad()
        return metrics
