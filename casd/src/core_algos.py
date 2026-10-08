import numpy as np
import torch
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

import verl.utils.torch_functional as verl_F

class AdaptiveKLController:

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class FixedKLController:

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass

def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

def extract_and_pad_by_mask(tensor: torch.Tensor, mask: torch.Tensor, padding_value=0.0):

    batch_size = tensor.shape[0]
    device = tensor.device

    extracted_tensors = []
    original_indices = []
    lengths = []

    for i in range(batch_size):

        indices = torch.where(mask[i] > 0)[0]

        if len(indices) == 0:

            extracted = torch.zeros(0, device=device)
        else:

            extracted = tensor[i, indices]

        extracted_tensors.append(extracted)
        original_indices.append(indices)
        lengths.append(len(indices))

    if any(lengths):
        padded_tensor = pad_sequence(extracted_tensors, batch_first=True, padding_value=padding_value)
    else:
        padded_tensor = torch.zeros((batch_size, 0), device=device)

    lengths = torch.tensor(lengths, device=device)

    return padded_tensor, lengths, original_indices

def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):

    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []

        extracted_rewards, lengths, indices = extract_and_pad_by_mask(token_level_rewards, action_mask)
        extracted_values, _, _ = extract_and_pad_by_mask(values, action_mask)

        max_length = max(lengths)

        for t in reversed(range(max_length)):
            nextvalues = extracted_values[:, t + 1] if t < max_length - 1 else 0.0
            delta = extracted_rewards[:, t] + gamma * nextvalues - extracted_values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        extracted_advantages = torch.stack(advantages_reversed[::-1], dim=1)

        advantages = torch.zeros_like(token_level_rewards)
        for i, length in enumerate(lengths):
            advantages[i, indices[i]] = extracted_advantages[i][:length]

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, action_mask)
    return advantages, returns

def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    action_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):

    response_length = token_level_rewards.shape[-1]

    scores = (token_level_rewards * action_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1).tile([1, response_length]) * action_mask

    return scores, scores

def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, action_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):

    response_length = token_level_rewards.shape[-1]

    scores = (token_level_rewards * action_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * action_mask
        scores = verl_F.masked_whiten(scores, action_mask)

    return scores, scores

def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   action_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):

    response_length = token_level_rewards.shape[-1]

    scores = (token_level_rewards * action_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * action_mask

    return scores, scores

def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, action_mask: torch.Tensor,
                                                  gamma: torch.Tensor):

    with torch.no_grad():
        running_return = 0
        extracted_rewards, lengths, indices = extract_and_pad_by_mask(token_level_rewards, action_mask)
        max_length = max(lengths)
        extracted_returns = torch.zeros_like(extracted_rewards)
        for t in reversed(range(max_length)):
            running_return = extracted_rewards[:, t] + gamma * running_return
            extracted_returns[:, t] = running_return

        returns = torch.zeros_like(token_level_rewards)
        for i, length in enumerate(lengths):
            returns[i, indices[i]] = extracted_returns[i][:length]

        advantages = verl_F.masked_whiten(returns, action_mask)
        advantages = advantages * action_mask

    return advantages, returns

def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    action_mask: torch.Tensor):

    response_length = token_level_rewards.shape[-1]

    scores = (token_level_rewards * action_mask).sum(dim=-1)

    with torch.no_grad():

        masked_rewards = token_level_rewards * action_mask
        returns = masked_rewards.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * action_mask

    return advantages, returns

def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):

    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]

    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss

def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    action_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):

    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, action_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), action_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), action_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=action_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

def compute_entropy_loss(logits, action_mask):

    entropy = verl_F.entropy_from_logits(logits)
    entropy_loss = verl_F.masked_mean(entropy, mask=action_mask)
    return entropy_loss

def compute_value_loss(vpreds, returns, values, state_mask, cliprange_value):

    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), state_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), state_mask)
    return vf_loss, vf_clipfrac

def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:

    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":

        raise NotImplementedError

    raise NotImplementedError
