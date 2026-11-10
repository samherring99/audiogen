"""Token GRPO primitives. No rollout, model loading, or reward-model side effects.

One token mean per completion; caller averages completions (not tokens across
unequal lengths). No padding is accepted: pass exactly the completion span,
including its EOS. Old/reference log probabilities and advantages are detached.
"""
import math

import torch


def group_advantages(rewards: torch.Tensor, group_ids: list[int]) -> torch.Tensor:
    if rewards.ndim != 1 or len(rewards) != len(group_ids) or not len(rewards):
        raise ValueError("Expected nonempty reward vector and matching group IDs")
    if not torch.isfinite(rewards).all():
        raise ValueError("Nonfinite rewards")
    if any(type(g) is not int for g in group_ids):
        raise ValueError("Rollout group IDs must be explicit integers")
    values = rewards.detach().float()
    result = torch.zeros_like(values)
    for group_id in dict.fromkeys(group_ids):
        indices = [i for i, g in enumerate(group_ids) if g == group_id]
        if len(indices) < 2:
            raise ValueError("Each rollout group needs at least two completions")
        group = values[indices]
        std = group.std(unbiased=False)
        result[indices] = (group - group.mean()) / (std + 1e-8)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite advantages")
    return result


def completion_loss(log_probs, old_log_probs, ref_log_probs, advantage, beta=.02, epsilon=.2):
    """Clipped token surrogate plus k3 KL; returns scalar loss and mean KL.

    k3 is nonnegative pointwise, not a claim of unbiased KL under top-p sampling.
    This primitive assumes unpadded completion vectors, not sequence-mean ratios.
    """
    if not math.isfinite(beta) or beta < 0 or not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("Invalid KL weight or clipping epsilon")
    if log_probs.ndim != 1 or not log_probs.numel():
        raise ValueError("Expected a nonempty unpadded completion vector")
    if old_log_probs.shape != log_probs.shape or ref_log_probs.shape != log_probs.shape:
        raise ValueError("Logprob shapes differ")
    current = log_probs.float()
    old = old_log_probs.detach().to(current)
    ref = ref_log_probs.detach().to(current)
    adv = torch.as_tensor(advantage, device=current.device, dtype=current.dtype).detach()
    if adv.numel() != 1 or not all(torch.isfinite(x).all() for x in (current, old, ref, adv)):
        raise ValueError("Nonfinite log probabilities/advantage or nonscalar advantage")
    if any((x > 0).any() for x in (current, old, ref)):
        raise ValueError("Log probabilities cannot be positive")
    ratio = torch.exp(current - old)
    if not torch.isfinite(ratio).all():
        raise ValueError("Nonfinite importance ratio; clipped forward can hide NaN backward")
    delta = ref - current
    kl = torch.expm1(delta) - delta
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1-epsilon, 1+epsilon) * adv)
    loss = (-surrogate + beta * kl).mean()
    if not torch.isfinite(loss) or not torch.isfinite(kl).all():
        raise ValueError("Nonfinite objective; update must abort")
    return loss, kl.mean()
