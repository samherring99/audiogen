"""Small explicit native FSDP1 GRPO building blocks.

The caller owns rollout/reference/reward models on OTHER devices/processes,
computes group advantages globally, and distributes equal counts of CPU records.
No generation, model deepcopy, reward loading, or optimizer resume occurs here.
"""
from dataclasses import dataclass
from functools import partial
import json
import math
import os
from pathlib import Path

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,
    FullStateDictConfig, StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from audiogen.training.grpo_objective import completion_loss

import torch

from audiogen.tokens.formatting import build_audio_prefix, prompt_policy_from_config


@dataclass(frozen=True)
class NativeCompletion:
    input_ids: torch.Tensor
    prefix_length: int
    old_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    advantage: float


def prepare_completion(tokenizer, metadata, config, prompt, completion_ids,
                       old_log_probs, ref_log_probs, advantage):
    """Use the exact shared inference prefix; retain invalid nonempty streams.

    Invalid audio receives a penalty from the coordinator, not dropped ranks.
    EOS, when generated, is part of the scored span; never fabricate an EOS.
    """
    ids = torch.as_tensor(completion_ids).detach().cpu()
    if ids.ndim != 1 or not ids.numel():
        raise ValueError('empty or non-vector completion')
    if ids.dtype != torch.long:
        raise ValueError('completion IDs must be int64')
    policy, kwargs = prompt_policy_from_config(config)
    prefix = build_audio_prefix(tokenizer, prompt, metadata, policy, kwargs)
    return NativeCompletion(
        torch.cat((torch.tensor(prefix, dtype=torch.long), ids)), len(prefix),
        old_log_probs.detach().cpu().float().clone(),
        ref_log_probs.detach().cpu().float().clone(), float(advantage),
    )


def completion_token_logprobs(model, input_ids, prefix_length):
    """One unpadded sequence, full causal logits, including the final token/EOS."""
    if (input_ids.ndim != 1 or input_ids.dtype != torch.long or
            not 0 < prefix_length < input_ids.numel()):
        raise ValueError('Expected one nonempty prompt and nonempty completion')
    ids = input_ids.unsqueeze(0)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False, logits_to_keep=0).logits
    if logits.shape[:2] != ids.shape:
        raise ValueError('Model must return full sequence logits')
    selected = logits[0, prefix_length - 1:-1].float()
    labels = input_ids[prefix_length:]
    return selected.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)


@torch.no_grad()
def copy_cpu_state_to_model(model, state, *, chunk_bytes=8 * 1024 * 1024):
    """Copy native-name CPU state in bounded chunks, preserving target dtypes.

    Suitable for an independently loaded BF16 rollout; native FP32 exceptions
    (e.g. routing buffers) stay FP32. Never move/cast the full state to CUDA.
    """
    target = model.state_dict(keep_vars=True)
    if target.keys() != state.keys():
        raise ValueError('state keys differ')
    if chunk_bytes <= 0:
        raise ValueError('chunk_bytes must be positive')
    for name, dest in target.items():
        source = state[name]
        if (source.device.type != 'cpu' or source.shape != dest.shape or
                not source.is_contiguous() or not dest.is_contiguous()):
            raise ValueError(f'Expected matching contiguous CPU source: {name}')
    for name, dest in target.items():
        source = state[name]
        count = max(1, chunk_bytes // max(source.element_size(), dest.element_size()))
        src, dst = source.view(-1), dest.view(-1)
        for offset in range(0, src.numel(), count):
            dst[offset:offset + count].copy_(src[offset:offset + count])


def _collective_require(ok, device, message):
    flag = torch.tensor(int(bool(ok)), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise RuntimeError(f'collective GRPO abort: {message}')


def wrap_native_policy(cpu_fp32_model, device, *, gradient_checkpointing=True):
    """Wrap CPU FP32 native blocks BEFORE moving/sharding each unit to CUDA.

    Caller initializes the process group and loads with device='cpu',
    dtype=torch.float32. Never call model.to(cuda) first. CPU full replicas
    are required per rank; GPU peak is shards plus the largest FSDP unit,
    not a full FP32 policy. Root embeddings/head form one additional unit.
    """
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock, NemotronHForCausalLM
    from audiogen.model.compat import prepare_nemotron_gradient_checkpointing
    device = torch.device(device)
    if not dist.is_initialized() or device.type != 'cuda':
        raise ValueError('Initialized CUDA process group required')
    valid = (type(cpu_fp32_model) is NemotronHForCausalLM and
             all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in cpu_fp32_model.parameters()) and
             all(b.device.type == 'cpu' and (not b.is_floating_point() or b.dtype == torch.float32)
                 for b in cpu_fp32_model.buffers()))
    _collective_require(valid, device, 'policy must be native CPU FP32 before wrapping')
    config = cpu_fp32_model.config
    _collective_require(not any(float(getattr(config, name, 0)) != 0
                                for name in ('attention_dropout', 'hidden_dropout')),
                        device, 'dropout must be zero for on-policy deterministic likelihoods')
    cpu_fp32_model.requires_grad_(True)
    config.use_cache = False
    if gradient_checkpointing:
        prepare_nemotron_gradient_checkpointing(cpu_fp32_model)
        cpu_fp32_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    cpu_fp32_model.train()
    return FSDP(
        cpu_fp32_model, device_id=device,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={NemotronHBlock}),
        sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.float32),
        sync_module_states=True, limit_all_gathers=True, forward_prefetch=False,
        backward_prefetch=None,
    )


def _record_valid(row, vocab_size):
    # Validation must never let rank-local malformed evidence bypass the next
    # collective. Check types BEFORE tensor comparisons, slicing, or conversion.
    try:
        if not isinstance(row, NativeCompletion) or type(row.prefix_length) is not int:
            return False
        ids = row.input_ids
        if (not isinstance(ids, torch.Tensor) or ids.layout != torch.strided or
                ids.device.type != 'cpu' or ids.ndim != 1 or ids.dtype != torch.long or
                not 0 < row.prefix_length < ids.numel() or
                not bool(((ids >= 0) & (ids < vocab_size)).all())):
            return False
        count = ids.numel() - row.prefix_length
        advantage = torch.as_tensor(row.advantage)
        if (advantage.device.type != 'cpu' or advantage.numel() != 1 or
                advantage.is_complex() or not math.isfinite(float(advantage))):
            return False
        return all(isinstance(x, torch.Tensor) and x.layout == torch.strided and
                   x.is_floating_point() and x.device.type == 'cpu' and
                   x.shape == (count,) and not x.requires_grad and
                   bool(torch.isfinite(x).all()) and bool((x <= 0).all())
                   for x in (row.old_log_probs, row.ref_log_probs))
    except (TypeError, ValueError, AttributeError, RuntimeError, OverflowError):
        return False


@torch.no_grad()
def native_gradient_coverage(policy):
    """Local, pre-clip shard coverage, NOT full-model or all-experts assertions.

    Empty original-parameter views are counted as parameters but own zero
    elements on this rank. Routed experts may legitimately have zero gradients.
    Every category is reported, even when absent from a rank's shard.
    """
    categories = ('embedding', 'head', 'norm', 'mamba', 'attention', 'mlp',
                  'router', 'routed_experts', 'shared_experts', 'moe_projection', 'other')
    result = {key: dict(parameter_count=0, parameter_numel=0, grad_parameter_count=0,
                        grad_numel=0, nonzero_grad_parameter_count=0,
                        nonzero_grad_numel=0, grad_norm=0.) for key in categories}
    config = policy.module.config if isinstance(policy, FSDP) else policy.config
    for wrapped_name, parameter in policy.named_parameters():
        name = wrapped_name.replace('_fsdp_wrapped_module.', '')
        category = 'other'
        if name.startswith('model.embeddings.'):
            category = 'embedding'
        elif name.startswith('lm_head.'):
            category = 'head'
        elif any(part in ('norm', 'norm_f') for part in name.split('.')):
            category = 'norm'
        elif name.startswith('model.layers.'):
            block = config.layers_block_type[int(name.split('.')[2])]
            if block == 'moe':
                category = ('shared_experts' if '.shared_experts.' in name else
                            'routed_experts' if '.experts.' in name else
                            'router' if '.gate.' in name else 'moe_projection')
            elif block in ('mamba', 'attention', 'mlp'):
                category = block
        group = result[category]
        group['parameter_count'] += 1
        group['parameter_numel'] += parameter.numel()
        grad = parameter.grad
        if grad is not None and grad.numel():
            nonzero = torch.count_nonzero(grad).item()
            group['grad_parameter_count'] += 1
            group['grad_numel'] += grad.numel()
            group['nonzero_grad_parameter_count'] += int(nonzero > 0)
            group['nonzero_grad_numel'] += nonzero
            group['grad_norm'] = math.hypot(group['grad_norm'],
                                           torch.linalg.vector_norm(grad, dtype=torch.float64).item())
    return result


def native_grpo_update(policy, optimizer, local_completions, *, beta=.02, epsilon=.2,
                       max_grad_norm=1.0, old_logprob_tolerance=None):
    """One update, equal nonzero completion counts on EVERY process-group rank.

    Group normalization is the coordinator's responsibility. Each backward
    synchronizes FSDP shards (never no_sync), dividing only by LOCAL completion
    count; FSDP supplies rank averaging. Invalid audio remains a penalized record.
    Any numerical failure aborts the run; after a failed backward/step do NOT
    reuse this policy. Post-step failure cannot roll back and MUST NOT be saved.

    old_logprob_tolerance=None preserves the general PPO helper contract. For a
    fresh same-policy rollout pass an explicit tolerance (e.g. .05, a hypothesis
    to validate for the target model). A no-grad preflight checks ALL records
    on ALL ranks before any backward, without retaining multiple forward graphs.
    Gradient coverage and CUDA peaks are local per rank; likelihood extrema and
    gradient norm are global. CUDA peaks are since the caller's last peak reset.
    """
    if not isinstance(policy, FSDP):
        raise ValueError('Expected a native FSDP policy')
    device = next(policy.parameters()).device
    rows = list(local_completions)
    counts = torch.tensor([len(rows), -len(rows)], device=device, dtype=torch.int64)
    dist.all_reduce(counts, op=dist.ReduceOp.MIN)
    _collective_require(counts[0].item() > 0 and counts[0].item() == -counts[1].item(),
                        device, 'equal nonempty completion counts required')
    try:
        valid = (math.isfinite(beta) and beta >= 0 and math.isfinite(epsilon) and 0 < epsilon < 1 and
                 math.isfinite(max_grad_norm) and max_grad_norm > 0 and
                 (old_logprob_tolerance is None or
                  (type(old_logprob_tolerance) in (int, float) and
                   math.isfinite(old_logprob_tolerance) and old_logprob_tolerance >= 0)) and
                 all(_record_valid(row, policy.module.config.vocab_size) for row in rows))
    except (TypeError, ValueError, AttributeError, RuntimeError, OverflowError):
        valid = False
    _collective_require(valid, device, 'invalid completion evidence or objective settings')
    tolerance = -1. if old_logprob_tolerance is None else float(old_logprob_tolerance)
    bounds = torch.tensor([tolerance, -tolerance], device=device, dtype=torch.float64)
    dist.all_reduce(bounds, op=dist.ReduceOp.MIN)
    _collective_require(bounds[0].item() == -bounds[1].item(), device,
                        'old-policy likelihood tolerance differs across ranks')
    _collective_require(not getattr(policy, '_grpo_update_incomplete', False), device,
                        'previous failed update; policy cannot be reused')
    policy._grpo_update_incomplete = True
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    # [max |log ratio|, max log ratio, -min log ratio], reduced by MAX.
    likelihood = torch.tensor([0., -float('inf'), -float('inf')], device=device)
    if old_logprob_tolerance is not None:
        with torch.no_grad():
            for row in rows:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    current = completion_token_logprobs(policy, row.input_ids.to(device), row.prefix_length)
                _collective_require(torch.isfinite(current).all().item(), device,
                                    'nonfinite forward log probabilities')
                delta = current - row.old_log_probs.to(current)
                likelihood = torch.maximum(likelihood, torch.stack((delta.abs().max(), delta.max(), -delta.min())))
                del current, delta
        dist.all_reduce(likelihood, op=dist.ReduceOp.MAX)
        _collective_require(likelihood[0].item() <= old_logprob_tolerance, device,
                            f'old-policy likelihood mismatch: max_abs_diff={likelihood[0].item():.8g}, '
                            f'tolerance={old_logprob_tolerance}')
    for row in rows:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            current = completion_token_logprobs(policy, row.input_ids.to(device), row.prefix_length)
        _collective_require(torch.isfinite(current).all().item(), device, 'nonfinite forward log probabilities')
        delta = current.detach() - row.old_log_probs.to(current)
        extrema = torch.stack((delta.abs().max(), delta.max(), -delta.min()))
        if old_logprob_tolerance is not None:
            dist.all_reduce(extrema, op=dist.ReduceOp.MAX)
            _collective_require(extrema[0].item() <= old_logprob_tolerance, device,
                                'old-policy likelihood mismatch on training forward')
        likelihood = torch.maximum(likelihood, extrema)
        del delta, extrema
        # Catch objective overflow on any rank BEFORE backward collectives.
        try:
            loss, kl = completion_loss(current, row.old_log_probs, row.ref_log_probs,
                                       row.advantage, beta=beta, epsilon=epsilon)
            valid = True
        except ValueError:
            valid = False
        _collective_require(valid, device, 'nonfinite objective')
        (loss / len(rows)).backward()
        totals += torch.stack((loss.detach(), kl.detach())).double() / len(rows)
        del current, loss, kl
    _collective_require(all(p.grad is None or torch.isfinite(p.grad).all().item()
                            for p in policy.parameters()), device, 'nonfinite gradient')
    coverage = native_gradient_coverage(policy)
    norm = policy.clip_grad_norm_(max_grad_norm)
    _collective_require(torch.isfinite(norm).item() and norm.item() > 0, device,
                        'finite positive global gradient norm required')
    _collective_require(all(p.grad is None or torch.isfinite(p.grad).all().item()
                            for p in policy.parameters()), device, 'nonfinite post-clip gradient')
    optimizer.step()
    _collective_require(all(torch.isfinite(p).all().item() for p in policy.parameters()) and
                        all(torch.isfinite(v).all().item() for state in optimizer.state.values()
                            for v in state.values() if isinstance(v, torch.Tensor)),
                        device, 'nonfinite post-step parameters or optimizer state')
    dist.all_reduce(totals)
    totals /= dist.get_world_size()
    dist.all_reduce(likelihood, op=dist.ReduceOp.MAX)
    policy._grpo_update_incomplete = False
    return {'loss': totals[0].item(), 'kl': totals[1].item(), 'grad_norm': norm.item(),
            'completions': len(rows) * dist.get_world_size(),
            'old_logprob_max_abs_diff': likelihood[0].item(),
            'importance_ratio_min': math.exp(-likelihood[2].item()),
            'importance_ratio_max': math.exp(likelihood[1].item()),
            'gradient_coverage': coverage,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}


def gather_cpu_state(policy):
    """COLLECTIVE: rank0 gets full native CPU state; others get {}.

    CPU RAM must accommodate a full FP32 checkpoint. FSDP gathers one unit at
    a time and offloads it, never a full FP32 policy on the training GPU.
    Caller releases this state after saving or synchronizing independent rollout.
    """
    with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = policy.state_dict()
    # PyTorch 2.11's buffer mixed-precision post-hook can cast offloaded buffers
    # back to compute_device. Normalize these few entries, never the whole GPU model.
    return {name: value.cpu() for name, value in state.items()}


def save_portable_checkpoint(policy, tokenizer, metadata, path, *, step, run_policy):
    """COLLECTIVE, weights-only native checkpoint. Resume explicitly unsupported.

    Exclusive destination mkdir prevents overwrites. A failure may leave an
    incomplete directory for inspection, but COMPLETE.json is atomically
    published last and is the only success marker. No optimizer/RNG resume.
    """
    from audiogen.tokens.vocab import save_token_metadata
    device = next(policy.parameters()).device
    path = Path(path)
    _collective_require(not getattr(policy, '_grpo_update_incomplete', False), device,
                        'cannot save a failed or incomplete update')
    ok = True
    if dist.get_rank() == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except OSError:
            ok = False
    _collective_require(ok, device, 'checkpoint destination exists or cannot be created')
    state = gather_cpu_state(policy)
    if dist.get_rank() == 0:
        try:
            policy.module.save_pretrained(path, state_dict=state, max_shard_size='2GB')
            tokenizer.save_pretrained(path)
            save_token_metadata(metadata, path / 'audio_token_metadata.json')
            receipt = {'step': step, 'resume_supported': False, 'checkpoint_kind': 'weights_only',
                       'world_size': dist.get_world_size(), 'run_policy': run_policy}
            (path / 'grpo_run_policy.json').write_text(json.dumps(receipt, indent=2) + '\n')
            temporary = path / '.COMPLETE.json.tmp'
            with temporary.open('x') as handle:
                json.dump(receipt, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.rename(path / 'COMPLETE.json')
        except Exception:
            ok = False
    del state
    _collective_require(ok, device, 'portable checkpoint save failed; no successful receipt')
    return path
