from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import warnings

from audiogen.data.dataset import AudioDataCollator, AudioGenerationDataset
from audiogen.data.io import load_records, split_records, save_records_jsonl
from audiogen.model.compat import prepare_nemotron_gradient_checkpointing
from audiogen.model.setup import (
    configure_trainable_parameters, expand_for_audio, load_base_model, load_trained_model,
    _checkpoint_signature, _native_resume_model, _native_state_fingerprint,
)
from audiogen.tokens.formatting import prompt_policy_from_config
from audiogen.tokens.vocab import save_token_metadata

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_callback import ExportableState


@dataclass
class SFTConfig:
    # Keep the original positional data_path/output_dir order for old callers.
    data_path: str | None = None
    output_dir: str = ""
    base_model: str = "Qwen/Qwen2.5-0.5B"
    train_path: str | None = None
    val_path: str | None = None
    revision: str | None = None
    dtype: str = "bfloat16"
    # None selects speech_v1 for a base model, or the saved checkpoint policy.
    prompt_format: str | None = None
    codec: str = "moss"
    num_codebooks: int = 8
    codebook_size: int = 1024
    val_split: float = 0.05
    max_length: int = 2048
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-6
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 0.1
    lr_scheduler_type: str = "constant_with_warmup"
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False
    seed: int = 42
    # Historical option intentionally initializes WEIGHTS ONLY, not Trainer state.
    resume_from_checkpoint: str | None = None
    initialize_from_checkpoint: str | None = None
    trainer_resume_from_checkpoint: str | None = None
    unfreeze_all: bool = False
    unfreeze_last_n_layers: int = 0
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 100
    save_total_limit: int = 3
    report_to: str | list[str] = "wandb"
    run_name: str | None = None
    plateau_patience: int = 0
    plateau_min_steps: int = 0
    plateau_warmup_steps: int = 0
    plateau_min_delta: float = 0.0
    # Operational pause, not the scheduler budget; may change on true resume.
    stop_after_step: int | None = None
    # Opt-in, torchrun only. FSDP1 full-shard/full-unfreeze is the supported path.
    fsdp: str | None = None
    fsdp_config: dict | str | None = None


def _distributed():
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def _fail_if_any_rank(bad, message):
    # Never raise locally before the collective: peers may otherwise hang in
    # backward, gradient clipping or a save. One scalar, including empty shards.
    flag = bad.detach().to(dtype=torch.int32)
    if _distributed():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    if flag.item():
        raise FloatingPointError(message)


def _check_gradients(model):
    bad = torch.zeros((), dtype=torch.bool, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.grad is not None:
            bad.logical_or_(~torch.isfinite(param.grad).all())
    _fail_if_any_rank(bad, "Non-finite gradient on at least one rank; optimizer update aborted")


def _check_gradient_norm(model):
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not _distributed():
        norm = torch.nn.utils.get_total_norm(gradients)
    else:
        # FSDP1 FULL_SHARD uses FP32 local norms and a SUM of their squares.
        # During no_sync full gradients may be replicated: this conservative
        # overflow check is not a clipping norm, and never modifies gradients.
        squared = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
        for grad in gradients:
            squared.add_(torch.linalg.vector_norm(grad, dtype=torch.float32).square())
        torch.distributed.all_reduce(squared, op=torch.distributed.ReduceOp.SUM)
        norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite aggregate gradient norm; optimizer update aborted")


class GradientHealthCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        # Check again after gradient clipping/unscaling.
        _check_gradients(model)


class RobustTrainer(Trainer):
    """Fail visibly before backward for bad loss, before updates for bad gradients."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_fsdp_enabled:
            plugin = self.accelerator.state.fsdp_plugin
            if plugin.fsdp_version != 1:
                raise ValueError("Only FSDP version 1 is supported")
            # Validate the effective plugin before prepare wraps the model:
            # inherited FSDP_* environment values can override the requested config.
            from torch.distributed.fsdp import ShardingStrategy
            if plugin.sharding_strategy != ShardingStrategy.FULL_SHARD:
                raise ValueError("FSDP requires effective FULL_SHARD sharding_strategy")
            if plugin.cpu_offload.offload_params is not False:
                raise ValueError("FSDP requires effective cpu_offload.offload_params=False")
            # Accelerate defaults buffer_dtype to the compute dtype. Native
            # Nemotron routing correction biases must stay FP32: casting these
            # nontrainable buffers to BF16 irreversibly changes expert selection.
            # Trainer has initialized the plugin, but prepare has not wrapped the
            # model yet. Preserve every other mixed-precision field and flag.
            if plugin.mixed_precision_policy is not None:
                plugin.mixed_precision_policy = replace(
                    plugin.mixed_precision_policy, buffer_dtype=torch.float32)
            # TrainingArguments 5.6.2 does not forward state_dict_type from its
            # JSON config. Set it through the actual Accelerate plugin API.
            plugin.state_dict_config = plugin.optim_state_dict_config = None
            plugin.set_state_dict_type("FULL_STATE_DICT")
        self.add_callback(GradientHealthCallback())

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = self.model if model is None else model
        if not _native_resume_model(target):
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        # HF 5.6.2 Trainer uses raw load_state_dict, which cannot reverse
        # save_original_format's backbone names / split MoE experts. Reuse only
        # our clean from_pretrained load of these exact local weights. Trainer
        # still restores optimizer, scheduler, RNG, callbacks and global step.
        receipt = getattr(target, "_audiogen_resume_preload", None)
        info = getattr(target, "audiogen_loading_info", {})
        expected = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
        valid = (
            target is self.model and receipt is not None
            and receipt[0] == id(target)
            and receipt[1] == str(Path(resume_from_checkpoint).resolve())
            and expected <= info.keys() and not any(info.values())
            and receipt[2] == _checkpoint_signature(resume_from_checkpoint)
            and self.state.global_step == 0
            and not self.is_fsdp_enabled and not self.is_deepspeed_enabled
            # Full exact-byte check LAST: cheap guards reject without a state scan.
            and receipt[3] == _native_state_fingerprint(target)
        )
        if not valid:
            raise ValueError("Native Nemotron Trainer resume requires an unchanged, verified conversion preload of the exact checkpoint; reload with load_trained_model")
        del target._audiogen_resume_preload

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        loss = result[0] if return_outputs else result
        _fail_if_any_rank(~torch.isfinite(loss).all(),
                          f"Non-finite loss at global_step={self.state.global_step}; backward aborted")
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        # Includes accumulated gradients, and runs before Trainer clips them.
        _check_gradients(model)
        # Finite entries can still overflow the norm used by clipping, which
        # would silently zero the gradients. Measure without modifying them.
        _check_gradient_norm(model)
        return loss


class SafetyPauseCallback(TrainerCallback):
    """Save full Trainer state at an update boundary without shortening its schedule."""
    def __init__(self, stop_after_step):
        self.stop_after_step = stop_after_step

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_step:
            raise ValueError("global_step has reached stop_after_step; explicitly raise/remove the operational gate")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_step:
            control.should_log = control.should_evaluate = True
            control.should_save = control.should_training_stop = True
            print(f"Safety pause at update {state.global_step}: log/evaluate/save/stop; full schedule remains unchanged.")
        return control


class PlateauCallback(TrainerCallback, ExportableState):
    """Optional eval-loss plateau stop; not a perceptual audio quality verdict."""
    def __init__(self, patience=0, min_steps=0, warmup_steps=0, min_delta=0.0):
        self.patience = patience
        self.min_steps = min_steps
        self.warmup_steps = warmup_steps
        self.min_delta = min_delta
        self.best = None
        self.bad_evaluations = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.patience or state.global_step < max(self.min_steps, self.warmup_steps):
            return control
        loss = (metrics or {}).get("eval_loss")
        if loss is None:
            return control
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite eval loss")
        if self.best is None or loss < self.best - self.min_delta:
            self.best, self.bad_evaluations = loss, 0
        else:
            self.bad_evaluations += 1
        if self.bad_evaluations >= self.patience:
            print(f"Eval-loss plateau at step {state.global_step}: saving and stopping cleanly (audio quality requires separate evaluation).")
            control.should_save = control.should_training_stop = True
        return control

    def state(self):
        return {
            "args": {key: getattr(self, key) for key in ("patience", "min_steps", "warmup_steps", "min_delta")},
            "attributes": {"best": self.best, "bad_evaluations": self.bad_evaluations},
        }


class PortableCheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer, metadata, policy):
        self.tokenizer, self.metadata, self.policy = tokenizer, metadata, policy

    def save(self, path, model):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        save_token_metadata(self.metadata, path / "audio_token_metadata.json")
        model.config.save_pretrained(path)
        (path / "sft_run_policy.json").write_text(json.dumps(self.policy, indent=2) + "\n")

    def on_save(self, args, state, control, model=None, **kwargs):
        if args.should_save:
            self.save(Path(args.output_dir) / f"checkpoint-{state.global_step}", model)


def configure_sft_model(model, config):
    n = config.unfreeze_last_n_layers
    if n < 0 or (n and config.unfreeze_all):
        raise ValueError("Choose unfreeze_all or a nonnegative unfreeze_last_n_layers")
    configure_trainable_parameters(model, unfreeze_all=config.unfreeze_all)
    if n:
        if (model.config.model_type not in ("qwen2", "qwen3") or
                not hasattr(model, "model") or not hasattr(model.model, "layers") or
                n > len(model.model.layers)):
            raise ValueError("unfreeze_last_n_layers unsupported: requires Qwen2/Qwen3 and N <= layer count")
        for module in [*model.model.layers[-n:], model.model.norm]:
            for param in module.parameters():
                param.requires_grad = True
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("Freeze policy selected no trainable parameters")
    model.config.use_cache = False
    if config.gradient_checkpointing:
        prepare_nemotron_gradient_checkpointing(model)


def _fsdp_options(config):
    supplied = config.fsdp_config or {}
    if isinstance(supplied, str):
        supplied = json.loads(Path(supplied).read_text())
    fixed = {"version": 1, "use_orig_params": True, "sync_module_states": True,
             "cpu_ram_efficient_loading": False, "activation_checkpointing": False,
             "state_dict_type": "FULL_STATE_DICT", "limit_all_gathers": True,
             "transformer_layer_cls_to_wrap": ["NemotronHBlock"],
             "backward_prefetch": "BACKWARD_PRE", "forward_prefetch": False}
    # Narrow implementation: reject unsupported keys rather than silently let HF
    # ignore them, or let launcher environment variables change sharding policy.
    for key, value in supplied.items():
        if key not in fixed or value != fixed[key]:
            raise ValueError(f"Unsupported FSDP setting {key}={value!r}; expected {fixed.get(key)!r}")
    return fixed


def _validate_config(config):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1 or _distributed()
    if distributed and not config.fsdp:
        raise ValueError("Multi-rank distributed SFT requires explicit FSDP")
    if config.fsdp:
        if config.fsdp != "full_shard auto_wrap":
            raise ValueError("Supported FSDP mode is full_shard auto_wrap")
        if not distributed or int(os.environ.get("LOCAL_RANK", "-1")) < 0:
            raise ValueError("FSDP requires torchrun with at least two ranks and LOCAL_RANK")
        if not config.unfreeze_all or config.unfreeze_last_n_layers:
            raise ValueError("FSDP requires unfreeze_all, not partial freezing")
        if not config.train_path or not config.val_path or config.data_path:
            raise ValueError("Distributed SFT requires explicit existing train_path and val_path; no splitting")
        _fsdp_options(config)
    elif config.fsdp_config:
        raise ValueError("fsdp_config requires fsdp")
    if not config.output_dir:
        raise ValueError("output_dir is required")
    if bool(config.train_path) != bool(config.val_path):
        raise ValueError("train_path and val_path must be supplied together")
    if config.train_path and config.data_path:
        raise ValueError("Use train_path/val_path OR legacy data_path, not both")
    if not config.train_path and not config.data_path:
        raise ValueError("Provide train_path and val_path (or legacy data_path)")
    if sum(bool(x) for x in (config.resume_from_checkpoint, config.initialize_from_checkpoint, config.trainer_resume_from_checkpoint)) > 1:
        raise ValueError("Choose exactly one weights initialization or Trainer resume option")
    if config.dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("dtype must be auto, float32, float16 or bfloat16")
    if config.max_steps == 0 or config.max_steps < -1:
        raise ValueError("max_steps must be -1 (epochs) or positive")
    if config.stop_after_step is not None and (type(config.stop_after_step) is not int or config.stop_after_step <= 0):
        raise ValueError("stop_after_step must be a positive integer or None")
    if min(config.plateau_patience, config.plateau_min_steps, config.plateau_warmup_steps, config.plateau_min_delta) < 0:
        raise ValueError("Plateau controls must be nonnegative")
    for path in (config.train_path, config.val_path, config.data_path):
        if path and not Path(path).is_file():
            raise ValueError(f"Dataset file does not exist: {path}")


# These changes would invalidate optimizer groups, data order, or scheduler continuity.
_RESUME_KEYS = (
    "unfreeze_all", "unfreeze_last_n_layers", "dtype", "bf16", "fp16", "seed",
    "train_path", "val_path", "data_path", "val_split", "max_length",
    "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate",
    "weight_decay", "warmup_steps", "lr_scheduler_type", "max_steps", "num_train_epochs",
    "plateau_patience", "plateau_min_steps", "plateau_warmup_steps", "plateau_min_delta",
)


def _validate_resume(config):
    path = Path(config.trainer_resume_from_checkpoint)
    world_size = torch.distributed.get_world_size() if _distributed() else int(os.environ.get("WORLD_SIZE", "1"))
    files = ["trainer_state.json", "scheduler.pt", "sft_run_policy.json"]
    if config.fsdp:
        files += ["pytorch_model_fsdp.bin", "optimizer.bin"]
        files += [f"rng_state_{rank}.pth" for rank in range(world_size)]
    else:
        files += ["optimizer.pt", "rng_state.pth"]
    for filename in files:
        if not (path / filename).is_file():
            raise ValueError(f"True Trainer resume requires {filename}; use initialize_from_checkpoint for weights only")
    state = json.loads((path / "trainer_state.json").read_text())
    if config.stop_after_step is not None and state["global_step"] >= config.stop_after_step:
        raise ValueError("Trainer resume global_step has reached stop_after_step; explicitly raise/remove the operational gate")
    if config.max_steps > 0 and state["global_step"] >= config.max_steps:
        raise ValueError("Trainer resume global_step has already exhausted max_steps budget")
    policy = json.loads((path / "sft_run_policy.json").read_text())
    saved = policy["config"]
    if config.fsdp and policy.get("world_size") != world_size:
        raise ValueError("Trainer resume world_size mismatch; data/RNG continuity requires the original rank count")
    if saved.get("fsdp") != config.fsdp:
        raise ValueError("Trainer resume policy mismatch: fsdp; use weights initialization for a new phase")
    current = asdict(config)
    mismatches = [key for key in _RESUME_KEYS if saved.get(key) != current[key]]
    if mismatches:
        raise ValueError(f"Trainer resume policy mismatch: {', '.join(mismatches)}; use weights initialization for a new phase")


def run_sft(config: SFTConfig) -> RobustTrainer:
    _validate_config(config)
    if config.fsdp:
        # Bind before any CUDA use, including Accelerate's sync_module_states.
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if config.trainer_resume_from_checkpoint:
        _validate_resume(config)
    # Seed before model expansion, not only after Trainer construction.
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    dtype = "auto" if config.dtype == "auto" else getattr(torch, config.dtype)
    checkpoint = config.trainer_resume_from_checkpoint or config.initialize_from_checkpoint or config.resume_from_checkpoint
    if config.resume_from_checkpoint:
        warnings.warn("resume_from_checkpoint is a legacy weights-only alias; use trainer_resume_from_checkpoint to restore Trainer state", FutureWarning)
    if checkpoint:
        model, tokenizer, meta = load_trained_model(checkpoint, dtype=dtype, revision=config.revision)
        saved_format, _ = prompt_policy_from_config(model.config)
        if config.prompt_format is not None and config.prompt_format != saved_format:
            raise ValueError("prompt_format conflicts with saved checkpoint policy")
    else:
        model, tokenizer = load_base_model(config.base_model, dtype=dtype, revision=config.revision, prompt_format=config.prompt_format or "speech_v1")
        model, tokenizer, meta = expand_for_audio(model, tokenizer, codec=config.codec, num_codebooks=config.num_codebooks, codebook_size=config.codebook_size)
    configure_sft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} / {sum(p.numel() for p in model.parameters()):,}")
    prompt_format, chat_kwargs = prompt_policy_from_config(model.config)
    if tokenizer.pad_token_id is None:
        raise ValueError("Saved tokenizer must have a pad_token_id for batching")
    if int(os.environ.get("RANK", "0")) == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if config.train_path:
        train_path, val_path = Path(config.train_path), Path(config.val_path)
    else:
        records = load_records(config.data_path)
        train_records, val_records = split_records(records, val_split=config.val_split, seed=config.seed)
        train_path, val_path = output_dir / "train.jsonl", output_dir / "val.jsonl"
        if Path(config.data_path).resolve() in (train_path.resolve(), val_path.resolve()):
            raise ValueError("Legacy split output would overwrite source data; use explicit prepared paths")
        save_records_jsonl(train_records, train_path)
        save_records_jsonl(val_records, val_path)
    train_dataset = AudioGenerationDataset(train_path, tokenizer, meta, config.max_length, prompt_format, chat_kwargs)
    val_dataset = AudioGenerationDataset(val_path, tokenizer, meta, config.max_length, prompt_format, chat_kwargs)
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Both training and validation datasets must be nonempty")
    training_args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=config.num_train_epochs, max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps, lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16, bf16=config.bf16, max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        logging_steps=config.logging_steps, logging_nan_inf_filter=False,
        eval_strategy="steps", eval_steps=config.eval_steps, save_steps=config.save_steps,
        save_total_limit=config.save_total_limit, report_to=config.report_to, run_name=config.run_name,
        seed=config.seed, data_seed=config.seed, restore_callback_states_from_checkpoint=True,
        fsdp=config.fsdp or "", fsdp_config=_fsdp_options(config) if config.fsdp else None,
        average_tokens_across_devices=True,
        # FSDP no_sync retains FULL gradients across accumulated microbatches.
        # Sync/reduce-scatter every backward, but still update only at the
        # gradient_accumulation_steps boundary (Trainer/Accelerate supported API).
        accelerator_config={"gradient_accumulation_kwargs": {"sync_each_batch": True}} if config.fsdp else None,
    )
    if not config.fsdp and (training_args.world_size != 1 or training_args.n_gpu > 1):
        raise ValueError("Single-device SFT only: select one GPU with CUDA_VISIBLE_DEVICES")
    policy = {"config": asdict(config), "audiogen": model.config.audiogen,
              "world_size": training_args.world_size,
              "train_path": str(train_path.resolve()), "val_path": str(val_path.resolve()),
              "trainable_parameters": [name for name, p in model.named_parameters() if p.requires_grad]}
    portable = PortableCheckpointCallback(tokenizer, meta, policy)
    callbacks = [portable]
    if config.stop_after_step is not None:
        callbacks.append(SafetyPauseCallback(config.stop_after_step))
    if config.plateau_patience:
        callbacks.append(PlateauCallback(config.plateau_patience, config.plateau_min_steps, config.plateau_warmup_steps, config.plateau_min_delta))
    trainer = RobustTrainer(model=model, args=training_args, train_dataset=train_dataset,
                            eval_dataset=val_dataset, data_collator=AudioDataCollator(tokenizer.pad_token_id),
                            processing_class=tokenizer, callbacks=callbacks)
    trainer.train(resume_from_checkpoint=config.trainer_resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    trainer.save_state()
    if trainer.is_world_process_zero():
        portable.save(output_dir, model)
    trainer.accelerator.wait_for_everyone()
    return trainer
