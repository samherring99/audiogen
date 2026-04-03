from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from audiogen.training.sft import SFTConfig
from audiogen.data.dataset import AudioDataCollator, AudioGenerationDataset
from audiogen.data.io import load_records, split_records, save_records_jsonl
from audiogen.model.setup import expand_for_audio, load_base_model, load_trained_model
from audiogen.tokens.vocab import save_token_metadata

from transformers import Trainer, TrainingArguments


@dataclass
class LoRAConfig(SFTConfig):
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    )
    freeze_embeddings: bool = False
    embedding_lr_scale: float = 0.01  # multiply base LR for embed_tokens/lm_head


def apply_lora(model, config: LoRAConfig):
    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # Unfreeze embeddings only if freeze_embeddings is not set
    if not config.freeze_embeddings:
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True

    model.print_trainable_parameters()
    return model


def run_lora(config: LoRAConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.resume_from_checkpoint is not None:
        print(f"Loading from checkpoint: {config.resume_from_checkpoint}")
        model, tokenizer, meta = load_trained_model(config.resume_from_checkpoint)
        from audiogen.tokens.vocab import save_token_metadata
        tokenizer.save_pretrained(output_dir)
        save_token_metadata(meta, output_dir / "audio_token_metadata.json")
    else:
        model, tokenizer = load_base_model(config.base_model)
        model, tokenizer, meta = expand_for_audio(
            model, tokenizer,
            codec=config.codec,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            save_dir=output_dir,
        )
    model = apply_lora(model, config)

    records = load_records(config.data_path)
    train_records, val_records = split_records(records, val_split=config.val_split)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    save_records_jsonl(train_records, train_path)
    save_records_jsonl(val_records, val_path)

    train_dataset = AudioGenerationDataset(train_path, tokenizer, meta, config.max_length)
    val_dataset = AudioGenerationDataset(val_path, tokenizer, meta, config.max_length)
    collator = AudioDataCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    save_token_metadata(meta, output_dir / "audio_token_metadata.json")
