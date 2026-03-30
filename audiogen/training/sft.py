from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from audiogen.data.dataset import AudioDataCollator, AudioGenerationDataset
from audiogen.data.io import load_records, split_records, save_records_jsonl
from audiogen.model.setup import expand_for_audio, load_base_model
from audiogen.tokens.vocab import TokenMetadata

from transformers import Trainer, TrainingArguments


@dataclass
class SFTConfig:
    # Paths
    data_path: str
    output_dir: str
    base_model: str = "Qwen/Qwen2.5-0.5B"

    # Codec
    codec: str = "moss"
    num_codebooks: int = 8
    codebook_size: int = 1024

    # Data
    val_split: float = 0.05
    max_length: int = 2048

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3


def run_sft(config: SFTConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_base_model(config.base_model)
    model, tokenizer, meta = expand_for_audio(
        model, tokenizer,
        codec=config.codec,
        num_codebooks=config.num_codebooks,
        codebook_size=config.codebook_size,
        save_dir=output_dir,
    )

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
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to="none",
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
