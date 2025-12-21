import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from audio_dataset import AudioGenerationDataset


# Sample training loop for small development case

@dataclass
class AudioTrainingArguments:

    model_path: str = field(
        default="./models/qwen-audio",
        metadata={"help": "Path to prepared model with expanded vocabulary"}
    )

    train_dataset: str = field(
        default="./data/train_dataset.json",
        metadata={"help": "Path to training dataset"}
    )
    val_dataset: Optional[str] = field(
        default="./data/val_dataset.json",
        metadata={"help": "Path to validation dataset"}
    )

    output_dir: str = field(
        default="./checkpoints",
        metadata={"help": "Directory to save checkpoints"}
    )

    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Training batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Evaluation batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of warmup steps"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for clipping"}
    )

    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every N steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Evaluate every N steps"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )

    # audio scales - default is 3
    num_scales: int = field(
        default=3,
        metadata={"help": "Number of SNAC scales"}
    )

    fp16: bool = field(
        default=True,
        metadata={"help": "Use mixed precision training"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of dataloader workers"}
    )


class AudioDataCollator:

    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        max_length = max(len(ids) for ids in input_ids)

        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1)
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, lbls, mask in zip(input_ids, labels, attention_mask):
            padding_length = max_length - len(ids)

            padded_ids = torch.cat([
                ids,
                torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=ids.dtype)
            ])
            padded_input_ids.append(padded_ids)

            padded_lbls = torch.cat([
                lbls,
                torch.full((padding_length,), -100, dtype=lbls.dtype)
            ])
            padded_labels.append(padded_lbls)

            padded_mask = torch.cat([
                mask,
                torch.zeros((padding_length,), dtype=mask.dtype)
            ])
            padded_attention_mask.append(padded_mask)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels)
        }


def train(args: AudioTrainingArguments):
    print("="*50)
    print("Audio Generation Training")
    print("="*50)

    print(f"\nLoading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    print(f"Vocab size: {len(tokenizer)}")

    print(f"\nLoading training dataset from {args.train_dataset}")
    train_dataset = AudioGenerationDataset(
        dataset_path=args.train_dataset,
        tokenizer_path=args.model_path,
        num_scales=args.num_scales
    )

    eval_dataset = None
    if args.val_dataset:
        print(f"Loading validation dataset from {args.val_dataset}")
        eval_dataset = AudioGenerationDataset(
            dataset_path=args.val_dataset,
            tokenizer_path=args.model_path,
            num_scales=args.num_scales
        )

    data_collator = AudioDataCollator(tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        logging_dir=f"{args.output_dir}/logs",
        report_to="tensorboard",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        save_safetensors=True,
        eval_strategy="steps",
        save_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    trainer.train()

    final_model_path = f"{args.output_dir}/final_model"
    print(f"\nSaving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print("\nTraining complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./models/qwen-audio")
    parser.add_argument("--train_dataset", type=str, default="./data/train_dataset.json")
    parser.add_argument("--val_dataset", type=str, default="./data/val_dataset.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--num_scales", type=int, default=3)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    args = parser.parse_args()
    training_args = AudioTrainingArguments(**vars(args))
    train(training_args)
