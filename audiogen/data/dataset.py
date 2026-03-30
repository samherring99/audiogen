from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from audiogen.data.io import load_records, split_records
from audiogen.data.schema import AudioRecord
from audiogen.tokens.formatting import get_formatter
from audiogen.tokens.vocab import TokenMetadata, load_token_metadata


class AudioGenerationDataset(Dataset):
    def __init__(
        self,
        records_path: str | Path,
        tokenizer,
        metadata: TokenMetadata,
        max_length: int = 2048,
    ):
        self.records = load_records(records_path)
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.max_length = max_length
        self.formatter = get_formatter(metadata.codec, metadata.num_codebooks)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]

        # Re-hydrate CodecOutput from flat tokens so the formatter can render the string
        if self.metadata.codec == "snac":
            from audiogen.codecs.snac import SNACOutput
            output = SNACOutput.from_flat_tokens(
                record.tokens, record.num_codebooks, record.num_frames
            )
        else:
            from audiogen.codecs.moss import MOSSOutput
            output = MOSSOutput.from_flat_tokens(
                record.tokens, record.num_codebooks, record.num_frames
            )

        messages = self.formatter.format_chat_messages(record.prompt, output)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        # Mask everything up to and including the <audio> opening tag from the loss.
        # The model only needs to learn to predict the audio tokens.
        audio_start_id = self.metadata.audio_start_token_id
        audio_positions = (input_ids == audio_start_id).nonzero(as_tuple=True)[0]
        if len(audio_positions) > 0:
            labels[: audio_positions[0] + 1] = -100

        # Mask padding
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # Sanity check — if no labels are valid, something is wrong
        if (labels != -100).sum() == 0:
            print(f"Warning: sample {idx} has no valid label tokens!", flush=True)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class AudioDataCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)
        if self.pad_to_multiple_of > 1:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        input_ids = torch.full((len(features), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(features), max_len, dtype=torch.long)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)

        for i, f in enumerate(features):
            seq_len = f["input_ids"].shape[0]
            input_ids[i, :seq_len] = f["input_ids"]
            attention_mask[i, :seq_len] = f["attention_mask"]
            labels[i, :seq_len] = f["labels"]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def create_dataloaders(
    train_path: str | Path,
    val_path: str | Path,
    tokenizer,
    metadata: TokenMetadata,
    batch_size: int = 4,
    max_length: int = 2048,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_ds = AudioGenerationDataset(train_path, tokenizer, metadata, max_length)
    val_ds = AudioGenerationDataset(val_path, tokenizer, metadata, max_length)
    collator = AudioDataCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers
    )
    return train_loader, val_loader
