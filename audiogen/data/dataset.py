from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from audiogen.data.io import load_records
from audiogen.data.schema import AudioRecord
from audiogen.tokens.formatting import (
    build_audio_prefix, get_formatter, resolve_prompt_policy, validate_audio_stream,
)
from audiogen.tokens.vocab import TokenMetadata


class AudioGenerationDataset(Dataset):
    def __init__(
        self,
        records_path: str | Path,
        tokenizer,
        metadata: TokenMetadata,
        max_length: int = 2048,
        prompt_format: str = "legacy",
        chat_template_kwargs: dict | None = None,
    ):
        self.prompt_format, self.chat_template_kwargs = resolve_prompt_policy(prompt_format, chat_template_kwargs)
        self.records = load_records(records_path)
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.max_length = max_length
        self.formatter = get_formatter(metadata.codec, metadata.num_codebooks)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        if self.prompt_format == "speech_v1":
            return self._native_item(record, idx)

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
            messages, tokenize=False, add_generation_prompt=False, **self.chat_template_kwargs
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

        # Padding is a position, not a token value (pad may equal EOS).
        labels[attention_mask == 0] = -100

        # Sanity check — if no labels are valid, something is wrong
        if (labels != -100).sum() == 0:
            print(f"Warning: sample {idx} has no valid label tokens!", flush=True)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _native_item(self, record: AudioRecord, idx: int) -> dict[str, torch.Tensor]:
        meta = self.metadata
        context = f"sample {idx}"
        if record.codec != meta.codec or record.num_codebooks != meta.num_codebooks:
            raise ValueError(f"{context}: codec/codebook metadata mismatch")
        if type(record.num_frames) is not int or record.num_frames <= 0:
            raise ValueError(f"{context}: num_frames must be a positive integer")
        expected = record.num_frames * (
            (2 ** meta.num_codebooks - 1) if meta.codec == "snac" else meta.num_codebooks
        )
        if len(record.tokens) != expected:
            raise ValueError(f"{context}: expected {expected} codec targets, got {len(record.tokens)}")
        if any(type(t) is not int or not 0 <= t < meta.codebook_size for t in record.tokens):
            raise ValueError(f"{context}: invalid codec target (integer/range required)")
        if meta.codec == "snac":
            from audiogen.codecs.snac import SNACOutput
            output = SNACOutput.from_flat_tokens(record.tokens, record.num_codebooks, record.num_frames)
        else:
            from audiogen.codecs.moss import MOSSOutput
            output = MOSSOutput.from_flat_tokens(record.tokens, record.num_codebooks, record.num_frames)
        prefix = build_audio_prefix(
            self.tokenizer, record.prompt, meta, self.prompt_format, self.chat_template_kwargs
        )
        messages = self.formatter.format_chat_messages(record.prompt, output, self.prompt_format)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **self.chat_template_kwargs
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if ids.count(meta.audio_start_token_id) != 1 or ids.count(meta.audio_end_token_id) != 1:
            raise ValueError(f"{context}: expected exactly one opening/closing audio boundary")
        if ids[:len(prefix)] != prefix:
            raise ValueError(f"{context}: native training prefix differs from inference through <audio>")
        end = ids.index(meta.audio_end_token_id)
        if end < len(prefix):
            raise ValueError(f"{context}: no audio targets after <audio> boundary")
        stream = validate_audio_stream(ids[len(prefix):end + 1], meta)
        if not stream['valid'] or stream['flat_tokens'] != record.tokens:
            raise ValueError(f"{context}: invalid encoded audio targets: {stream['errors']}")
        if len(ids) > self.max_length:
            raise ValueError(f"{context}: sequence length {len(ids)} exceeds max_length={self.max_length}; truncation is disabled")
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:len(prefix)] = -100
        labels[attention_mask == 0] = -100
        if not (labels != -100).any():
            raise ValueError(f"{context}: no valid label tokens")
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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
    prompt_format: str = "legacy",
    chat_template_kwargs: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = AudioGenerationDataset(
        train_path, tokenizer, metadata, max_length, prompt_format, chat_template_kwargs
    )
    val_ds = AudioGenerationDataset(
        val_path, tokenizer, metadata, max_length, prompt_format, chat_template_kwargs
    )
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
