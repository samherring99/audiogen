import json
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

class AudioGenerationDataset(Dataset):
    # formats audio tokens like: <audio>[AUDIO_S0_123][AUDIO_S0_456]...</audio>
    def __init__(
        self,
        dataset_path: str,
        tokenizer_path: str,
        num_scales: int = 3, # default from dataset gen
        num_tokens_per_scale: int = 4096,
        max_length: Optional[int] = None
    ):
        self.dataset_path = Path(dataset_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.num_scales = num_scales
        self.num_tokens_per_scale = num_tokens_per_scale
        self.max_length = max_length

        with open(dataset_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {dataset_path}")

        metadata_path = Path(tokenizer_path) / "audio_token_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"Audio token range: {self.metadata['audio_token_start_id']} - {self.metadata['audio_token_end_id']}")

    def __len__(self):
        return len(self.data)

    def audio_tokens_to_string(self, audio_token_ids: List[int], scale_lengths: List[int]) -> str:
        token_strings = []
        offset = 0

        # <s0>[AUDIO_S0_X]...</s0><s1>[AUDIO_S1_Y]...</s1><s2>[AUDIO_S2_Z]...</s2>
        for scale_idx, scale_length in enumerate(scale_lengths):
            # Scale start delimiter
            token_strings.append(f"<s{scale_idx}>")

            scale_tokens = audio_token_ids[offset:offset + scale_length]

            for token_id in scale_tokens:
                token_str = f"[AUDIO_S{scale_idx}_{token_id}]"
                token_strings.append(token_str)

            # scale end delimiter
            token_strings.append(f"</s{scale_idx}>")

            offset += scale_length

        return "".join(token_strings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        prompt = sample["prompt"]
        audio_tokens = sample["audio_tokens"]
        scale_lengths = sample.get("scale_lengths", None)

        if scale_lengths is None:
            tokens_per_scale = len(audio_tokens) // self.num_scales
            scale_lengths = [tokens_per_scale] * self.num_scales

        audio_sequence = self.audio_tokens_to_string(audio_tokens, scale_lengths)

        # TODO: maybe fix the generation prompting here
        messages = [
            {"role": "user", "content": f"Generate audio: [{prompt}]"},
            {"role": "assistant", "content": f"<audio>{audio_sequence}</audio>"}
        ]

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            padding=False,
            truncation=True if self.max_length else False,
            max_length=self.max_length
        )

        input_ids = encoded.squeeze(0)

        labels = input_ids.clone()

        # mask out the prompt part so we only compute loss on audio tokens
        audio_start_id = self.metadata["audio_start_token_id"]
        audio_start_positions = (input_ids == audio_start_id).nonzero(as_tuple=True)[0]

        if len(audio_start_positions) > 0:
            audio_start_pos = audio_start_positions[0].item()
            labels[:audio_start_pos + 1] = -100
        else:
            # mask everything - this shouldn't happen
            print(f"Warning: No <audio> token found in sample {idx}")
            labels[:] = -100

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def create_dataloaders(
    train_dataset_path: str,
    val_dataset_path: Optional[str],
    tokenizer_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    **dataset_kwargs
):
    train_dataset = AudioGenerationDataset(
        train_dataset_path,
        tokenizer_path,
        **dataset_kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = None
    if val_dataset_path:
        val_dataset = AudioGenerationDataset(
            val_dataset_path,
            tokenizer_path,
            **dataset_kwargs
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing AudioGenerationDataset...")

    dataset = AudioGenerationDataset(
        dataset_path="./data/dataset.json",
        tokenizer_path="./models/qwen-audio",
        num_scales=3
    )

    print(f"\nDataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  Number of tokens: {sample['input_ids'].shape[0]}")

    tokenizer = dataset.tokenizer
    decoded_input = tokenizer.decode(sample['input_ids'])
    print(f"\nDecoded input:")
    print(decoded_input[:500] + "..." if len(decoded_input) > 500 else decoded_input)

    num_masked = (sample['labels'] == -100).sum().item()
    print(f"\nLabel masking:")
    print(f"  Masked tokens: {num_masked}")
    print(f"  Training tokens: {sample['labels'].shape[0] - num_masked}")

    print("\nTesting 10 samples for errors...")
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        assert sample['input_ids'].shape == sample['labels'].shape
        assert sample['input_ids'].shape == sample['attention_mask'].shape

    print("All tests passed!")
