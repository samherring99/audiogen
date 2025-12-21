import json
import random
from pathlib import Path


def split_dataset(
    dataset_path: str = "./data/dataset.json",
    train_path: str = "./data/train_dataset.json",
    val_path: str = "./data/val_dataset.json",
    val_split: float = 0.1,
    seed: int = 42
):
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    random.seed(seed)
    random.shuffle(data)

    val_size = int(len(data) * val_split)
    train_size = len(data) - val_size

    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f"\nSplit:")
    print(f"  Training: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")

    print(f"\nSaving train set to {train_path}")
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)

    print(f"Saving validation set to {val_path}")
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)

    print("\nDataset split complete!")

    train_tokens = [sample['n_tokens'] for sample in train_data]
    val_tokens = [sample['n_tokens'] for sample in val_data]

    print("\nToken statistics:")
    print(f"  Train - Mean: {sum(train_tokens)/len(train_tokens):.0f}, Min: {min(train_tokens)}, Max: {max(train_tokens)}")
    print(f"  Val   - Mean: {sum(val_tokens)/len(val_tokens):.0f}, Min: {min(val_tokens)}, Max: {max(val_tokens)}")


if __name__ == "__main__":
    split_dataset(
        dataset_path="./data/dataset.json",
        train_path="./data/train_dataset.json",
        val_path="./data/val_dataset.json",
        val_split=0.1,
        seed=42
    )
