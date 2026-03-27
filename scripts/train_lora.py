"""LoRA fine-tuning."""

import argparse

from audiogen.training.lora import LoRAConfig, run_lora


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--codec", default="moss", choices=["moss", "snac"])
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()

    config = LoRAConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        codec=args.codec,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        fp16=not args.no_fp16,
    )
    run_lora(config)


if __name__ == "__main__":
    main()
