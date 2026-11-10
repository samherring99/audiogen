import argparse

from audiogen.training.sft import SFTConfig, run_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Legacy input: split and write train/val files. Prefer prepared --train_path/--val_path.")
    parser.add_argument("--train_path", help="Prepared training JSONL, used directly without copying or splitting")
    parser.add_argument("--val_path", help="Prepared validation JSONL, used directly")
    parser.add_argument("--revision", help="Pinned base-model/tokenizer revision")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--prompt_format", choices=["legacy", "speech_v1"], default=None, help="Default: speech_v1 for base model; checkpoint's saved policy for initialization/resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name")
    parser.add_argument("--max_steps", type=int, default=-1, help="Total optimizer-step budget; -1 uses epochs")
    parser.add_argument("--stop_after_step", type=int, default=None, help="Operational log/eval/save/pause after this global update; does not change scheduler budget. Raise or omit explicitly on true resume.")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", default="wandb", help="Trainer integration, e.g. wandb or none")
    parser.add_argument("--plateau_patience", type=int, default=0, help="Nonimproving evaluations before clean save/stop; 0 disables")
    parser.add_argument("--plateau_min_steps", type=int, default=0)
    parser.add_argument("--plateau_warmup_steps", type=int, default=0)
    parser.add_argument("--plateau_min_delta", type=float, default=0.0)
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
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--lr_scheduler", default=None, help="Override lr_scheduler_type")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--resume_from_checkpoint", default=None, help="Legacy WEIGHTS-ONLY initialization alias (fresh optimizer/state)")
    checkpoints.add_argument("--initialize_from_checkpoint", default=None, help="Initialize weights/tokenizer from checkpoint with fresh Trainer state")
    checkpoints.add_argument("--trainer_resume_from_checkpoint", default=None, help="True same-policy resume: optimizer/scheduler/RNG/global step restored")
    parser.add_argument("--unfreeze_all", action="store_true", help="Train all parameters, not just embeddings")
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=0, help="Unfreeze last N transformer layers + embeddings")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing for large models")
    parser.add_argument("--fsdp", default=None, choices=["full_shard auto_wrap"], help="Opt-in torchrun FSDP1; requires full unfreeze and prepared train/val")
    parser.add_argument("--fsdp_config", default=None, help="FSDP JSON config path (validated supported settings only)")
    parser.add_argument("--save_steps", type=int, default=None, help="Override checkpoint save frequency")
    parser.add_argument("--eval_steps", type=int, default=None, help="Override evaluation frequency")
    parser.add_argument("--logging_steps", type=int, default=None, help="Override logging frequency")
    args = parser.parse_args()

    config = SFTConfig(
        data_path=args.data_path,
        train_path=args.train_path,
        val_path=args.val_path,
        revision=args.revision,
        dtype=args.dtype,
        prompt_format=args.prompt_format,
        seed=args.seed,
        run_name=args.run_name,
        max_steps=args.max_steps,
        stop_after_step=args.stop_after_step,
        per_device_eval_batch_size=args.eval_batch_size,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        initialize_from_checkpoint=args.initialize_from_checkpoint,
        trainer_resume_from_checkpoint=args.trainer_resume_from_checkpoint,
        plateau_patience=args.plateau_patience,
        plateau_min_steps=args.plateau_min_steps,
        plateau_warmup_steps=args.plateau_warmup_steps,
        plateau_min_delta=args.plateau_min_delta,
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
        fp16=False,
        bf16=not args.no_bf16,
        resume_from_checkpoint=args.resume_from_checkpoint,
        unfreeze_all=args.unfreeze_all,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        fsdp=args.fsdp,
        fsdp_config=args.fsdp_config,
        **({'save_steps': args.save_steps} if args.save_steps is not None else {}),
        **({'eval_steps': args.eval_steps} if args.eval_steps is not None else {}),
        **({'logging_steps': args.logging_steps} if args.logging_steps is not None else {}),
        **({'lr_scheduler_type': args.lr_scheduler} if args.lr_scheduler else {}),
    )
    run_sft(config)


if __name__ == "__main__":
    main()
