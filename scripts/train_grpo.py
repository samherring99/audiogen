"""GRPO fine-tuning with audio quality reward (UTMOS / CLAP)."""

import argparse

from audiogen.training.grpo import GRPOConfig, run_grpo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", required=True, help="Path to SFT checkpoint to start from")
    parser.add_argument("--prompts_path", required=True, help="Path to JSON file with list of prompts")
    parser.add_argument("--output_dir", required=True)

    # Rollout
    parser.add_argument("--num_prompts_per_step", type=int, default=4, help="Prompts sampled per step")
    parser.add_argument("--num_generations", type=int, default=4, help="Completions per prompt per step")
    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)

    # Reward
    parser.add_argument("--reward_type", default="whisper", choices=["utmos", "squim", "clap", "whisper", "combined", "gender"])
    parser.add_argument("--clap_model", default="laion/clap-htsat-unfused")
    parser.add_argument("--whisper_model", default="openai/whisper-base")
    parser.add_argument("--min_audio_frames", type=int, default=10)

    # GRPO loss
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip ratio")

    # Training
    parser.add_argument("--grpo_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=100)

    # Dummy required fields for SFTConfig
    parser.add_argument("--data_path", default="")
    args = parser.parse_args()

    config = GRPOConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        prompts_path=args.prompts_path,
        num_prompts_per_step=args.num_prompts_per_step,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reward_type=args.reward_type,
        clap_model=args.clap_model,
        whisper_model=args.whisper_model,
        min_audio_frames=args.min_audio_frames,
        beta=args.beta,
        epsilon=args.epsilon,
        grpo_steps=args.grpo_steps,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
    )
    run_grpo(config)


if __name__ == "__main__":
    main()
