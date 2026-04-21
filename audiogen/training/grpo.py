from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import numpy as np

from audiogen.training.sft import SFTConfig
from audiogen.data.io import load_prompts
from audiogen.model.setup import load_trained_model
from audiogen.tokens.vocab import save_token_metadata
from audiogen.inference.generate import generate_token_ids, decode_token_ids
from audiogen.tokens.formatting import get_formatter


@dataclass
class GRPOConfig(SFTConfig):
    # GRPO rollout
    num_prompts_per_step: int = 4  # prompts sampled per step
    num_generations: int = 4       # completions per prompt per step
    max_new_tokens: int = 1200     # max audio tokens to generate
    temperature: float = 1.0
    top_p: float = 0.95

    # Reward
    reward_type: str = "whisper"   # "utmos" | "squim" | "clap" | "whisper" | "combined"
    clap_model: str = "laion/clap-htsat-unfused"
    whisper_model: str = "openai/whisper-base"
    min_audio_frames: int = 10     # completions shorter than this get reward -1

    # GRPO loss
    beta: float = 0.01             # KL penalty coefficient
    epsilon: float = 0.2           # PPO-style clip ratio

    # Training
    grpo_steps: int = 1000
    prompts_path: str = ""         # path to JSON list of prompts
    save_steps: int = 100


def _load_utmos():
    try:
        import utmos
        predictor = utmos.Score(device="cuda" if torch.cuda.is_available() else "cpu")
        return predictor
    except ImportError:
        raise ImportError("UTMOS not installed. Run: pip install utmos")


def _load_squim():
    import torchaudio
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    model = SQUIM_OBJECTIVE.get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    squim_sr = SQUIM_OBJECTIVE.sample_rate
    return model, squim_sr


def compute_squim_reward(waveforms: list[torch.Tensor | None], sample_rate: int, squim_model, squim_sr: int) -> list[float]:
    device = next(squim_model.parameters()).device
    rewards = []
    for wav in waveforms:
        if wav is None:
            rewards.append(-1.0)
            continue
        try:
            audio = wav.squeeze().unsqueeze(0)  # (1, samples)
            if sample_rate != squim_sr:
                import torchaudio
                audio = torchaudio.functional.resample(audio, sample_rate, squim_sr)
            audio = audio.to(device)
            with torch.no_grad():
                stoi, pesq, si_sdr = squim_model(audio)
            # si_sdr is most discriminative for speech quality, normalize loosely
            score = float(si_sdr.mean().item())
            # clip to reasonable range and normalize to [-1, 1]
            score = max(-20.0, min(20.0, score)) / 20.0
            rewards.append(score)
        except Exception as e:
            print(f"SQUIM scoring failed: {e}")
            rewards.append(-1.0)
    return rewards


def _load_clap(model_name: str):
    try:
        from transformers import ClapModel, ClapProcessor
        model = ClapModel.from_pretrained(model_name)
        processor = ClapProcessor.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        return model, processor
    except ImportError:
        raise ImportError("transformers not installed or CLAP model unavailable")


def compute_utmos_reward(waveforms: list[torch.Tensor | None], sample_rate: int, predictor) -> list[float]:
    rewards = []
    for wav in waveforms:
        if wav is None:
            rewards.append(-1.0)
            continue
        try:
            # utmos expects (samples,) numpy array at 16kHz
            audio = wav.squeeze().numpy()
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            score = predictor.score(audio, 16000)
            # normalize to [-1, 1] range (UTMOS is 1-5 MOS scale)
            reward = (score - 3.0) / 2.0
            rewards.append(float(reward))
        except Exception as e:
            print(f"UTMOS scoring failed: {e}")
            rewards.append(-1.0)
    return rewards


def _load_whisper(model_name: str):
    from transformers import pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)
    return pipe


def _normalize_text(text: str) -> str:
    import re
    text = text.lower()
    # strip common prompt prefixes
    for prefix in ("generate speech audio: ", "generate audio: "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 1.0
    # Simple WER via dynamic programming
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_whisper_reward(
    waveforms: list[torch.Tensor | None],
    prompts: list[str],
    sample_rate: int,
    whisper_pipe,
) -> list[float]:
    rewards = []
    for wav, prompt in zip(waveforms, prompts):
        if wav is None:
            rewards.append(-1.0)
            continue
        try:
            audio = wav.squeeze().numpy()
            # resample to 16kHz for whisper
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            result = whisper_pipe({"array": audio, "sampling_rate": 16000})
            transcription = _normalize_text(result["text"])
            reference = _normalize_text(prompt)
            wer = _wer(reference, transcription)
            # convert WER to reward: 0 WER = 1.0 reward, 1+ WER = -1.0 reward
            reward = 1.0 - 2.0 * min(wer, 1.0)
            rewards.append(float(reward))
        except Exception as e:
            print(f"Whisper scoring failed: {e}")
            rewards.append(-1.0)
    return rewards


def compute_clap_reward(
    waveforms: list[torch.Tensor | None],
    prompts: list[str],
    sample_rate: int,
    clap_model,
    clap_processor,
) -> list[float]:
    rewards = []
    device = next(clap_model.parameters()).device
    for wav, prompt in zip(waveforms, prompts):
        if wav is None:
            rewards.append(-1.0)
            continue
        try:
            audio = wav.squeeze().numpy()
            clap_sr = 48000
            if sample_rate != clap_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=clap_sr)
            inputs = clap_processor(
                text=[prompt],
                audio=[audio],
                return_tensors="pt",
                sampling_rate=clap_sr,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = clap_model(**inputs)
            similarity = torch.nn.functional.cosine_similarity(
                outputs.text_embeds, outputs.audio_embeds
            ).item()
            rewards.append(float(similarity))
        except Exception as e:
            print(f"CLAP scoring failed: {e}")
            rewards.append(-1.0)
    return rewards


def grpo_loss(
    log_probs: torch.Tensor,        # (G,) log probs of completion under current model
    ref_log_probs: torch.Tensor,    # (G,) log probs under reference model
    rewards: torch.Tensor,          # (G,) scalar rewards
    beta: float = 0.01,
    epsilon: float = 0.2,
) -> torch.Tensor:
    # Normalize rewards within group
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Direct policy gradient: maximize E[reward * log_prob] + KL penalty
    # No PPO clipping — simpler and works better when current == ref at step 0
    kl = torch.clamp(log_probs - ref_log_probs.detach(), -10.0, 10.0)
    policy_loss = -(log_probs * rewards)

    loss = (policy_loss + beta * kl).mean()
    return loss


def _completion_log_prob(model, tokenizer, input_ids: torch.Tensor, completion_ids: torch.Tensor, device: str, no_grad: bool = False) -> torch.Tensor:
    full = torch.cat([input_ids, completion_ids.unsqueeze(0)], dim=1).to(device)
    if no_grad:
        with torch.no_grad():
            logits = model(full).logits
    else:
        logits = model(full).logits  # (1, T, V)
    # log probs for completion tokens only
    log_probs = torch.nn.functional.log_softmax(logits[0, input_ids.shape[1]-1:-1], dim=-1)
    token_log_probs = log_probs.gather(1, completion_ids.unsqueeze(1).to(device)).squeeze(1)
    return token_log_probs.mean()  # mean per token to avoid underflow on long sequences


def run_grpo(config: GRPOConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        import wandb
        wandb.init(project="audiogen-grpo", config=vars(config))
    except Exception:
        pass

    print(f"Loading model from: {config.resume_from_checkpoint}")
    model, tokenizer, meta = load_trained_model(config.resume_from_checkpoint)
    model = model.to(device=device, dtype=torch.bfloat16).train()
    print(f"Model dtype: {next(model.parameters()).dtype}")
    save_token_metadata(meta, output_dir / "audio_token_metadata.json")
    tokenizer.save_pretrained(output_dir)

    # Reference model (frozen copy for KL)
    import copy
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Load codec for decoding
    from audiogen.codecs.moss import MOSSCodec
    codec = MOSSCodec.load(device=device, num_codebooks=meta.num_codebooks)

    formatter = get_formatter(meta.codec, meta.num_codebooks)

    # Load reward model(s)
    utmos_predictor = None
    squim_model = squim_sr = None
    clap_model = clap_processor = None
    whisper_pipe = None
    if config.reward_type in ("utmos", "combined"):
        print("Loading UTMOS...")
        utmos_predictor = _load_utmos()
    if config.reward_type == "squim":
        print("Loading SQUIM...")
        squim_model, squim_sr = _load_squim()
    if config.reward_type in ("clap", "combined"):
        print("Loading CLAP...")
        clap_model, clap_processor = _load_clap(config.clap_model)
    if config.reward_type in ("whisper", "combined"):
        print("Loading Whisper...")
        whisper_pipe = _load_whisper(config.whisper_model)

    # Load prompts
    prompts = load_prompts(config.prompts_path)
    print(f"Loaded {len(prompts)} prompts")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    import random
    rng = random.Random(42)

    # Build prefix for a prompt
    def build_prefix_tensor(prompt):
        messages = formatter.format_chat_messages(prompt, output=None)
        if tokenizer.chat_template is None:
            from transformers import AutoTokenizer as _AT
            tokenizer.chat_template = _AT.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").chat_template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<audio>"
        prefix_ids = tokenizer.encode(text[:-len("<audio>")], add_special_tokens=False) + [meta.audio_start_token_id]
        return torch.tensor([prefix_ids], dtype=torch.long).to(device)

    for step in range(config.grpo_steps):
        # Sample num_prompts_per_step prompts, generate num_generations each
        step_prompts = [rng.choice(prompts) for _ in range(config.num_prompts_per_step)]

        model.eval()
        all_completions = []   # list of (prompt, token_ids) pairs
        all_waveforms = []
        all_prompt_labels = []  # which prompt each completion belongs to

        for prompt in step_prompts:
            for _ in range(config.num_generations):
                with torch.no_grad():
                    token_ids = generate_token_ids(
                        model, tokenizer, prompt, meta,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        device=device,
                    )
                all_completions.append((prompt, token_ids))
                all_prompt_labels.append(prompt)

                codec_output = decode_token_ids(token_ids, tokenizer, formatter, meta)
                if codec_output is not None and codec_output.num_frames >= config.min_audio_frames:
                    wav = codec.decode(codec_output)
                    all_waveforms.append(wav)
                else:
                    all_waveforms.append(None)

        model.train()

        # Score all completions
        if config.reward_type == "utmos":
            rewards_list = compute_utmos_reward(all_waveforms, codec.sample_rate, utmos_predictor)
        elif config.reward_type == "squim":
            rewards_list = compute_squim_reward(all_waveforms, codec.sample_rate, squim_model, squim_sr)
        elif config.reward_type == "clap":
            rewards_list = compute_clap_reward(all_waveforms, all_prompt_labels, codec.sample_rate, clap_model, clap_processor)
        elif config.reward_type == "whisper":
            rewards_list = compute_whisper_reward(all_waveforms, all_prompt_labels, codec.sample_rate, whisper_pipe)
        else:  # combined: whisper + squim
            whisper_r = compute_whisper_reward(all_waveforms, all_prompt_labels, codec.sample_rate, whisper_pipe)
            squim_r = compute_squim_reward(all_waveforms, codec.sample_rate, squim_model, squim_sr)
            rewards_list = [0.6 * w + 0.4 * s for w, s in zip(whisper_r, squim_r)]

        # Normalize rewards per-prompt group
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
        normalized_rewards = torch.zeros_like(rewards_tensor)
        for i, prompt in enumerate(step_prompts):
            idxs = [j for j, p in enumerate(all_prompt_labels) if p == prompt]
            group = rewards_tensor[idxs]
            normalized_rewards[idxs] = (group - group.mean()) / (group.std() + 1e-8)

        mean_reward = rewards_tensor.mean().item()
        print(f"Step {step}: mean_reward={mean_reward:.3f} rewards={[f'{r:.3f}' for r in rewards_list]}")


        # Compute log probs and loss per prompt group
        log_probs_list = []
        ref_log_probs_list = []
        for prompt, token_ids in all_completions:
            if len(token_ids) == 0:
                log_probs_list.append(torch.tensor(-9.0, device=device))
                ref_log_probs_list.append(torch.tensor(-9.0, device=device))
                continue
            prefix_tensor = build_prefix_tensor(prompt)
            lp = _completion_log_prob(model, tokenizer, prefix_tensor, token_ids, device, no_grad=False)
            ref_lp = _completion_log_prob(ref_model, tokenizer, prefix_tensor, token_ids, device, no_grad=True)
            log_probs_list.append(lp)
            ref_log_probs_list.append(ref_lp)

        log_probs = torch.stack(log_probs_list)
        ref_log_probs = torch.stack(ref_log_probs_list)

        loss = grpo_loss(log_probs, ref_log_probs, normalized_rewards.to(device), beta=config.beta, epsilon=config.epsilon)

        if not torch.isfinite(loss):
            print(f"Step {step}: non-finite loss {loss.item()}, skipping update")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        # Check for NaN weights after update — if corrupted, reload from checkpoint
        has_nan = any(not torch.isfinite(p).all() for p in model.parameters())
        if has_nan:
            print(f"Step {step}: NaN weights detected after optimizer step, skipping")
            optimizer.zero_grad()
            continue

        mean_reward = rewards_tensor.mean().item()
        print(f"Step {step}: loss={loss.item():.4f} mean_reward={mean_reward:.3f} grad_norm={grad_norm:.3f}")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"loss": loss.item(), "mean_reward": mean_reward, "grad_norm": grad_norm}, step=step)
        except Exception:
            pass

        if (step + 1) % config.save_steps == 0:
            ckpt_dir = output_dir / f"checkpoint-{step+1}"
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            save_token_metadata(meta, ckpt_dir / "audio_token_metadata.json")
            print(f"Saved checkpoint to {ckpt_dir}")

    model.save_pretrained(str(output_dir))
    print("GRPO training complete.")
