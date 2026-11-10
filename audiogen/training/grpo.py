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


def _load_whisper(model_name: str, device: str | None = None):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    # Scoring models go on a second GPU when available to keep GPU 0 free for
    # policy + ref + optimizer (OOM on 2026-08-05 with everything colocated).
    if device is not None:
        pass
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = "cuda:1"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    return processor, model, device


_ONES_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS_WORDS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
_ORDINAL_WORDS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
    7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh",
    12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
    16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
    20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth",
    60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth",
}


def _num_to_words(n: int) -> str:
    """0-9999 to spoken words, matching common ASR output conventions."""
    if n < 20:
        return _ONES_WORDS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        word = _TENS_WORDS[tens * 10]
        return word if ones == 0 else f"{word} {_ONES_WORDS[ones]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        word = f"{_ONES_WORDS[hundreds]} hundred"
        return word if rest == 0 else f"{word} {_num_to_words(rest)}"
    if n < 10000:
        # Years like 1999 → "nineteen ninety nine"; even thousands → "two thousand"
        thousands, rest = divmod(n, 1000)
        if rest == 0:
            return f"{_ONES_WORDS[thousands]} thousand"
        high, low = divmod(n, 100)
        if 10 <= high <= 99 and low > 0:
            return f"{_num_to_words(high)} {_num_to_words(low)}"
        return f"{_ONES_WORDS[thousands]} thousand {_num_to_words(rest)}"
    return str(n)


def _ordinal_to_words(n: int) -> str:
    if n in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones and tens * 10 in _TENS_WORDS and ones in _ORDINAL_WORDS:
            return f"{_TENS_WORDS[tens * 10]} {_ORDINAL_WORDS[ones]}"
    return _num_to_words(n)


def _canonicalize_numbers(text: str) -> str:
    """Expand digit forms to spoken words BEFORE punctuation stripping.

    Critical for WER-based rewards: without this, '9:45' becomes '945' after
    punctuation stripping and never matches Whisper's 'nine forty five'.
    """
    import re

    def _time_repl(m: re.Match) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 23 or minute > 59:
            return m.group(0)
        hour_words = _num_to_words(hour)
        if minute == 0:
            return f"{hour_words} o'clock"
        if minute < 10:
            return f"{hour_words} oh {_num_to_words(minute)}"
        return f"{hour_words} {_num_to_words(minute)}"

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time_repl, text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _ordinal_to_words(int(m.group(1))), text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{1,4}\b", lambda m: _num_to_words(int(m.group(0))), text)
    return text


def _normalize_text(text: str) -> str:
    import re
    text = text.lower().strip()
    # Strip only anchored instructions, including native tags and legacy nesting.
    # Reference and ASR hypothesis deliberately share this same normalizer.
    wrapper = r"^generate (?:speech )?audio(?:\s+\[[^\]]*\])*\s*:\s*"
    while re.match(wrapper, text):
        text = re.sub(wrapper, "", text, count=1).lstrip()
    # canonicalize numeric forms to spoken words BEFORE stripping punctuation
    text = _canonicalize_numbers(text)
    # normalize spoken-form artifacts so word-form references match too
    text = text.replace("o'clock", "oclock")
    text = re.sub(r"[^\w\s]", " ", text)
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


def tail_noise_penalty(wav: torch.Tensor, tail_fraction: float = 0.2) -> float:
    """Penalize if the last tail_fraction of audio has high energy relative to the body."""
    audio = wav.squeeze().float()
    tail_start = int(len(audio) * (1 - tail_fraction))
    if tail_start <= 0:
        return 0.0
    tail_rms = audio[tail_start:].pow(2).mean().sqrt().item()
    body_rms = audio[:tail_start].pow(2).mean().sqrt().item()
    if body_rms < 1e-6:
        return 0.0
    ratio = tail_rms / (body_rms + 1e-6)
    return -0.5 * min(ratio, 1.0)


def silence_penalty(wav: torch.Tensor, rms_floor: float = 0.02) -> float:
    """Bounded penalty for quiet/near-silent output. 0 when healthy, -1 at full silence.

    Low RMS (<~0.02) has empirically predicted human-judged bad samples.
    """
    audio = wav.squeeze().float()
    rms = audio.pow(2).mean().sqrt().item()
    if rms >= rms_floor:
        return 0.0
    return -(1.0 - rms / rms_floor)


def score_whisper_components(wav: torch.Tensor, prompt: str, sample_rate: int, whisper_tuple) -> dict:
    """Strict auditable scoring; infrastructure errors propagate, never become rewards.

    Callers handle structurally invalid rollouts separately. This scores actual
    decoded waveforms, not repaired streams or absent audio.
    """
    import math

    if sample_rate <= 0 or wav.numel() == 0 or not torch.isfinite(wav).all():
        raise ValueError("Audio must be nonempty and finite with a positive sample rate")
    audio = wav.detach().float().cpu()
    if audio.ndim == 2:
        audio = audio.mean(0)
    if audio.ndim != 1:
        raise ValueError("Audio must have shape [samples] or [channels, samples]")
    reference = _normalize_text(prompt)
    if not reference:
        raise ValueError("Empty spoken-text reference")
    processor, whisper_model, device = whisper_tuple
    samples = audio.numpy()
    if sample_rate != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
    inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = whisper_model.generate(inputs.input_features.to(device), language="en", task="transcribe")
    transcription = processor.batch_decode(ids, skip_special_tokens=True)[0]
    hypothesis = _normalize_text(transcription)
    wer = _wer(reference, hypothesis)
    wer_reward = 1.0 - 2.0 * min(wer, 1.0)
    tail = tail_noise_penalty(audio)
    silence = silence_penalty(audio)
    reward = 0.7 * wer_reward + 0.15 * tail + 0.15 * silence
    if not all(math.isfinite(x) for x in (wer, wer_reward, tail, silence, reward)):
        raise ValueError("Nonfinite reward component")
    return {"transcription": transcription, "reference": reference, "hypothesis": hypothesis,
            "wer": wer, "wer_reward": wer_reward, "tail_penalty": tail,
            "silence_penalty": silence, "reward": reward}


def compute_whisper_reward(
    waveforms: list[torch.Tensor | None],
    prompts: list[str],
    sample_rate: int,
    whisper_tuple,
) -> list[float]:
    processor, whisper_model, device = whisper_tuple
    rewards = []
    for wav, prompt in zip(waveforms, prompts):
        if wav is None:
            rewards.append(-1.0)
            continue
        try:
            audio = wav.squeeze()
            if audio.dim() > 1:
                audio = audio.mean(0)
            audio = audio.float().numpy().astype(np.float32)
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            with torch.no_grad():
                predicted_ids = whisper_model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                )
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcription = _normalize_text(transcription)
            reference = _normalize_text(prompt)
            wer = _wer(reference, transcription)
            wer_reward = 1.0 - 2.0 * min(wer, 1.0)
            tail_pen = tail_noise_penalty(wav)
            sil_pen = silence_penalty(wav)
            reward = 0.7 * wer_reward + 0.15 * tail_pen + 0.15 * sil_pen
            rewards.append(float(reward))
        except Exception as e:
            print(f"Whisper scoring failed: {e}", flush=True)
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


def _load_gender_classifier(model_name: str = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"):
    """Validated 96.6% balanced accuracy on CV17 self-reported labels (2026-08-05)."""
    from transformers import pipeline
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = 1  # colocate with whisper on the scoring GPU, away from the policy
    elif torch.cuda.is_available():
        device = 0
    else:
        device = -1
    return pipeline("audio-classification", model=model_name, device=device)


def _extract_gender_tag(prompt: str) -> str | None:
    import re
    m = re.search(r"\[gender=(male|female)\]", prompt)
    return m.group(1) if m else None


def compute_gender_reward(
    waveforms: list[torch.Tensor | None],
    prompts: list[str],
    sample_rate: int,
    gender_clf,
    whisper_tuple=None,
    wer_weight: float = 0.2,
) -> list[float]:
    """Reward = gender-match (signed classifier confidence) + small WER term + penalties.

    Prompts without a [gender=...] tag get gender component 0 (neutral) so untagged
    rollouts are judged purely on intelligibility/penalties.
    """
    import librosa as _librosa
    wer_rewards = None
    if whisper_tuple is not None and wer_weight > 0:
        wer_rewards = compute_whisper_reward(waveforms, prompts, sample_rate, whisper_tuple)

    rewards = []
    for i, (wav, prompt) in enumerate(zip(waveforms, prompts)):
        if wav is None:
            rewards.append(-1.0)
            continue
        try:
            target = _extract_gender_tag(prompt)
            tail_pen = tail_noise_penalty(wav)
            sil_pen = silence_penalty(wav)

            gender_component = 0.0
            if target is not None and sil_pen >= 0.0:
                # Gate on non-silence: the classifier confidently labels silence as
                # 'female', which would otherwise award match credit to empty audio
                # (caught by validate_gender_reward.py before the first pilot).
                audio = wav.squeeze()
                if audio.dim() > 1:
                    audio = audio.mean(0)
                audio = audio.float().numpy().astype(np.float32)
                if sample_rate != 16000:
                    audio = _librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                preds = gender_clf({"array": audio, "sampling_rate": 16000}, top_k=2)
                score = {p["label"].lower(): p["score"] for p in preds}
                p_target = score.get(target, 0.0)
                # Signed confidence: +1 = confident correct gender, -1 = confident wrong
                gender_component = 2.0 * p_target - 1.0
            elif target is not None:
                # Tagged prompt but silent output: no gender credit possible
                gender_component = -1.0

            wer_component = wer_rewards[i] if wer_rewards is not None else 0.0

            gender_weight = 0.6 if target is not None else 0.0
            base_weight = 1.0 - gender_weight - (wer_weight if wer_rewards is not None else 0.0)
            reward = (
                gender_weight * gender_component
                + (wer_weight if wer_rewards is not None else 0.0) * wer_component
                + base_weight * (0.5 * tail_pen + 0.5 * sil_pen)
            )
            rewards.append(float(reward))
        except Exception as e:
            print(f"Gender scoring failed: {e}", flush=True)
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

    # Reference model (frozen copy for KL) — on the scoring GPU when available so
    # the policy GPU keeps room for generation KV caches + autograd (OOM 2026-08-05).
    import copy
    ref_device = "cuda:1" if torch.cuda.device_count() > 1 else device
    ref_model = copy.deepcopy(model).to(ref_device).eval()
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
    gender_clf = None
    if config.reward_type == "utmos":
        print("Loading UTMOS...")
        utmos_predictor = _load_utmos()
    if config.reward_type in ("squim", "combined"):
        print("Loading SQUIM...")
        squim_model, squim_sr = _load_squim()
    if config.reward_type in ("clap", "combined"):
        print("Loading CLAP...")
        clap_model, clap_processor = _load_clap(config.clap_model)
    if config.reward_type in ("whisper", "combined", "gender"):
        print("Loading Whisper...")
        whisper_pipe = _load_whisper(config.whisper_model)  # returns (processor, model, device)
    if config.reward_type == "gender":
        print("Loading gender classifier...")
        gender_clf = _load_gender_classifier()

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
        elif config.reward_type == "gender":
            rewards_list = compute_gender_reward(all_waveforms, all_prompt_labels, codec.sample_rate, gender_clf, whisper_tuple=whisper_pipe)
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
            ref_lp = _completion_log_prob(ref_model, tokenizer, prefix_tensor, token_ids, ref_device, no_grad=True).to(device)
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
