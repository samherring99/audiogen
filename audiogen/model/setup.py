from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from audiogen.tokens.vocab import (
    TokenMetadata,
    build_moss_vocab,
    build_snac_vocab,
    expand_tokenizer,
    load_token_metadata,
    resize_model_embeddings,
    save_token_metadata,
)


logger = logging.getLogger(__name__)


def _checkpoint_signature(path):
    """Cheap local artifact identity; never read/copy multi-GB model tensors."""
    path = Path(path)
    names = {"config.json"}
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (path / name).is_file():
            names.add(name)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (path / name).is_file():
            names.add(name)
            names.update(json.loads((path / name).read_text())["weight_map"].values())
    if len(names) == 1:
        raise ValueError("Native Trainer resume requires local checkpoint weights")
    signature = []
    for name in sorted(names):
        stat = (path / name).stat()
        signature.append((name, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(signature)


def _native_resume_model(model):
    import transformers
    if transformers.__version__ != "5.6.2" or model.config.model_type != "nemotron_h":
        return False
    from transformers import NemotronHForCausalLM
    return type(model) is NemotronHForCausalLM


_RESUME_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _native_state_fingerprint(model):
    """SHA-256 of every parameter/buffer's name, physical dtype, shape and bytes.

    Only the validated native dense/contiguous layout is supported; fail closed
    rather than flattening a noncontiguous tensor into a model-sized allocation.
    CPU data is hashed through zero-copy memoryviews. On CUDA, only one <=8 MiB
    chunk is copied to CPU at a time (blocking); there is no second model, dtype
    cast, whole-tensor clone, sampling or approximate numeric checksum.

    Device is deliberately excluded: a byte-preserving CPU -> CUDA move is OK.
    This costs two full state scans: after conversion on CPU, then before resume
    skips HF's raw loader. At 30B BF16 this is ~60 GB per scan, ~120 GB SHA input
    and ~60 GB D2H if resumed on GPU (plus FP32 buffers). Runtime is UNMEASURED
    at that scale: estimate 120 GB / SHA throughput + 60 GB / D2H throughput,
    plus per-chunk overhead. Do not mistake bounded memory for a cheap check.
    """
    digest = hashlib.sha256()
    for kind, tensors in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in sorted(tensors):
            if (tensor.device.type not in ("cpu", "cuda") or tensor.layout != torch.strided
                    or not tensor.is_contiguous()):
                raise ValueError("Native verified conversion preload requires dense contiguous CPU/CUDA tensors")
            header = json.dumps([kind, name, str(tensor.dtype), list(tensor.shape)],
                                separators=(",", ":")).encode("utf-8")
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            # view, not reshape/contiguous: never silently allocate a full tensor.
            raw = tensor.detach().view(-1).view(torch.uint8)
            digest.update(raw.numel().to_bytes(8, "little"))
            for start in range(0, raw.numel(), _RESUME_HASH_CHUNK_BYTES):
                chunk = raw[start:start + _RESUME_HASH_CHUNK_BYTES].to("cpu")
                digest.update(memoryview(chunk.numpy()))
                del chunk  # release the CPU copy before allocating the next
    return digest.digest()


def _load_model(model_path, *, dtype, revision=None):
    # Capture before AND after load to reject a checkpoint replaced during loading.
    config_path = Path(model_path) / "config.json"
    local = config_path.is_file() and json.loads(config_path.read_text()).get("model_type") == "nemotron_h"
    signature = _checkpoint_signature(model_path) if local else None
    model, info = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, revision=revision, output_loading_info=True,
    )
    # Keep conversion diagnostics available to callers, including unused MTP weights.
    # An ignored source tensor is not part of the native model and is never trained.
    model.audiogen_loading_info = info
    required = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    if (local and _native_resume_model(model) and required <= info.keys()
            and not any(info.values()) and signature == _checkpoint_signature(model_path)):
        # One-use receipt of the actual post-conversion state, including buffers.
        # Identity/version counters alone miss .data writes and dtype conversions.
        # Bounded full scans are intentional; see _native_state_fingerprint cost.
        model._audiogen_resume_preload = (
            id(model), str(Path(model_path).resolve()), signature,
            _native_state_fingerprint(model),
        )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        logger.warning("Native model loading diagnostics (unexpected keys were not loaded or trained): %s", info)
    else:
        logger.info("Native model loading diagnostics: %s", info)
    return model


def load_base_model(
    model_path: str, device: str = "cpu", *, dtype=torch.float16,
    revision: str | None = None, prompt_format: str = "speech_v1",
):
    """Load a native model without implicit dtype conversion; pin both artifacts."""
    if prompt_format not in ("legacy", "speech_v1"):
        raise ValueError(f"Unsupported prompt_format: {prompt_format!r}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    model = _load_model(model_path, dtype=dtype, revision=revision)
    model.config.audiogen = {
        "base_model": str(model_path), "revision": revision,
        "prompt_format": prompt_format,
        "chat_template_kwargs": {"enable_thinking": False} if prompt_format == "speech_v1" else {},
    }
    model = model.to(device)
    return model, tokenizer


def configure_trainable_parameters(model, *, unfreeze_all: bool = False) -> None:
    """Train full input/output embedding modules, or all parameters (not row masks)."""
    for param in model.parameters():
        param.requires_grad = unfreeze_all
    if not unfreeze_all:
        for module in (model.get_input_embeddings(), model.get_output_embeddings()):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True


AUDIO_INIT_TEXT_SCALE = "text_scale"
AUDIO_INIT_LEGACY_TINY = "legacy_tiny"


def _init_new_rows(weight, old_vocab_size: int, audio_init: str) -> None:
    """Initialize rows [old_vocab_size:] in place.

    text_scale: mean of the existing rows plus Gaussian noise at the existing
    per-element std, so a new row has the same RMS as a text row and passes the
    first pre-norm RMSNorm at unit scale. legacy_tiny: N(0, 0.001*std) — kept
    only for regression tests/forensics. With RMSNorm eps=1e-5 and a hidden size
    of a few thousand, legacy_tiny rows have mean(x^2) ~1e-10, four orders of
    magnitude below eps, so every audio token is normalized to ~0 and the
    backbone cannot distinguish audio inputs (observed: zero text conditioning,
    ~6.2 CE floor across LRs/parents on Nemotron 3.5).
    """
    old = weight[:old_vocab_size].detach().float()
    std = old.std().item()
    if audio_init == AUDIO_INIT_TEXT_SCALE:
        mean = old.mean(dim=0)
        new = weight[old_vocab_size:]
        noise = torch.empty(new.shape, dtype=torch.float32, device=weight.device).normal_(mean=0, std=std)
        new.copy_((mean.to(weight.device) + noise).to(weight.dtype))
    elif audio_init == AUDIO_INIT_LEGACY_TINY:
        weight[old_vocab_size:].normal_(mean=0, std=std * 0.001)
    else:
        raise ValueError(f"Unknown audio_init: {audio_init!r}")


def expand_for_audio(
    model,
    tokenizer,
    codec: str,
    num_codebooks: int,
    codebook_size: int,
    save_dir: str | Path | None = None,
    audio_init: str = AUDIO_INIT_TEXT_SCALE,
) -> tuple[object, object, TokenMetadata]:
    # Expand the tokenizer vocab with audio tokens and resize model embeddings.
    old_vocab_size = len(tokenizer)
    # Preserve chat template — add_special_tokens can clear it in some transformers versions
    chat_template = getattr(tokenizer, "chat_template", None)
    tokenizer, meta = expand_tokenizer(tokenizer, codec, num_codebooks, codebook_size)
    if chat_template:
        tokenizer.chat_template = chat_template
    resize_model_embeddings(model, tokenizer)

    with torch.no_grad():
        input_weight = model.get_input_embeddings().weight
        output_embeddings = model.get_output_embeddings()
        _init_new_rows(input_weight, old_vocab_size, audio_init)
        if output_embeddings is not None and output_embeddings.weight.data_ptr() != input_weight.data_ptr():
            _init_new_rows(output_embeddings.weight, old_vocab_size, audio_init)
    config = getattr(model, "config", None)
    if config is not None:
        policy = dict(getattr(config, "audiogen", None) or {})
        policy["audio_init"] = audio_init
        config.audiogen = policy

    if save_dir is not None:
        save_dir = Path(save_dir)
        tokenizer.save_pretrained(save_dir)
        save_token_metadata(meta, save_dir / "audio_token_metadata.json")

    return model, tokenizer, meta


def _validate_audio_token_ids(tokenizer, meta: TokenMetadata) -> None:
    """Check every code and delimiter, not just range endpoints or vocabulary size."""
    if meta.codec == "moss":
        codes = build_moss_vocab(meta.num_codebooks, meta.codebook_size)
    elif meta.codec == "snac":
        codes = build_snac_vocab(meta.num_codebooks, meta.codebook_size)
    else:
        raise ValueError(f"Unknown audio codec: {meta.codec!r}")
    if meta.num_codebooks <= 0 or meta.codebook_size <= 0 or meta.audio_token_start_id < 0:
        raise ValueError("Invalid audio token ID range")
    expected = {token: meta.audio_token_start_id + i for i, token in enumerate(codes)}
    expected.update({"<audio>": meta.audio_start_token_id, "</audio>": meta.audio_end_token_id})
    if meta.codec == "snac":
        scales = set(range(meta.num_codebooks))
        if (meta.tokens_per_scale != meta.codebook_size or
                set(meta.scale_open_ids or {}) != scales or set(meta.scale_close_ids or {}) != scales):
            raise ValueError("Invalid SNAC scale token ID metadata")
        for scale in scales:
            expected[f"<s{scale}>"] = meta.scale_open_ids[scale]
            expected[f"</s{scale}>"] = meta.scale_close_ids[scale]
    # Existing metadata's end is exclusive of the whole added vocabulary, including delimiters.
    ids = list(expected.values())
    if (len(ids) != len(set(ids)) or meta.audio_token_end_id != max(ids) + 1 or
            set(ids) != set(range(meta.audio_token_start_id, meta.audio_token_end_id)) or
            meta.audio_token_end_id > len(tokenizer)):
        raise ValueError("Inconsistent audio token ID range or duplicate token IDs")
    vocab = tokenizer.get_vocab()
    for token, expected_id in expected.items():
        if vocab.get(token) != expected_id or tokenizer.encode(token, add_special_tokens=False) != [expected_id]:
            raise ValueError(f"Audio token ID mismatch for {token!r}: expected {expected_id}, got {vocab.get(token)!r}")
    if not tokenizer.encode("hello", add_special_tokens=False):
        raise ValueError("Saved tokenizer encodes ordinary text to empty")


def load_trained_model(
    model_path: str | Path, base_model: str | None = None, device: str = "cpu", *,
    dtype="auto", revision: str | None = None,
):
    """Prefer the checkpoint tokenizer; reconstruct only from an identified source.

    New checkpoints store source and prompt policy in ``config.audiogen``. Native
    reconstruction requires a revision. Old Qwen checkpoints may use their exact
    recorded Qwen source without a revision; there is never a default Qwen source.
    ``dtype='auto'`` retains the checkpoint dtype, including BF16.
    """
    model_path = Path(model_path)
    meta = load_token_metadata(model_path / "audio_token_metadata.json")
    cfg = json.loads((model_path / "config.json").read_text())
    policy = cfg.get("audiogen")
    if policy is not None:
        if not isinstance(policy, dict) or policy.get("prompt_format") not in ("legacy", "speech_v1"):
            raise ValueError("Invalid audiogen prompt_format policy")
        if (policy["prompt_format"] == "speech_v1" and
                policy.get("chat_template_kwargs") != {"enable_thinking": False}):
            raise ValueError("speech_v1 requires chat_template_kwargs={'enable_thinking': False}")
        source = policy.get("base_model")
        source_revision = policy.get("revision")
        if base_model is not None and base_model != source:
            raise ValueError("base_model override conflicts with checkpoint source")
        if revision is not None and revision != source_revision:
            raise ValueError("revision override conflicts with checkpoint source revision")
    else:
        source = base_model or cfg.get("_name_or_path")
        source_revision = revision
    if ("nemotron" in cfg.get("model_type", "").lower() and
            isinstance(source, str) and source.lower().startswith("qwen/")):
        raise ValueError("A Nemotron checkpoint cannot use a Qwen tokenizer source")
    known_legacy_qwen = (
        policy is None and cfg.get("model_type") in ("qwen2", "qwen3") and
        isinstance(source, str) and source.startswith("Qwen/Qwen")
    )
    if policy is None and not known_legacy_qwen:
        raise ValueError(
            "Checkpoint requires an audiogen prompt policy; only identified legacy "
            "Qwen checkpoints with an explicit Qwen source may omit it"
        )
    has_tokenizer = any((model_path / name).exists() for name in (
        "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "vocab.json",
    ))
    tokenizer = None
    if has_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            if not tokenizer.encode("hello", add_special_tokens=False):
                raise ValueError("Saved tokenizer encodes ordinary text to empty")
        except (OSError, ValueError):
            if not known_legacy_qwen:
                raise
            logger.warning("Broken legacy Qwen tokenizer; rebuilding from recorded source %s", source)
            tokenizer = None
    if tokenizer is None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Checkpoint has no saved tokenizer or explicit base_model source")
        if not known_legacy_qwen and not source_revision:
            raise ValueError("Tokenizer reconstruction requires an explicit base_model source and pinned revision")
        tokenizer = AutoTokenizer.from_pretrained(source, revision=source_revision)
        tokenizer, rebuilt_meta = expand_tokenizer(tokenizer, meta.codec, meta.num_codebooks, meta.codebook_size)
        if rebuilt_meta != meta:
            raise ValueError("Reconstructed tokenizer token IDs do not match checkpoint metadata")
    _validate_audio_token_ids(tokenizer, meta)
    model = _load_model(model_path, dtype=dtype)
    for embedding in (model.get_input_embeddings(), model.get_output_embeddings()):
        if embedding is not None and embedding.weight.shape[0] < len(tokenizer):
            raise ValueError("Checkpoint embedding vocabulary is smaller than its tokenizer")
    if policy is None:
        model.config.audiogen = {
            "base_model": source, "revision": source_revision,
            "prompt_format": "legacy", "chat_template_kwargs": {},
        }
    model = model.to(device).eval()
    return model, tokenizer, meta


def load_lora_model(base_model_path: str, lora_path: str | Path, device: str = "cpu"):
    lora_path = Path(lora_path)
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    model = model.to(device).eval()

    meta = load_token_metadata(lora_path / "audio_token_metadata.json")
    return model, tokenizer, meta
