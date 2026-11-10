"""Narrow, instance-local compatibility for validated native model omissions."""
import logging

import transformers

logger = logging.getLogger(__name__)


def prepare_nemotron_gradient_checkpointing(model) -> bool:
    """Prepare only native Transformers 5.6.2 NemotronH for Trainer's public API.

    Native blocks already implement checkpointing. Upstream commit
    3713bd839e580d07e4b70f2c89e986cb3c0e8ddf declares the missing support flag.
    The backbone flag additionally lets merge_with_config_defaults suppress
    cache allocation during checkpointed training. No methods/classes are
    replaced, and this function does not enable checkpointing itself.

    Other versions/architectures and already-supported models remain untouched;
    unsupported models still fail through the public API rather than silently
    disabling checkpointing. Revalidate before broadening these guards.
    """
    if (transformers.__version__ != "5.6.2" or
            getattr(getattr(model, "config", None), "model_type", None) != "nemotron_h" or
            getattr(model, "supports_gradient_checkpointing", False)):
        return False

    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHBlock, NemotronHForCausalLM, NemotronHModel,
    )

    # Exact native types deliberately exclude remote code, wrappers and subclasses.
    if type(model) is not NemotronHForCausalLM:
        return False
    backbone = model.model
    if (type(backbone) is not NemotronHModel or not len(backbone.layers) or
            len(backbone.layers) != model.config.num_hidden_layers or
            not all(type(block) is NemotronHBlock and
                    isinstance(block, GradientCheckpointingLayer)
                    for block in backbone.layers)):
        raise ValueError("NemotronH native gradient checkpointing compatibility requires all native checkpoint blocks")

    policy = dict(getattr(model.config, "audiogen", None) or {})
    policy["gradient_checkpointing_compat"] = {
        "id": "nemotron_h_transformers_5_6_2_native_gc_v1",
        "transformers_version": transformers.__version__,
        "native_block_count": len(backbone.layers),
        "scope": "instance_only",
        "backbone_cache_guard": True,
    }
    model.supports_gradient_checkpointing = True
    # Trainer's standard recursive setter will toggle this and every native block.
    # Do not reset it if setup is called on an already active instance.
    if not hasattr(backbone, "gradient_checkpointing"):
        backbone.gradient_checkpointing = False
    model.config.audiogen = policy
    logger.warning("Applied native NemotronH checkpointing compatibility: %s", policy["gradient_checkpointing_compat"])
    return True
