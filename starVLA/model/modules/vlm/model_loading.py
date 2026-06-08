"""VLM loading helpers shared by Qwen wrappers."""

from __future__ import annotations

import os
from typing import Any

import torch


def resolve_torch_dtype(value: Any):
    """Normalize config dtype values for Hugging Face model loading."""
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "torch.float16"}:
        return torch.float16
    if normalized in {"float32", "fp32", "torch.float32"}:
        return torch.float32

    raise ValueError(f"Unsupported torch dtype value: {value!r}")


def resolve_model_id_and_load_kwargs(model_id: str) -> tuple[str, dict[str, bool]]:
    """Resolve local model paths without allowing accidental hub downloads."""
    raw_model_id = str(model_id).strip()
    expanded_model_id = os.path.expanduser(raw_model_id)
    is_local_path = (
        os.path.isabs(expanded_model_id)
        or expanded_model_id.startswith("./")
        or expanded_model_id.startswith("../")
        or os.path.lexists(expanded_model_id)
    )
    if is_local_path:
        return expanded_model_id, {"local_files_only": True}
    return raw_model_id, {}


def call_qwen_backbone_without_lm_head(model, kwargs: dict[str, Any]):
    """Call the Qwen backbone directly when only hidden states are needed."""
    model_kwargs = dict(kwargs)
    model_kwargs.pop("labels", None)
    model_kwargs.pop("logits_to_keep", None)

    backbone = getattr(model, "model", None)
    if callable(backbone):
        return backbone(**model_kwargs)

    base_model = getattr(model, "base_model", None)
    if callable(base_model) and base_model is not model:
        return base_model(**model_kwargs)

    raise RuntimeError(
        "skip_lm_head=True was requested, but the loaded Qwen class does not expose "
        "a callable `.model` or `.base_model` backbone. Refusing to fall back to the "
        "causal-LM forward because that would silently execute the LM head."
    )
