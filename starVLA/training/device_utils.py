from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Optional

import torch


_FALSE_ENV_VALUES = {"0", "false", "no", "off", ""}


def env_flag_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in _FALSE_ENV_VALUES


def normalize_device_type(device: Optional[object] = None) -> str:
    """Return the torch device/autocast type for a device-like object."""
    if device is None:
        npu = getattr(torch, "npu", None)
        if npu is not None:
            try:
                if npu.is_available():
                    return "npu"
            except Exception:
                pass
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    if isinstance(device, torch.device):
        return device.type

    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type

    return str(device).split(":", 1)[0]


def get_autocast_context(device: Optional[object] = None, dtype: Optional[torch.dtype] = torch.bfloat16):
    """Create an autocast context for CUDA/NPU while preserving float32 regions."""
    device_type = normalize_device_type(device)
    if dtype is None or device_type == "cpu":
        return nullcontext()
    if dtype == torch.float32:
        return torch.autocast(device_type=device_type, enabled=False)
    return torch.autocast(device_type=device_type, dtype=dtype)


def raise_for_non_finite_loss(loss: torch.Tensor, step: Optional[int] = None) -> None:
    if torch.isfinite(loss.detach()).all():
        return

    step_label = "unknown" if step is None else step
    detached_loss = loss.detach().float()
    loss_value = detached_loss.item() if loss.numel() == 1 else detached_loss
    raise FloatingPointError(f"Non-finite training loss at step {step_label}: {loss_value}")


def should_check_non_finite_loss(
    step: Optional[int],
    interval: Optional[int] = 1,
    warmup_steps: Optional[int] = 0,
) -> bool:
    if interval is None:
        return True

    interval = int(interval)
    if interval <= 1:
        return True

    step = 0 if step is None else int(step)
    warmup_steps = 0 if warmup_steps is None else int(warmup_steps)
    if step < warmup_steps:
        return True

    return step % interval == 0


def should_run_step_interval_event(step: Optional[int], interval: Optional[int]) -> bool:
    if step is None or interval is None:
        return False

    step = int(step)
    interval = int(interval)
    if step <= 0 or interval <= 0:
        return False

    return step % interval == 0


def _zero_tensor_for_optimizer(optimizer: Any) -> torch.Tensor:
    return torch.tensor(0.0, dtype=torch.float32).to(getattr(optimizer, "device", "cpu"))


def patch_deepspeed_zero_grad_norm_for_ascend(zero_stage_module=None) -> bool:
    if zero_stage_module is None:
        try:
            from deepspeed.runtime.zero import stage_1_and_2 as zero_stage
        except ImportError:
            return False
    else:
        zero_stage = zero_stage_module

    optimizer_cls = getattr(zero_stage, "DeepSpeedZeroOptimizer", None)
    if optimizer_cls is None or not hasattr(optimizer_cls, "get_grad_norm_direct"):
        return False

    pipe_replicated_attr = getattr(zero_stage, "PIPE_REPLICATED", None)
    is_model_parallel_parameter = getattr(zero_stage, "is_model_parallel_parameter", lambda param: False)
    dist_module = getattr(zero_stage, "dist", None)
    if dist_module is None:
        return False

    patched = False

    def _should_count_param(optimizer, param):
        if pipe_replicated_attr and hasattr(param, pipe_replicated_attr):
            if getattr(param, pipe_replicated_attr):
                return False
        return is_model_parallel_parameter(param) or getattr(optimizer, "model_parallel_rank", 0) == 0

    def _ascend_get_grad_norm_direct(self, gradients, params, norm_type=2):
        norm_type = float(norm_type)
        if norm_type == float("inf"):
            local_max = None
            for grad, param in zip(gradients, params):
                if not _should_count_param(self, param):
                    continue
                value = grad.detach().float().abs().max()
                local_max = value if local_max is None else torch.maximum(local_max, value)
            total_norm = local_max if local_max is not None else _zero_tensor_for_optimizer(self)
            dist_module.all_reduce(total_norm, op=dist_module.ReduceOp.MAX, group=self.dp_process_group)
            self._model_parallel_all_reduce(tensor=total_norm, op=dist_module.ReduceOp.MAX)
            return total_norm

        total_sq_or_power = None
        for grad, param in zip(gradients, params):
            if not _should_count_param(self, param):
                continue
            grad_float = grad.detach().float()
            if norm_type == 2.0:
                local = (grad_float * grad_float).sum()
            else:
                local = grad_float.abs().pow(norm_type).sum()
            total_sq_or_power = local if total_sq_or_power is None else total_sq_or_power + local

        total_norm = total_sq_or_power if total_sq_or_power is not None else _zero_tensor_for_optimizer(self)
        dist_module.all_reduce(total_norm, op=dist_module.ReduceOp.SUM, group=self.dp_process_group)
        self._model_parallel_all_reduce(tensor=total_norm, op=dist_module.ReduceOp.SUM)
        return total_norm.pow(1.0 / norm_type)

    if not getattr(optimizer_cls, "_starvla_ascend_grad_norm_patched", False):
        optimizer_cls._starvla_original_get_grad_norm_direct = optimizer_cls.get_grad_norm_direct
        optimizer_cls.get_grad_norm_direct = _ascend_get_grad_norm_direct
        optimizer_cls._starvla_ascend_grad_norm_patched = True
        patched = True

    if hasattr(optimizer_cls, "scaled_global_norm") and not getattr(
        optimizer_cls, "_starvla_ascend_skip_no_clip_norm_patched", False
    ):
        optimizer_cls._starvla_original_scaled_global_norm = optimizer_cls.scaled_global_norm

        def _ascend_scaled_global_norm(self, norm_type=2):
            try:
                clip_grad = float(getattr(self, "clip_grad", 0.0) or 0.0)
            except (TypeError, ValueError):
                clip_grad = 0.0
            if clip_grad <= 0.0:
                return _zero_tensor_for_optimizer(self)
            return self._starvla_original_scaled_global_norm(norm_type=norm_type)

        optimizer_cls.scaled_global_norm = _ascend_scaled_global_norm
        optimizer_cls._starvla_ascend_skip_no_clip_norm_patched = True
        patched = True

    return patched


def patch_qwen_rms_norm_for_ascend(*args, **kwargs) -> int:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return 0
    return 0


def patch_qwen3_vl_placeholder_mask_for_ascend(*args, **kwargs) -> int:
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl  # noqa: F401
    except ImportError:
        return 0
    return 0


def patch_npu_flash_attention_varlen_cache_for_ascend(*args, **kwargs) -> bool:
    try:
        import transformers.integrations.npu_flash_attention  # noqa: F401
    except ImportError:
        return False
    return False


def patch_qwen3_vl_vision_attention_lengths_for_ascend(*args, **kwargs) -> int:
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl  # noqa: F401
    except ImportError:
        return 0
    return 0


def patch_qwen3_vl_vision_flash_attention_for_ascend(*args, **kwargs) -> int:
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl  # noqa: F401
    except ImportError:
        return 0
    return 0


def patch_deepspeed_zero2_reduce_scatter_for_ascend(*args, **kwargs) -> bool:
    return False


def _record_patch_result(results: dict[str, Any], name: str, patch_fn, logger=None) -> None:
    try:
        result = patch_fn()
    except ImportError:
        result = 0
    except Exception as exc:
        result = 0
        if logger is not None:
            logger.warning(f"Ascend patch `{name}` skipped after error: {exc}")

    results[name] = result
    if result and logger is not None:
        logger.info(f"Applied Ascend patch `{name}`: {result}")


def apply_ascend_patches_from_env(device: Optional[object] = None, logger=None) -> dict[str, Any]:
    """Apply optional Ascend patches gated by environment variables."""
    results: dict[str, Any] = {}
    if normalize_device_type(device) != "npu":
        return results

    if env_flag_enabled("ASCEND_PATCH_DEEPSPEED_GRAD_NORM", "1"):
        _record_patch_result(results, "deepspeed_zero_grad_norm", patch_deepspeed_zero_grad_norm_for_ascend, logger)
    if env_flag_enabled("ASCEND_PATCH_DS_ZERO2_REDUCE_SCATTER", "0"):
        _record_patch_result(results, "deepspeed_zero2_reduce_scatter", patch_deepspeed_zero2_reduce_scatter_for_ascend, logger)
    if env_flag_enabled("ASCEND_PATCH_QWEN_RMSNORM", "0"):
        _record_patch_result(results, "qwen_rms_norm", patch_qwen_rms_norm_for_ascend, logger)
    if env_flag_enabled("ASCEND_PATCH_QWEN_PLACEHOLDER_MASK", "0"):
        _record_patch_result(results, "qwen3_vl_placeholder_mask", patch_qwen3_vl_placeholder_mask_for_ascend, logger)
    if env_flag_enabled("ASCEND_PATCH_NPU_FA_VARLEN_CACHE", "0"):
        _record_patch_result(results, "npu_flash_attention_varlen_cache", patch_npu_flash_attention_varlen_cache_for_ascend, logger)
    if env_flag_enabled("ASCEND_PATCH_QWEN_VISION_LENGTHS", "0"):
        _record_patch_result(results, "qwen3_vl_vision_lengths", patch_qwen3_vl_vision_attention_lengths_for_ascend, logger)
    if env_flag_enabled("ASCEND_PATCH_QWEN_VISION_FLASH", "0"):
        _record_patch_result(results, "qwen3_vl_vision_flash", patch_qwen3_vl_vision_flash_attention_for_ascend, logger)

    return results


def build_adamw_optimizer(
    param_groups,
    name=None,
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.0,
    eps=1e-8,
    foreach=None,
    fused=None,
):
    optimizer_name = str(name or "adamw").lower().replace("-", "_")
    optimizer_kwargs = {
        "lr": lr,
        "betas": betas,
        "weight_decay": weight_decay,
        "eps": eps,
    }
    if optimizer_name in {"adamw", "torch_adamw"}:
        if foreach is not None:
            optimizer_kwargs["foreach"] = foreach
        if fused is not None:
            optimizer_kwargs["fused"] = fused
        return torch.optim.AdamW(param_groups, **optimizer_kwargs)

    if optimizer_name == "npu_fused_adamw":
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError("trainer.optimizer.name=npu_fused_adamw requires torch_npu") from exc
        return torch_npu.optim.NpuFusedAdamW(param_groups, **optimizer_kwargs)

    raise ValueError(f"Unsupported AdamW optimizer name: {name}")
