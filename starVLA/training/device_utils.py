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


def _safe_tensor_version(tensor) -> Optional[int]:
    try:
        return getattr(tensor, "_version", None)
    except RuntimeError:
        return None


def _get_attention_interface(attention_functions, attn_impl: str, default=None):
    get_interface = getattr(attention_functions, "get_interface", None)
    if callable(get_interface):
        return get_interface(attn_impl, default)

    if attn_impl == "eager":
        return default

    return attention_functions[attn_impl]


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


def patch_qwen_rms_norm_for_ascend(
    torch_npu_module=None,
    class_modules=None,
    *,
    require_npu_device: bool = True,
) -> int:
    if torch_npu_module is None:
        try:
            import torch_npu as torch_npu_module
        except ImportError:
            return 0

    if not hasattr(torch_npu_module, "npu_rms_norm"):
        return 0

    if class_modules is None:
        class_modules = []
        for module_name, class_names in (
            ("transformers.models.qwen3.modeling_qwen3", ("Qwen3RMSNorm",)),
            ("transformers.models.qwen3_vl.modeling_qwen3_vl", ("Qwen3VLTextRMSNorm",)),
        ):
            try:
                module = __import__(module_name, fromlist=list(class_names))
            except ImportError:
                continue
            class_modules.append((module, class_names))

    patched_count = 0

    def _can_use_npu_rms_norm(hidden_states, weight):
        if require_npu_device and normalize_device_type(hidden_states.device) != "npu":
            return False
        if hidden_states.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            return False
        if weight.dtype != hidden_states.dtype:
            return False
        return True

    def _build_forward(original_forward):
        def _ascend_qwen_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            if not _can_use_npu_rms_norm(hidden_states, self.weight):
                return original_forward(self, hidden_states)
            return torch_npu_module.npu_rms_norm(
                hidden_states,
                self.weight,
                epsilon=float(self.variance_epsilon),
            )[0]

        return _ascend_qwen_rms_norm_forward

    for module, class_names in class_modules:
        for class_name in class_names:
            rms_norm_cls = getattr(module, class_name, None)
            if rms_norm_cls is None or getattr(rms_norm_cls, "_starvla_ascend_rms_norm_patched", False):
                continue

            rms_norm_cls._starvla_original_forward = rms_norm_cls.forward
            rms_norm_cls.forward = _build_forward(rms_norm_cls._starvla_original_forward)
            rms_norm_cls._starvla_ascend_rms_norm_patched = True
            patched_count += 1

    return patched_count


def patch_qwen3_vl_placeholder_mask_for_ascend(class_modules=None) -> int:
    if class_modules is None:
        class_modules = []
        for module_name, class_names in (
            ("transformers.models.qwen3_vl.modeling_qwen3_vl", ("Qwen3VLModel",)),
        ):
            try:
                module = __import__(module_name, fromlist=list(class_names))
            except ImportError:
                continue
            class_modules.append((module, class_names))

    patched_count = 0

    def _build_get_placeholder_mask(original_method):
        def _ascend_get_placeholder_mask(
            self,
            input_ids: torch.LongTensor,
            inputs_embeds: torch.FloatTensor,
            image_features: Optional[torch.FloatTensor] = None,
            video_features: Optional[torch.FloatTensor] = None,
        ):
            if input_ids is None or normalize_device_type(inputs_embeds.device) != "npu":
                return original_method(
                    self,
                    input_ids,
                    inputs_embeds,
                    image_features=image_features,
                    video_features=video_features,
                )

            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            hidden_size = inputs_embeds.shape[-1]

            if image_features is not None and image_features.shape[-1] != hidden_size:
                raise ValueError(
                    f"Image feature hidden size does not match text embeddings: "
                    f"image features {image_features.shape[-1]}, text embeddings {hidden_size}"
                )

            if video_features is not None and video_features.shape[-1] != hidden_size:
                raise ValueError(
                    f"Video feature hidden size does not match text embeddings: "
                    f"video features {video_features.shape[-1]}, text embeddings {hidden_size}"
                )

            return (
                special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device),
                special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device),
            )

        return _ascend_get_placeholder_mask

    for module, class_names in class_modules:
        for class_name in class_names:
            model_cls = getattr(module, class_name, None)
            if (
                model_cls is None
                or not hasattr(model_cls, "get_placeholder_mask")
                or getattr(model_cls, "_starvla_ascend_placeholder_mask_patched", False)
            ):
                continue

            model_cls._starvla_original_get_placeholder_mask = model_cls.get_placeholder_mask
            model_cls.get_placeholder_mask = _build_get_placeholder_mask(
                model_cls._starvla_original_get_placeholder_mask
            )
            model_cls._starvla_ascend_placeholder_mask_patched = True
            patched_count += 1

    return patched_count


def patch_npu_flash_attention_varlen_cache_for_ascend(
    npu_flash_attention_module=None,
    modeling_flash_attention_utils_module=None,
) -> bool:
    if npu_flash_attention_module is None:
        try:
            from transformers.integrations import npu_flash_attention as npu_flash_attention_module
        except ImportError:
            return False

    original_varlen = getattr(npu_flash_attention_module, "npu_flash_attn_varlen_func", None)
    if original_varlen is None or getattr(npu_flash_attention_module, "_starvla_varlen_cache_patched", False):
        return False
    if not hasattr(npu_flash_attention_module, "npu_fusion_attention"):
        return False

    import functools
    import math

    cache_attr = "_starvla_actual_seq_lens_cache"

    def _actual_seq_lens(cu_seqlens):
        version = _safe_tensor_version(cu_seqlens)
        cached = getattr(cu_seqlens, cache_attr, None)
        if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == version:
            return cached[1]

        actual_seq_lens = tuple(cu_seqlens[1:].detach().cpu().numpy().tolist())
        try:
            setattr(cu_seqlens, cache_attr, (version, actual_seq_lens))
        except Exception:
            pass
        return actual_seq_lens

    @functools.wraps(original_varlen)
    def _cached_npu_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        **kwargs,
    ):
        keep_prob = 1.0 - dropout_p

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        head_num = q.shape[1]
        actual_seq_qlen = _actual_seq_lens(cu_seqlens_q)
        actual_seq_kvlen = actual_seq_qlen if cu_seqlens_k is cu_seqlens_q else _actual_seq_lens(cu_seqlens_k)

        if not causal:
            return npu_flash_attention_module.npu_fusion_attention(
                q,
                k,
                v,
                head_num,
                pse=None,
                atten_mask=None,
                scale=softmax_scale,
                keep_prob=keep_prob,
                input_layout="TND",
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )[0]

        attn_mask_npu = npu_flash_attention_module.get_attn_mask_npu(q.device)
        return npu_flash_attention_module.npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            padding_mask=None,
            atten_mask=attn_mask_npu,
            scale=softmax_scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            sparse_mode=getattr(npu_flash_attention_module, "SPARSE_MODE", 3),
        )[0]

    npu_flash_attention_module._starvla_original_npu_flash_attn_varlen_func = original_varlen
    npu_flash_attention_module._starvla_actual_seq_lens_cache_attr = cache_attr
    npu_flash_attention_module.npu_flash_attn_varlen_func = _cached_npu_flash_attn_varlen_func
    npu_flash_attention_module._starvla_varlen_cache_patched = True

    if modeling_flash_attention_utils_module is None:
        try:
            import transformers.modeling_flash_attention_utils as modeling_flash_attention_utils_module
        except ImportError:
            modeling_flash_attention_utils_module = None

    if modeling_flash_attention_utils_module is not None:
        if getattr(modeling_flash_attention_utils_module, "_flash_varlen_fn", None) is original_varlen:
            modeling_flash_attention_utils_module._flash_varlen_fn = _cached_npu_flash_attn_varlen_func
            process_fn = getattr(modeling_flash_attention_utils_module, "_process_flash_kwargs_fn", None)
            lazy_define = getattr(modeling_flash_attention_utils_module, "_lazy_define_process_function", None)
            if process_fn is not None and lazy_define is not None:
                modeling_flash_attention_utils_module._process_flash_kwargs_fn = lazy_define(
                    _cached_npu_flash_attn_varlen_func
                )

    return True


def patch_qwen3_vl_vision_attention_lengths_for_ascend(class_modules=None) -> int:
    if class_modules is None:
        class_modules = []
        for module_name, class_names in (
            ("transformers.models.qwen3_vl.modeling_qwen3_vl", ("Qwen3VLVisionAttention",)),
        ):
            try:
                module = __import__(module_name, fromlist=list(class_names))
            except ImportError:
                continue
            class_modules.append((module, class_names))

    import functools

    patched_count = 0
    cache_attr = "_starvla_lengths_tuple_cache"

    def _lengths_tuple(cu_seqlens):
        version = _safe_tensor_version(cu_seqlens)
        cached = getattr(cu_seqlens, cache_attr, None)
        if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == version:
            return cached[1]

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        lengths_tuple = tuple(int(item) for item in lengths.detach().cpu().numpy().tolist())
        try:
            setattr(cu_seqlens, cache_attr, (version, lengths_tuple))
        except Exception:
            pass
        return lengths_tuple

    def _build_forward(module, original_forward):
        rotary_fn = getattr(module, "apply_rotary_pos_emb_vision", None)
        eager_attention_forward = getattr(module, "eager_attention_forward", None)
        attention_functions = getattr(module, "ALL_ATTENTION_FUNCTIONS", None)
        if rotary_fn is None or eager_attention_forward is None or attention_functions is None:
            return None

        @functools.wraps(original_forward)
        def _ascend_vision_attention_forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: Optional[torch.Tensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ) -> torch.Tensor:
            attn_impl = getattr(self.config, "_attn_implementation", "eager")
            if (
                normalize_device_type(hidden_states.device) != "npu"
                or attn_impl == "flash_attention_2"
                or position_embeddings is None
            ):
                return original_forward(
                    self,
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            seq_length = hidden_states.shape[0]
            query_states, key_states, value_states = (
                self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
            cos, sin = position_embeddings
            query_states, key_states = rotary_fn(query_states, key_states, cos, sin)

            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            attention_interface = eager_attention_forward
            if attn_impl != "eager":
                attention_interface = _get_attention_interface(
                    attention_functions,
                    attn_impl,
                    eager_attention_forward,
                )

            split_lengths = _lengths_tuple(cu_seqlens)
            splits = [
                torch.split(tensor, split_lengths, dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            attn_output = self.proj(attn_output)
            return attn_output

        return _ascend_vision_attention_forward

    for module, class_names in class_modules:
        for class_name in class_names:
            attention_cls = getattr(module, class_name, None)
            if (
                attention_cls is None
                or getattr(attention_cls, "_starvla_ascend_lengths_patched", False)
                or getattr(attention_cls, "_starvla_ascend_vision_flash_patched", False)
            ):
                continue

            patched_forward = _build_forward(module, attention_cls.forward)
            if patched_forward is None:
                continue

            attention_cls._starvla_original_forward = attention_cls.forward
            attention_cls.forward = patched_forward
            attention_cls._starvla_ascend_lengths_patched = True
            patched_count += 1

    return patched_count


def patch_qwen3_vl_vision_flash_attention_for_ascend(class_modules=None) -> int:
    if class_modules is None:
        class_modules = []
        for module_name, class_names in (
            ("transformers.models.qwen3_vl.modeling_qwen3_vl", ("Qwen3VLVisionAttention",)),
        ):
            try:
                module = __import__(module_name, fromlist=list(class_names))
            except ImportError:
                continue
            class_modules.append((module, class_names))

    import functools

    patched_count = 0

    def _build_forward(module, original_forward):
        rotary_fn = getattr(module, "apply_rotary_pos_emb_vision", None)
        attention_functions = getattr(module, "ALL_ATTENTION_FUNCTIONS", None)
        if rotary_fn is None or attention_functions is None:
            return None

        try:
            flash_attention_forward = _get_attention_interface(attention_functions, "flash_attention_2")
        except (KeyError, TypeError):
            return None
        if flash_attention_forward is None:
            return None

        @functools.wraps(original_forward)
        def _ascend_vision_flash_forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: Optional[torch.Tensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ) -> torch.Tensor:
            if (
                normalize_device_type(hidden_states.device) != "npu"
                or getattr(self.config, "_attn_implementation", "eager") == "flash_attention_2"
                or position_embeddings is None
            ):
                return original_forward(
                    self,
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            seq_length = hidden_states.shape[0]
            query_states, key_states, value_states = (
                self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
            cos, sin = position_embeddings
            query_states, key_states = rotary_fn(query_states, key_states, cos, sin)

            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = flash_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )

            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            attn_output = self.proj(attn_output)
            return attn_output

        return _ascend_vision_flash_forward

    for module, class_names in class_modules:
        for class_name in class_names:
            attention_cls = getattr(module, class_name, None)
            if (
                attention_cls is None
                or getattr(attention_cls, "_starvla_ascend_vision_flash_patched", False)
                or getattr(attention_cls, "_starvla_ascend_lengths_patched", False)
            ):
                continue

            patched_forward = _build_forward(module, attention_cls.forward)
            if patched_forward is None:
                continue

            attention_cls._starvla_original_forward = attention_cls.forward
            attention_cls.forward = patched_forward
            attention_cls._starvla_ascend_vision_flash_patched = True
            patched_count += 1

    return patched_count


def patch_deepspeed_zero2_reduce_scatter_for_ascend(zero_stage_module=None) -> bool:
    if zero_stage_module is None:
        try:
            from deepspeed.runtime.zero import stage_1_and_2 as zero_stage
        except ImportError:
            return False
    else:
        zero_stage = zero_stage_module

    optimizer_cls = getattr(zero_stage, "DeepSpeedZeroOptimizer", None)
    if optimizer_cls is None or not hasattr(optimizer_cls, "allreduce_and_scatter"):
        return False
    if getattr(optimizer_cls, "_starvla_ascend_zero2_reduce_scatter_patched", False):
        return False

    dist_module = getattr(zero_stage, "dist", None)
    if dist_module is None or not hasattr(dist_module, "reduce_scatter_tensor"):
        return False

    pg_correctness_test = getattr(zero_stage, "pg_correctness_test", False)
    original_allreduce_and_scatter = optimizer_cls.allreduce_and_scatter

    try:
        max_pad_ratio = float(os.environ.get("ASCEND_DS_ZERO2_RS_MAX_PAD_RATIO", "1.5"))
    except ValueError:
        max_pad_ratio = 1.5
    debug_stats = env_flag_enabled("ASCEND_DS_ZERO2_RS_DEBUG", "0")

    def _record_reduce_scatter_stats(self, used_reduce_scatter, pad_ratio=None):
        stats = getattr(self, "_starvla_zero2_rs_stats", None)
        if stats is None:
            stats = {
                "reduce_scatter_calls": 0,
                "fallback_calls": 0,
                "max_pad_ratio": 0.0,
                "max_fallback_pad_ratio": 0.0,
            }
            self._starvla_zero2_rs_stats = stats

        if used_reduce_scatter:
            stats["reduce_scatter_calls"] += 1
            stats["max_pad_ratio"] = max(stats["max_pad_ratio"], float(pad_ratio or 0.0))
        else:
            stats["fallback_calls"] += 1
            stats["max_fallback_pad_ratio"] = max(stats["max_fallback_pad_ratio"], float(pad_ratio or 0.0))

        return stats

    def _should_try_reduce_scatter(self, bucket, process_group):
        if not bucket:
            return False
        if getattr(self, "ipg_bucket_has_moe_params", False):
            return False
        if getattr(self, "sequence_parallel_size", 1) != 1:
            return False
        if process_group is None:
            return False

        first_tensor = bucket[0][1]
        return normalize_device_type(first_tensor.device) == "npu"

    def _communication_dtype(self, tensor):
        if pg_correctness_test or getattr(self, "sequence_parallel_size", 1) > 1:
            return torch.float32
        return getattr(self, "communication_data_type", tensor.dtype)

    def _try_reduce_scatter_with_padding(self, small_bucket, process_group):
        if not _should_try_reduce_scatter(self, small_bucket, process_group):
            return False

        world_size = dist_module.get_world_size(group=process_group)
        rank = dist_module.get_rank(group=process_group)
        if world_size <= 1:
            return False

        tensors_by_rank = [[] for _ in range(world_size)]
        for dst, tensor in small_bucket:
            dst = int(dst)
            if dst < 0 or dst >= world_size:
                return False
            tensors_by_rank[dst].append(tensor)

        chunk_sizes = [sum(tensor.numel() for tensor in tensors) for tensors in tensors_by_rank]
        total_numel = sum(chunk_sizes)
        if total_numel <= 0:
            return True

        max_chunk_size = max(chunk_sizes)
        pad_ratio = (max_chunk_size * world_size) / float(total_numel)
        if max_pad_ratio > 0.0 and pad_ratio > max_pad_ratio:
            stats = _record_reduce_scatter_stats(self, False, pad_ratio=pad_ratio)
            if debug_stats and dist_module.get_rank(group=process_group) == 0 and stats["fallback_calls"] <= 5:
                print(
                    "[starvla] ZeRO2 padded reduce_scatter fallback "
                    f"call={stats['fallback_calls']} pad_ratio={pad_ratio:.3f} "
                    f"limit={max_pad_ratio:.3f} chunk_sizes={chunk_sizes}"
                )
            return False

        dtype = _communication_dtype(self, small_bucket[0][1])
        device = small_bucket[0][1].device
        chunks = []
        for tensors, chunk_size in zip(tensors_by_rank, chunk_sizes):
            if tensors:
                chunk = self.flatten(tensors)
                if chunk.dtype != dtype:
                    chunk = chunk.to(dtype)
            else:
                chunk = torch.empty(0, dtype=dtype, device=device)

            if chunk_size < max_chunk_size:
                padding = torch.zeros(max_chunk_size - chunk_size, dtype=dtype, device=device)
                chunk = torch.cat((chunk, padding), dim=0)
            chunks.append(chunk)

        input_tensor = torch.cat(chunks, dim=0)
        output_tensor = torch.empty(max_chunk_size, dtype=dtype, device=device)
        dist_module.reduce_scatter_tensor(output_tensor, input_tensor, group=process_group)

        local_tensors = tensors_by_rank[rank]
        local_numel = chunk_sizes[rank]
        if local_tensors and local_numel > 0:
            local_output = output_tensor.narrow(0, 0, local_numel)
            for buf, synced in zip(local_tensors, self.unflatten(local_output, local_tensors)):
                buf.copy_(synced)

        stats = _record_reduce_scatter_stats(self, True, pad_ratio=pad_ratio)
        if debug_stats and dist_module.get_rank(group=process_group) == 0 and stats["reduce_scatter_calls"] <= 5:
            print(
                "[starvla] ZeRO2 padded reduce_scatter "
                f"call={stats['reduce_scatter_calls']} pad_ratio={pad_ratio:.3f} "
                f"chunk_sizes={chunk_sizes}"
            )
        return True

    def _ascend_allreduce_and_scatter(self, bucket, numel_per_bucket=500000000, log=None, divide=True, process_group=None):
        process_group = self.dp_process_group if process_group is None else process_group
        if divide:
            return original_allreduce_and_scatter(
                self,
                bucket,
                numel_per_bucket=numel_per_bucket,
                log=log,
                divide=divide,
                process_group=process_group,
            )

        small_bucket = []
        small_bucket_ranks = []
        numel = 0

        for bucket_rank, tensor in bucket:
            small_bucket.append((bucket_rank, tensor))
            small_bucket_ranks.append(bucket_rank)
            numel += tensor.numel()
            if numel > numel_per_bucket:
                if not _try_reduce_scatter_with_padding(self, small_bucket, process_group):
                    self.allreduce_and_copy_with_multiple_ranks(
                        [tensor for _, tensor in small_bucket],
                        log=None,
                        divide=divide,
                        process_group=process_group,
                        bucket_ranks=small_bucket_ranks,
                    )
                small_bucket = []
                small_bucket_ranks = []
                numel = 0

        if small_bucket:
            if not _try_reduce_scatter_with_padding(self, small_bucket, process_group):
                self.allreduce_and_copy_with_multiple_ranks(
                    [tensor for _, tensor in small_bucket],
                    log=log,
                    divide=divide,
                    process_group=process_group,
                    bucket_ranks=small_bucket_ranks,
                )

    optimizer_cls._starvla_original_allreduce_and_scatter = original_allreduce_and_scatter
    optimizer_cls.allreduce_and_scatter = _ascend_allreduce_and_scatter
    optimizer_cls._starvla_ascend_zero2_reduce_scatter_patched = True
    optimizer_cls._starvla_ascend_zero2_reduce_scatter_max_pad_ratio = max_pad_ratio
    return True


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
