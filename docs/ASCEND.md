# 昇腾训练配置指南

本文说明 StarVLA 在 Ascend NPU 上训练时需要关注的环境、启动方式和配置项。文档面向训练使用者和后续框架适配者，不绑定某一个 action head、backbone 或数据集。

仓库中提供了一个 LIBERO 训练示例，可作为 Ascend 配置模板。接入新的 framework 时，应复用本文的运行配置，再替换 framework 自身的模型、action head、loss 和数据字段。

## 环境依赖

推荐使用独立 Python 环境，并先加载 CANN 环境变量：

```bash
cd <starvla-root>

export PYTHON_BIN=/path/to/python
export ASCEND_SET_ENV=/path/to/Ascend/set_env.sh

source "${ASCEND_SET_ENV}"
"${PYTHON_BIN}" -m pip install -r requirements-ascend.txt
```

当前依赖版本参考：

| 组件 | 版本 |
| --- | --- |
| CANN | 8.5.2 |
| torch | 2.5.1 |
| torch-npu | 2.5.1.post1 |
| transformers | 4.57.0 |
| accelerate | 1.5.2 |
| deepspeed | 0.16.9 |

如果机器上的 CANN、torch-npu 或 transformers 版本不同，需要重新检查模型侧 patch 和 DeepSpeed 行为。

## 配置文件

Ascend 训练相关文件分为四类：

| 文件 | 用途 |
| --- | --- |
| `requirements-ascend.txt` | Ascend 环境依赖版本 |
| `starVLA/config/deepseeds/deepspeed_zero2.yaml` | Accelerate 多进程启动配置 |
| `starVLA/config/deepseeds/ds_config.yaml` | DeepSpeed ZeRO2 与 bf16 配置 |
| `examples/<dataset>/train_files/*_ascend.yaml` | 训练任务配置 |
| `examples/<dataset>/train_files/run_*_ascend.sh` | 训练启动脚本 |
| `scripts/ascend/summarize_train_log.py` | 训练日志摘要 |
| `scripts/ascend/summarize_npu_smi.py` | NPU 采样日志摘要 |

新增训练任务时，优先复用已有 `*_ascend.yaml` 和 `run_*_ascend.sh` 的 Ascend 部分，再替换数据集与 framework 字段。

## 启动方式

训练入口应通过模块方式启动：

```bash
"${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" \
  -m \
  starVLA.training.train_starvla \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}"
```

不建议依赖 `PYTHONPATH=<repo-root>` 暴露包路径。若仓库路径包含 `:` 等路径分隔符，`PYTHONPATH` 会被拆分，导致 `starVLA` 包无法导入。模块启动方式在不同路径下更稳定。

一个通用启动脚本通常需要设置以下变量：

```bash
export PYTHON_BIN=/path/to/python
export ASCEND_SET_ENV=/path/to/Ascend/set_env.sh
export CONFIG_YAML=examples/<dataset>/train_files/<config>_ascend.yaml
export FRAMEWORK_NAME=<FrameworkName>

export NUM_PROCESSES=8
export BASE_MODEL=/path/to/pretrained/model
export DATA_ROOT_DIR=/path/to/dataset
export DATA_MIX=<data_mix>
export RUN_ROOT_DIR=/path/to/output
export RUN_ID=<run_name>

export PER_DEVICE_BATCH_SIZE=16
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
export PERSISTENT_WORKERS=true
export MAX_TRAIN_STEPS=80000
export SAVE_INTERVAL=10000
export EVAL_INTERVAL=100
export LOGGING_FREQUENCY=100

source "${ASCEND_SET_ENV}"
```

具体脚本可以按任务命名，例如 `run_libero_<framework>_ascend.sh`。脚本内部只应负责设置运行环境和透传配置，不应把 framework 逻辑写死在 launcher 中。

## DeepSpeed 与精度

Ascend 推荐默认组合：

| 配置 | 推荐值 |
| --- | --- |
| 分布式后端 | HCCL |
| 进程数 | 8 |
| DeepSpeed | ZeRO2 |
| 精度 | bf16 |
| optimizer | Torch `AdamW` |
| gradient accumulation | `auto` 或由 launcher 覆盖 |

训练 YAML 中保持混合精度：

```yaml
trainer:
  enable_mixed_precision_training: true
```

DeepSpeed JSON 中使用 bf16：

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

模型侧如果有 attention backend 选项，Ascend 默认优先使用 `sdpa`，避免进入 CUDA FlashAttention 路径：

```yaml
framework:
  <model_config>:
    model_dtype: bfloat16
    attn_implementation: sdpa
```

非 Qwen framework 不一定使用同名字段，但应保持同样原则：bf16、避免 CUDA-only kernel、把不兼容算子替换为 NPU 可执行路径。

## Ascend Patch 开关

启动脚本应在训练前导出 Ascend patch 开关：

```bash
export ASCEND_PATCH_DEEPSPEED_GRAD_NORM=1
export ASCEND_PATCH_DS_ZERO2_REDUCE_SCATTER=0
export ASCEND_PATCH_QWEN_RMSNORM=1
export ASCEND_PATCH_QWEN_PLACEHOLDER_MASK=1
export ASCEND_PATCH_NPU_FA_VARLEN_CACHE=1
export ASCEND_PATCH_QWEN_VISION_FLASH=1
export ASCEND_PATCH_QWEN_VISION_LENGTHS=0
```

| 变量 | 作用 |
| --- | --- |
| `ASCEND_PATCH_DEEPSPEED_GRAD_NORM` | 修正 DeepSpeed ZeRO grad norm 在 NPU 上的兼容问题 |
| `ASCEND_PATCH_DS_ZERO2_REDUCE_SCATTER` | 预留 ZeRO2 reduce-scatter patch，默认关闭 |
| `ASCEND_PATCH_QWEN_RMSNORM` | Qwen RMSNorm 的 NPU 兼容 patch |
| `ASCEND_PATCH_QWEN_PLACEHOLDER_MASK` | Qwen3-VL placeholder mask 的 NPU 兼容 patch |
| `ASCEND_PATCH_NPU_FA_VARLEN_CACHE` | NPU varlen flash attention cache patch |
| `ASCEND_PATCH_QWEN_VISION_FLASH` | Qwen3-VL vision attention 的 NPU 兼容 patch |
| `ASCEND_PATCH_QWEN_VISION_LENGTHS` | Qwen vision lengths patch，默认关闭 |

这些 patch 由 `starVLA/training/device_utils.py` 统一管理。新增 framework 时，优先复用集中 patch；只有 framework 需要额外兼容时，再在对应模型 wrapper 中增加局部处理。

## DataLoader 与视频读取

Ascend 多进程训练对主机侧视频解码、worker 数和内存占用较敏感。训练配置中建议显式设置：

```yaml
datasets:
  vla_data:
    num_workers: 8
    prefetch_factor: 4
    persistent_workers: true
    video_backend: torchvision_av
    video_backend_kwargs:
      num_threads: 1
    video_gc_collect: false
```

资源紧张或定位问题时，可以从低并发开始：

```bash
export PER_DEVICE_BATCH_SIZE=1
export NUM_WORKERS=0
export PREFETCH_FACTOR=2
export PERSISTENT_WORKERS=false
```

如果 data time 明显高于 model time，应优先检查数据磁盘、视频线程数和 DataLoader worker 设置。

## 保存与日志

完整训练默认保存中间 checkpoint 和最终模型。最终模型保存会收集大模型权重，耗时和磁盘开销都较高。

```yaml
trainer:
  save_interval: 10000
  eval_interval: 100
  skip_final_save: false
```

不需要最终整模输出时，可以通过 launcher 覆盖：

```bash
export SKIP_FINAL_SAVE=true
```

W&B 由环境变量控制：

```bash
export WANDB_MODE=disabled
```

需要记录 W&B 时，移除该设置，并按实际项目配置 W&B 的 project、entity 和登录状态。

## 新增 Framework 的配置流程

新增 framework 时，建议按以下顺序处理：

1. 复制一个已有 Ascend YAML：

```bash
cp examples/<dataset>/train_files/<base>_ascend.yaml \
   examples/<dataset>/train_files/<new_framework>_ascend.yaml
```

2. 保留 Ascend 运行配置：

```yaml
trainer:
  enable_mixed_precision_training: true

datasets:
  vla_data:
    num_workers: 8
    prefetch_factor: 4
    persistent_workers: true
    video_backend: torchvision_av
```

3. 替换 framework 字段：

```yaml
framework:
  name: <FrameworkName>
```

并按新 framework 更新：

- backbone 或 VLM 路径
- action/state 维度
- action horizon
- projector/action head 参数
- loss scale
- freeze modules
- framework 自有 dtype 和 attention 配置

4. 复制或新建 launcher，保留：

- CANN `set_env.sh`
- Ascend patch 开关
- `python -m accelerate.commands.launch ... -m starVLA.training.train_starvla`
- batch、worker、save、logging 的环境变量覆盖

5. 对模型 wrapper 做最小兼容修改：

- 使用设备感知 autocast
- 避免硬编码 CUDA device 或 CUDA-only kernel
- 推理输出在返回 numpy 前显式 `.detach().float().cpu()`
- 对 NPU 不友好的 gather/scatter/conv backward 路径做等价替换

## 诊断

训练日志摘要：

```bash
python scripts/ascend/summarize_train_log.py /path/to/train.log
python scripts/ascend/summarize_train_log.py /path/to/train.log --format json
```

NPU 采样：

```bash
while true; do
  echo "=== $(date '+%F %T') ==="
  npu-smi info
  sleep 30
done | tee /path/to/npu-smi.log
```

NPU 采样摘要：

```bash
python scripts/ascend/summarize_npu_smi.py /path/to/npu-smi.log --loaded-hbm-threshold 10000
python scripts/ascend/summarize_npu_smi.py /path/to/npu-smi.log --format json
```

训练日志中应能看到：

```text
Distributed environment: DEEPSPEED  Backend: hccl
Num processes: 8
Device: npu:0
Mixed precision type: bf16
Step <n>, Loss:
```

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError: No module named 'starVLA'` | 确认 launcher 使用模块启动，不依赖 `PYTHONPATH=<repo-root>` |
| 进入 CUDA FlashAttention 路径 | 将 attention backend 改为 `sdpa`，或为该 framework 增加 NPU 兼容实现 |
| HBM 占用过高 | 降低 `PER_DEVICE_BATCH_SIZE`，减少 worker，关闭不必要的最终整模保存 |
| 训练结束保存很慢 | 调整 checkpoint 策略，或在不需要最终整模时设置 `SKIP_FINAL_SAVE=true` |
| data time 偏高 | 检查视频线程数、worker 数、数据盘性能和 `npu-smi` AICore 利用率 |

## 参考示例

LIBERO 目录下的 Ascend 配置展示了一套完整的启动方式：

```text
examples/LIBERO/train_files/starvla_train_libero_qwenoft_ascend.yaml
examples/LIBERO/train_files/run_libero_qwenoft_ascend.sh
```

该示例用于说明 Ascend 配置骨架。新增 GR00T、PI 或其他 framework 时，应复用运行层配置，并替换 framework 相关字段。
