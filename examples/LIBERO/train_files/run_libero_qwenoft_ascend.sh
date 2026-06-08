#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hfang/code/FangHeng:starVLA}"
cd "${REPO_ROOT}"

DEFAULT_ASCEND_SET_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"
if [[ -f "/usr/local/Ascend/cann-8.5.2/set_env.sh" ]]; then
  DEFAULT_ASCEND_SET_ENV="/usr/local/Ascend/cann-8.5.2/set_env.sh"
fi

ASCEND_ENV="${ASCEND_SET_ENV:-${DEFAULT_ASCEND_SET_ENV}}"
if [[ -f "${ASCEND_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_ENV}"
fi

DEFAULT_PYTHON_BIN=""
if [[ -n "${CONDA_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    DEFAULT_PYTHON_BIN="$(command -v python)"
  else
    echo "CONDA_ENV is set to '${CONDA_ENV}', but conda was not found in PATH." >&2
    exit 1
  fi
fi

if [[ -z "${DEFAULT_PYTHON_BIN}" ]]; then
  if [[ -x "/mnt/hfang/miniconda3/envs/vla_ascend_py310/bin/python" ]]; then
    DEFAULT_PYTHON_BIN="/mnt/hfang/miniconda3/envs/vla_ascend_py310/bin/python"
  else
    DEFAULT_PYTHON_BIN="$(command -v python3 || true)"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No Python interpreter found. Set PYTHON_BIN or activate an Ascend Python environment." >&2
  exit 1
fi

export ACCELERATE_USE_DEEPSPEED="${ACCELERATE_USE_DEEPSPEED:-1}"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${ACCELERATE_GRADIENT_ACCUMULATION_STEPS:-1}"
export ASCEND_PATCH_DEEPSPEED_GRAD_NORM="${ASCEND_PATCH_DEEPSPEED_GRAD_NORM:-1}"
export ASCEND_PATCH_DS_ZERO2_REDUCE_SCATTER="${ASCEND_PATCH_DS_ZERO2_REDUCE_SCATTER:-0}"
export ASCEND_PATCH_QWEN_RMSNORM="${ASCEND_PATCH_QWEN_RMSNORM:-1}"
export ASCEND_PATCH_QWEN_PLACEHOLDER_MASK="${ASCEND_PATCH_QWEN_PLACEHOLDER_MASK:-1}"
export ASCEND_PATCH_NPU_FA_VARLEN_CACHE="${ASCEND_PATCH_NPU_FA_VARLEN_CACHE:-1}"
export ASCEND_PATCH_QWEN_VISION_FLASH="${ASCEND_PATCH_QWEN_VISION_FLASH:-1}"
export ASCEND_PATCH_QWEN_VISION_LENGTHS="${ASCEND_PATCH_QWEN_VISION_LENGTHS:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG_YAML="${CONFIG_YAML:-examples/LIBERO/train_files/starvla_train_libero_qwenoft_ascend.yaml}"
BASE_VLM="${BASE_VLM:-/home/hfang/public_datasets/starVLA/Pretrained_models/Qwen3-VL-4B-Instruct}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/home/hfang/public_datasets/starVLA/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_all}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-80000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/home/hfang/output/VLA_Ascend}"
RUN_ID="${RUN_ID:-libero_qwenoft_ascend}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-false}"

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" \
  -m \
  starVLA.training.train_starvla \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --datasets.vla_data.prefetch_factor "${PREFETCH_FACTOR}" \
  --datasets.vla_data.persistent_workers "${PERSISTENT_WORKERS}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.skip_final_save "${SKIP_FINAL_SAVE}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
