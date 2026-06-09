#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
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
if [[ "${WANDB_MODE}" == "disabled" ]]; then
  export WANDB_DISABLED="${WANDB_DISABLED:-true}"
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT="${TORCH_DISTRIBUTED_DEFAULT_TIMEOUT:-3600}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1836}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

unset CUDA_HOME CUDA_VISIBLE_DEVICES CUDA_PATH
unset NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_BLOCKING_WAIT NCCL_ASYNC_ERROR_HANDLING NCCL_TIMEOUT NCCL_SOCKET_TIMEOUT_MS

CONFIG_YAML="${CONFIG_YAML:-examples/LIBERO/train_files/starvla_train_libero_qwengroot_ascend.yaml}"
BASE_VLM="${BASE_VLM:-/home/hfang/public_datasets/starVLA/Pretrained_models/Qwen3-VL-4B-Instruct}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/home/hfang/public_datasets/starVLA/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_all}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/home/hfang/output/VLA_Ascend}"
RUN_ID="${RUN_ID:-libero_qwengroot_ascend}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-true}"

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"

LAUNCH_ARGS=()
if [[ -n "${MAIN_PROCESS_PORT:-${MASTER_PORT:-}}" ]]; then
  LAUNCH_ARGS+=(--main_process_port "${MAIN_PROCESS_PORT:-${MASTER_PORT}}")
fi

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" \
  "${LAUNCH_ARGS[@]}" \
  -m \
  starVLA.training.train_starvla \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name QwenGR00T \
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
