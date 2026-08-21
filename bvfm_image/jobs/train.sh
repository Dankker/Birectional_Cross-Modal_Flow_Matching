#!/bin/bash
#SBATCH --job-name=bvfm_image_train
#SBATCH --account=MST114566
#SBATCH --partition=normal2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=36:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BVFM_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/bvfm_image_xl.py}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/train}"
COCO_ROOT="${COCO_ROOT:-${REPO_ROOT}/data/coco}"
JOB_TMP="${BVFM_JOB_TMP_ROOT:-/tmp/${USER}}/bvfm_image_train_${SLURM_JOB_ID}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export BVFM_OUTPUT_DIR="${OUT_DIR}"
export COCO_ROOT
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${JOB_TMP}" \
  "${REPO_ROOT}/logs" "${OUT_DIR}"

module purge
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate biflow_fix2
if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

required_files=(
  "${REPO_ROOT}/assets/FlowTok-XL.pth"
  "${REPO_ROOT}/assets/FlowTiTok_512.bin"
  "${REPO_ROOT}/assets/decoder_init.pt"
  "${COCO_ROOT}/train2017"
  "${COCO_ROOT}/annotations/captions_train2017.json"
)
for required in "${required_files[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[FAIL] missing required path: ${required}" >&2
    exit 1
  fi
done

cd "${REPO_ROOT}"
python -m py_compile \
  bvfm_image/common.py bvfm_image/model_factory.py \
  bvfm_image/runtime.py bvfm_image/training.py \
  scripts/train.py configs/bvfm_image_xl.py \
  libs/model/flowtok_t2i.py libs/model/bvfm_variational.py

NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}}"
EXTRA_ARGS=()
if [[ -n "${BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--batch-size-per-gpu "${BATCH_SIZE}")
fi
if [[ -n "${N_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--n-steps "${N_STEPS}")
fi
if [[ "${RESUME:-auto}" == "none" ]]; then
  EXTRA_ARGS+=(--resume none)
fi

python -m torch.distributed.run \
  --standalone --nproc_per_node="${NGPU}" \
  scripts/train.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"
