#!/bin/bash
#SBATCH --job-name=bvfm_image_smoke
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BVFM_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUT_DIR="${REPO_ROOT}/runs/smoke_${SLURM_JOB_ID}"
JOB_TMP="${BVFM_JOB_TMP_ROOT:-/tmp/${USER}}/bvfm_image_smoke_${SLURM_JOB_ID}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${JOB_TMP}" \
  "${REPO_ROOT}/logs" "${OUT_DIR}"

module purge
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate biflow_fix2
if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

cd "${REPO_ROOT}"
python -m py_compile \
  bvfm_image/common.py bvfm_image/model_factory.py \
  bvfm_image/runtime.py bvfm_image/training.py \
  scripts/train.py scripts/infer_t2i.py scripts/infer_i2t.py \
  scripts/export_trajectories.py configs/bvfm_image_xl.py

python scripts/infer_t2i.py \
  --prompt "A corgi wearing sunglasses on a tropical beach" \
  --output-dir "${OUT_DIR}/t2i" \
  --zv-mode mean --samples-per-prompt 1

python scripts/infer_i2t.py \
  --image "${OUT_DIR}/t2i/prompt000_sample000.png" \
  --output "${OUT_DIR}/i2t.json" \
  --zv-mode mean

echo "[RESULT] smoke outputs=${OUT_DIR}"
