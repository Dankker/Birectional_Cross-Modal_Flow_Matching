#!/bin/bash
#SBATCH --job-name=bvfm_image_t2i
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BVFM_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${BVFM_REPO_ROOT}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ -n "${BVFM_WEIGHTS_ROOT:-}" ]]; then
  DEFAULT_CKPT="${BVFM_WEIGHTS_ROOT}/image/bvfm_image_step40000.pt"
else
  DEFAULT_CKPT="${REPO_ROOT}/checkpoints/bvfm_image_step40000.pt"
fi
CKPT="${CKPT:-${DEFAULT_CKPT}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/infer_t2i}"
PROMPT="${PROMPT:-A corgi wearing sunglasses on a tropical beach at sunset}"
ZV_MODE="${ZV_MODE:-mean}"
SAMPLES="${SAMPLES:-1}"
TEXT_TEMPERATURE="${TEXT_TEMPERATURE:-1.0}"
ZV_TEMPERATURE="${ZV_TEMPERATURE:-1.0}"
JOB_TMP="${BVFM_JOB_TMP_ROOT:-/tmp/${USER}}/bvfm_image_t2i_${SLURM_JOB_ID}"

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
ACTIVE_VENV="${VENV_PATH:-/work/dankker0900/flowtok_venv}"
if [[ -d "${ACTIVE_VENV}" ]]; then
  source "${ACTIVE_VENV}/bin/activate"
fi

cd "${REPO_ROOT}"
python scripts/infer_t2i.py \
  --checkpoint "${CKPT}" \
  --prompt "${PROMPT}" \
  --output-dir "${OUT_DIR}" \
  --samples-per-prompt "${SAMPLES}" \
  --zv-mode "${ZV_MODE}" \
  --zv-temperature "${ZV_TEMPERATURE}" \
  --text-temperature "${TEXT_TEMPERATURE}"
