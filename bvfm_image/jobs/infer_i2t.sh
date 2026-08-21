#!/bin/bash
#SBATCH --job-name=bvfm_image_i2t
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
OUT_JSON="${OUT_JSON:-${REPO_ROOT}/runs/infer_i2t/captions.json}"
ZV_MODE="${ZV_MODE:-mean}"
ZV_TEMPERATURE="${ZV_TEMPERATURE:-1.0}"
JOB_TMP="${BVFM_JOB_TMP_ROOT:-/tmp/${USER}}/bvfm_image_i2t_${SLURM_JOB_ID}"

if [[ -z "${IMAGE_PATH:-}" && -z "${IMAGE_DIR:-}" ]]; then
  echo "[FAIL] set IMAGE_PATH=/path/image.png or IMAGE_DIR=/path/images" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${JOB_TMP}" \
  "${REPO_ROOT}/logs" "$(dirname "${OUT_JSON}")"

module purge
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate biflow_fix2
ACTIVE_VENV="${VENV_PATH:-/work/dankker0900/flowtok_venv}"
if [[ -d "${ACTIVE_VENV}" ]]; then
  source "${ACTIVE_VENV}/bin/activate"
fi

INPUT_ARGS=()
if [[ -n "${IMAGE_PATH:-}" ]]; then
  INPUT_ARGS+=(--image "${IMAGE_PATH}")
fi
if [[ -n "${IMAGE_DIR:-}" ]]; then
  INPUT_ARGS+=(--image-dir "${IMAGE_DIR}")
fi

cd "${REPO_ROOT}"
python scripts/infer_i2t.py \
  --checkpoint "${CKPT}" \
  "${INPUT_ARGS[@]}" \
  --output "${OUT_JSON}" \
  --zv-mode "${ZV_MODE}" \
  --zv-temperature "${ZV_TEMPERATURE}"
