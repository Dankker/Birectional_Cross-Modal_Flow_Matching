#!/bin/bash
#SBATCH --job-name=bvfm_img_err
#SBATCH --account=MST114566
#SBATCH --partition=normal2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BVFM_IMAGE_ROOT:-}" ]]; then
  REPO_ROOT="${BVFM_IMAGE_ROOT}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
COCO_ROOT="${COCO_ROOT:-/work/dankker0900/dataset/coco}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/runs/transport_error/image_transport.npz}"
JOB_TMP="${BVFM_JOB_TMP_ROOT:-/tmp/${USER}}/bvfm_img_err_${SLURM_JOB_ID}"

export HF_HOME="${HF_HOME:-/work/dankker0900/hf_home}"
export TORCH_HOME="${TORCH_HOME:-/work/dankker0900/torch_home}"
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${JOB_TMP}" \
  "${REPO_ROOT}/logs" "$(dirname "${OUTPUT}")"

module purge
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-biflow_fix2}"
ACTIVE_VENV="${VENV_PATH:-/work/dankker0900/flowtok_venv}"
if [[ -d "${ACTIVE_VENV}" ]]; then
  source "${ACTIVE_VENV}/bin/activate"
fi

cd "${REPO_ROOT}"
python scripts/eval_transport_error.py \
  --images-dir "${COCO_ROOT}/val2017" \
  --captions "${COCO_ROOT}/annotations/captions_val2017.json" \
  --output "${OUTPUT}" \
  --samples "${SAMPLES:-128}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --workers "${WORKERS:-4}" \
  --steps "${STEPS:-20}" \
  --cfg "${CFG:-2.0}" \
  --text-temperature "${TEXT_TEMPERATURE:-0.0}" \
  --zv-temperature "${ZV_TEMPERATURE:-0.0}" \
  --seed "${SEED:-20260821}"
