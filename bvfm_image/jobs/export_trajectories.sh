#!/bin/bash
#SBATCH --job-name=bvfm_image_traj
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BVFM_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/trajectories}"
COCO_ROOT="${COCO_ROOT:-${REPO_ROOT}/data/coco}"
JOB_TMP="${BVFM_JOB_TMP_ROOT:-/tmp/${USER}}/bvfm_image_traj_${SLURM_JOB_ID}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export MPLBACKEND=Agg
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
python scripts/export_trajectories.py \
  --coco-captions "${COCO_ROOT}/annotations/captions_val2017.json" \
  --output-dir "${OUT_DIR}" \
  --samples "${SAMPLES:-16}" \
  --text-temperature "${TEXT_TEMPERATURE:-0.5}" \
  --zv-temperature "${ZV_TEMPERATURE:-1.0}"
