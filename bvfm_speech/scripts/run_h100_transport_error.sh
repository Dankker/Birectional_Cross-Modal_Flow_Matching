#!/bin/bash
#SBATCH --job-name=bvfm_sp_err
#SBATCH --account=MST114566
#SBATCH --partition=normal2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BVFM_SPEECH_ROOT:-}" ]]; then
  REPO_ROOT="${BVFM_SPEECH_ROOT}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
OUTPUT="${OUTPUT:-${REPO_ROOT}/runs/transport_error/speech_transport.npz}"
JOB_TMP="${BVFM_JOB_TMP_ROOT:-/tmp/${USER}}/bvfm_sp_err_${SLURM_JOB_ID}"

export HF_HOME="${HF_HOME:-/work/dankker0900/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/dankker0900/hf_datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/work/dankker0900/hf_models}"
export TORCH_HOME="${TORCH_HOME:-/work/dankker0900/torch_home}"
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" "${JOB_TMP}" "${REPO_ROOT}/logs" "$(dirname "${OUTPUT}")"

module purge
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-biflow_fix2}"
if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}/Semantic-VAE:${REPO_ROOT}:${PYTHONPATH:-}"

ARGS=(
  --ckpt-dir "${CKPT_DIR:-${REPO_ROOT}/checkpoints/ckpt_joint_svae_zeroshot_norm}"
  --checkpoint "${CHECKPOINT:-latest.pt}"
  --output "${OUTPUT}"
  --samples "${SAMPLES:-100}"
  --steps "${STEPS:-20}"
  --solver "${SOLVER:-heun}"
  --tts-cfg "${TTS_CFG:-1.0}"
  --zv-temperature "${ZV_TEMPERATURE:-0.0}"
  --reference-mode "${REFERENCE_MODE:-other}"
  --seed "${SEED:-20260821}"
)
if [[ -n "${CONFIG:-}" ]]; then
  ARGS+=(--config "${CONFIG}")
fi
if [[ -n "${MANIFEST:-}" ]]; then
  ARGS+=(--manifest "${MANIFEST}")
fi
if [[ -n "${SPEAKER_MODEL:-}" ]]; then
  ARGS+=(--speaker-model "${SPEAKER_MODEL}")
fi
if [[ -n "${SPEAKER_SAVEDIR:-}" ]]; then
  ARGS+=(--speaker-savedir "${SPEAKER_SAVEDIR}")
fi

cd "${REPO_ROOT}"
python scripts/eval_transport_error.py "${ARGS[@]}"
