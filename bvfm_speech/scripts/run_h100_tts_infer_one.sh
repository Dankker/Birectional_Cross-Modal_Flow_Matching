#!/bin/bash
#SBATCH --job-name=tts_one_zs
#SBATCH --account=MST114566
#SBATCH --partition=normal2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH -o /work/dankker0900/dataset/logs/%x_%j.out
#SBATCH -e /work/dankker0900/dataset/logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hm1326abc.ee13@nycu.edu.tw

set -eo pipefail

export HF_HOME=/work/dankker0900/hf_home
export HF_DATASETS_CACHE=/work/dankker0900/hf_datasets
export TRANSFORMERS_CACHE=/work/dankker0900/hf_models
export TORCH_HOME=/work/dankker0900/torch_home
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

module purge
module load miniconda3/24.11.1

export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

eval "$(conda shell.bash hook)"
conda activate biflow_fix2

REPO_ROOT="/work/dankker0900/bvfm/bvfm_speech"
SCRIPT_PATH="${REPO_ROOT}/scripts/infer_tts_one.py"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/infer_tts_one_zeroshot.json}"

CKPT_DIR="${CKPT_DIR:-}"
CHECKPOINT="${CHECKPOINT:-}"
REF_WAV="${REF_WAV:-${1:-}}"
TEXT="${TEXT:-${2:-}}"
OUT_DIR="${OUT_DIR:-}"
OUT_NAME="${OUT_NAME:-}"
DEVICE="${DEVICE:-}"
SOLVER="${SOLVER:-}"
NFE="${NFE:-}"
CFG_SCALE="${CFG_SCALE:-}"
PRIOR_TEMP="${PRIOR_TEMP:-}"
STYLE_TEMP="${STYLE_TEMP:-}"
SEED="${SEED:-}"

export PYTHONPATH="/work/dankker0900/bvfm/bvfm_speech/Semantic-VAE:${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] CKPT_DIR=${CKPT_DIR:-<config>}"
echo "[INFO] CHECKPOINT=${CHECKPOINT:-<config>}"
echo "[INFO] TEXT=${TEXT:-<config>}"
echo "[INFO] REF_WAV=${REF_WAV:-<config>}"
echo "[INFO] OUT_DIR=${OUT_DIR:-<config>}"
echo "[INFO] OUT_NAME=${OUT_NAME:-<config>}"
echo "[INFO] DEVICE=${DEVICE:-<config>}"
echo "[INFO] SOLVER=${SOLVER:-<config>}"
echo "[INFO] NFE=${NFE:-<config>}"
echo "[INFO] CFG_SCALE=${CFG_SCALE:-<config>}"
echo "[INFO] PRIOR_TEMP=${PRIOR_TEMP:-<config>}"
echo "[INFO] STYLE_TEMP=${STYLE_TEMP:-<config>}"
echo "[INFO] SEED=${SEED:-<random>}"

EXTRA_ARGS=()
if [[ -n "${CKPT_DIR}" ]]; then
  EXTRA_ARGS+=(--ckpt-dir "${CKPT_DIR}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${TEXT}" ]]; then
  EXTRA_ARGS+=(--text "${TEXT}")
fi
if [[ -n "${REF_WAV}" ]]; then
  EXTRA_ARGS+=(--ref-wav "${REF_WAV}")
fi
if [[ -n "${OUT_DIR}" ]]; then
  EXTRA_ARGS+=(--out-dir "${OUT_DIR}")
fi
if [[ -n "${OUT_NAME}" ]]; then
  EXTRA_ARGS+=(--out-name "${OUT_NAME}")
fi
if [[ -n "${DEVICE}" ]]; then
  EXTRA_ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${SOLVER}" ]]; then
  EXTRA_ARGS+=(--solver "${SOLVER}")
fi
if [[ -n "${NFE}" ]]; then
  EXTRA_ARGS+=(--nfe "${NFE}")
fi
if [[ -n "${CFG_SCALE}" ]]; then
  EXTRA_ARGS+=(--cfg-scale "${CFG_SCALE}")
fi
if [[ -n "${PRIOR_TEMP}" ]]; then
  EXTRA_ARGS+=(--prior-temp "${PRIOR_TEMP}")
fi
if [[ -n "${STYLE_TEMP}" ]]; then
  EXTRA_ARGS+=(--style-temp "${STYLE_TEMP}")
fi
if [[ -n "${SEED}" ]]; then
  EXTRA_ARGS+=(--seed "${SEED}")
fi

python "${SCRIPT_PATH}" \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"
