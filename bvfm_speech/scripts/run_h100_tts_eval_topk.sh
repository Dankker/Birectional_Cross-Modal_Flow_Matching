#!/bin/bash
#SBATCH --job-name=tts_eval_topk
#SBATCH --account=MST114566
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -p normal,normal2
#SBATCH -o /work/dankker0900/dataset/logs/%x_%j.out
#SBATCH -e /work/dankker0900/dataset/logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hm1326abc.ee13@nycu.edu.tw

set -euo pipefail

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
SCRIPT_PATH="${REPO_ROOT}/scripts/eval_tts_test_topk.py"

export PYTHONPATH="/work/dankker0900/bvfm/bvfm_speech/Semantic-VAE:${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/ckpt_joint_svae_zeroshot_grad}"
CHECKPOINT="${CHECKPOINT:-latest.pt}"
MAX_TEST_ROWS="${MAX_TEST_ROWS:-50}"
NUM_SPEAKERS="${NUM_SPEAKERS:-20}"
PAIRING="${PAIRING:-round_robin}"
DEVICE="${DEVICE:-cuda}"
PRINT_EVERY="${PRINT_EVERY:-1}"

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

EXTRA_ARGS=()
if [[ -n "${TEST_MANIFEST:-}" ]]; then
  EXTRA_ARGS+=(--test-manifest "${TEST_MANIFEST}")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  EXTRA_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${DEMO_CFG_SCALE:-}" ]]; then
  EXTRA_ARGS+=(--demo-cfg-scale "${DEMO_CFG_SCALE}")
fi
if [[ -n "${DEMO_PRIOR_TEMP:-}" ]]; then
  EXTRA_ARGS+=(--demo-prior-temp "${DEMO_PRIOR_TEMP}")
fi
if [[ -n "${DEMO_STYLE_TEMP:-}" ]]; then
  EXTRA_ARGS+=(--demo-style-temp "${DEMO_STYLE_TEMP}")
fi
if [[ -n "${ODE_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--ode-steps "${ODE_STEPS}")
fi
if [[ -n "${WHISPER_MODEL:-}" ]]; then
  EXTRA_ARGS+=(--whisper-model "${WHISPER_MODEL}")
fi
if [[ -n "${UTMOS_REPO:-}" ]]; then
  EXTRA_ARGS+=(--utmos-repo "${UTMOS_REPO}")
fi
if [[ -n "${UTMOS_MODEL:-}" ]]; then
  EXTRA_ARGS+=(--utmos-model "${UTMOS_MODEL}")
fi
if is_true "${FORCE_RESYNTHESIZE:-}"; then
  EXTRA_ARGS+=(--force-resynthesize)
fi
if is_true "${SKIP_UTMOS:-}"; then
  EXTRA_ARGS+=(--skip-utmos)
fi
if is_true "${SKIP_WHISPER:-}"; then
  EXTRA_ARGS+=(--skip-whisper)
fi

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] SCRIPT_PATH=${SCRIPT_PATH}"
echo "[INFO] CKPT_DIR=${CKPT_DIR}"
echo "[INFO] CHECKPOINT=${CHECKPOINT}"
echo "[INFO] MAX_TEST_ROWS=${MAX_TEST_ROWS}"
echo "[INFO] NUM_SPEAKERS=${NUM_SPEAKERS}"
echo "[INFO] PAIRING=${PAIRING}"
echo "[INFO] DEVICE=${DEVICE}"
echo "[INFO] EXTRA_ARGS=${EXTRA_ARGS[*]:-<none>}"

python "${SCRIPT_PATH}" \
  --ckpt-dir "${CKPT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --max-test-rows "${MAX_TEST_ROWS}" \
  --num-speakers "${NUM_SPEAKERS}" \
  --pairing "${PAIRING}" \
  --device "${DEVICE}" \
  --print-every "${PRINT_EVERY}" \
  "${EXTRA_ARGS[@]}"
