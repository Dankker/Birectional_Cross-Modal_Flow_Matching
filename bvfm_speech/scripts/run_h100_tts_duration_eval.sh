#!/bin/bash
#SBATCH --job-name=tts_dur_eval
#SBATCH --account=MST114566
#SBATCH --partition=normal2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
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
export PYTHONPATH="/work/dankker0900/bvfm/bvfm_speech/Semantic-VAE:${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/ckpt_joint_svae_zeroshot_norm}"
CHECKPOINT="${CHECKPOINT:-latest.pt}"
DURATION_MODE="${DURATION_MODE:-both}"   # both, mas, or pred
MAX_ITEMS="${MAX_ITEMS:-50}"
DEVICE="${DEVICE:-cuda}"
SOLVER="${SOLVER:-heun}"
ODE_STEPS="${ODE_STEPS:-20}"
CFG_SCALE="${CFG_SCALE:-1.0}"
PRIOR_TEMP="${PRIOR_TEMP:-0.0}"
STYLE_TEMP="${STYLE_TEMP:-0.0}"
SEED="${SEED:-1234}"

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

COMMON_ARGS=(
  --ckpt-dir "${CKPT_DIR}"
  --checkpoint "${CHECKPOINT}"
  --max-items "${MAX_ITEMS}"
  --device "${DEVICE}"
  --solver "${SOLVER}"
  --ode-steps "${ODE_STEPS}"
  --cfg-scale "${CFG_SCALE}"
  --prior-temp "${PRIOR_TEMP}"
  --style-temp "${STYLE_TEMP}"
  --seed "${SEED}"
)

if [[ -n "${CONFIG:-}" ]]; then
  COMMON_ARGS+=(--config "${CONFIG}")
fi
if [[ -n "${MANIFEST:-}" ]]; then
  COMMON_ARGS+=(--manifest "${MANIFEST}")
fi
if [[ -n "${SPEAKER:-}" ]]; then
  COMMON_ARGS+=(--speaker "${SPEAKER}")
fi
if [[ -n "${WHISPER_MODEL:-}" ]]; then
  COMMON_ARGS+=(--whisper-model "${WHISPER_MODEL}")
fi
if [[ -n "${UTMOS_REPO:-}" ]]; then
  COMMON_ARGS+=(--utmos-repo "${UTMOS_REPO}")
fi
if [[ -n "${UTMOS_MODEL:-}" ]]; then
  COMMON_ARGS+=(--utmos-model "${UTMOS_MODEL}")
fi
if [[ -n "${SPK_SIM_MODEL:-}" ]]; then
  COMMON_ARGS+=(--spk-sim-model "${SPK_SIM_MODEL}")
fi
if [[ -n "${SPK_SIM_SAVEDIR:-}" ]]; then
  COMMON_ARGS+=(--spk-sim-savedir "${SPK_SIM_SAVEDIR}")
fi
if [[ -n "${SPK_SIM_MAX_SEC:-}" ]]; then
  COMMON_ARGS+=(--spk-sim-max-sec "${SPK_SIM_MAX_SEC}")
fi
if is_true "${SKIP_WHISPER:-}"; then
  COMMON_ARGS+=(--skip-whisper)
fi
if is_true "${SKIP_UTMOS:-}"; then
  COMMON_ARGS+=(--skip-utmos)
fi
if is_true "${SKIP_SPK_SIM:-}"; then
  COMMON_ARGS+=(--skip-spk-sim)
fi

run_mode() {
  local mode="$1"
  local script_path
  script_path="${REPO_ROOT}/scripts/test_tts_duration_${mode}.py"
  local mode_args=("${COMMON_ARGS[@]}")
  if [[ -n "${OUT_ROOT:-}" ]]; then
    mode_args+=(--out-dir "${OUT_ROOT}/${mode}")
  fi

  echo "[INFO] Running duration mode: ${mode}"
  echo "[INFO] Script: ${script_path}"
  echo "[INFO] CKPT_DIR=${CKPT_DIR}"
  echo "[INFO] CHECKPOINT=${CHECKPOINT}"
  echo "[INFO] MAX_ITEMS=${MAX_ITEMS}"
  echo "[INFO] DEVICE=${DEVICE}"
  echo "[INFO] SOLVER=${SOLVER}"
  echo "[INFO] ODE_STEPS=${ODE_STEPS}"
  echo "[INFO] OUT_ROOT=${OUT_ROOT:-<default>}"

  python "${script_path}" "${mode_args[@]}"
}

case "${DURATION_MODE}" in
  both)
    run_mode mas
    run_mode pred
    ;;
  mas|pred)
    run_mode "${DURATION_MODE}"
    ;;
  *)
    echo "DURATION_MODE must be one of: both, mas, pred" >&2
    exit 2
    ;;
esac
