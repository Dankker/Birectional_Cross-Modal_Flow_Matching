#!/bin/bash
#SBATCH --job-name=train_svae_asrbwd
#SBATCH --account=MST114566
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH -p normal,normal2
#SBATCH -o /work/dankker0900/dataset/logs/%x_%j.out
#SBATCH -e /work/dankker0900/dataset/logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hm1326abc.ee13@nycu.edu.tw

set -euo pipefail

export HF_HOME=/work/dankker0900/hf_home
export HF_DATASETS_CACHE=/work/dankker0900/hf_datasets
export TRANSFORMERS_CACHE=/work/dankker0900/hf_models
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

module purge
module load miniconda3/24.11.1

export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

eval "$(conda shell.bash hook)"
conda activate biflow_fix2

REPO_ROOT="/work/dankker0900/bvfm/bvfm_speech"
CONFIG_PATH="${REPO_ROOT}/configs/cutmanifest_svae_latent_asr_bwdfm.json"
SPEAKER_BANK="${SPEAKER_BANK:-/work/dankker0900/dataset/processed_svae_unified/speaker_ecapa_avg.pt}"

export PYTHONPATH="/work/dankker0900/bvfm/bvfm_speech/Semantic-VAE:${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_HOME="/work/dankker0900/torch_home"
mkdir -p "$TORCH_HOME"

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"
echo "[INFO] SPEAKER_BANK=${SPEAKER_BANK}"

if [[ "${PRECOMPUTE_SPK_BANK:-1}" == "1" && ! -f "${SPEAKER_BANK}" ]]; then
  echo "[INFO] speaker bank missing; precomputing ECAPA speaker averages"
  python "${REPO_ROOT}/scripts/precompute_ecapa_speaker_bank.py" \
    --out "${SPEAKER_BANK}" \
    --top-k-spk "${TOP_K_SPK:-1000}" \
    --max-utts-per-spk "${SPK_BANK_MAX_UTTS_PER_SPK:-80}"
fi

EXTRA_ARGS=(--speaker-emb-path "${SPEAKER_BANK}")
if [[ -n "${BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${COMPILE_ENABLE:-}" ]]; then
  EXTRA_ARGS+=(--compile-enable "${COMPILE_ENABLE}")
fi
if [[ -n "${MATMUL_PRECISION:-}" ]]; then
  EXTRA_ARGS+=(--matmul-precision "${MATMUL_PRECISION}")
fi
if [[ -n "${GPU_TEXT_CACHE_LIMIT_GIB:-}" ]]; then
  EXTRA_ARGS+=(--gpu-text-cache-limit-gib "${GPU_TEXT_CACHE_LIMIT_GIB}")
fi
if [[ -n "${GPU_MEL_CACHE:-}" ]]; then
  EXTRA_ARGS+=(--gpu-mel-cache "${GPU_MEL_CACHE}")
fi
if [[ -n "${GPU_TEXT_CACHE:-}" ]]; then
  EXTRA_ARGS+=(--gpu-text-cache "${GPU_TEXT_CACHE}")
fi
if [[ -n "${DEMO_EVERY:-}" ]]; then
  EXTRA_ARGS+=(--demo-every "${DEMO_EVERY}")
fi
if [[ -n "${LOAD_BIGVGAN_MODEL:-}" ]]; then
  EXTRA_ARGS+=(--load-bigvgan-model "${LOAD_BIGVGAN_MODEL}")
fi

echo "[INFO] EXTRA_ARGS=${EXTRA_ARGS[*]:-<none>}"
python "${REPO_ROOT}/train.py" --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
