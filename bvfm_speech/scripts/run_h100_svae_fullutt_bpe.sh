#!/bin/bash
#SBATCH --job-name=train_zuvf_bpe
#SBATCH --account=MST114566
#SBATCH --partition=normal2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH -o /work/dankker0900/dataset/logs/%x_%j.out
#SBATCH -e /work/dankker0900/dataset/logs/%x_%j.err

set -eo pipefail

module purge
module load miniconda3/24.11.1

export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

eval "$(conda shell.bash hook)"
conda activate biflow_fix2

REPO_ROOT="/work/dankker0900/bvfm/bvfm_speech"
CONFIG_PATH="${REPO_ROOT}/configs/cutmanifest_svae_latent_bpe.json"
TRAIN_MANIFEST="/work/dankker0900/dataset/processed_svae_unified/full_manifest_clean.jsonl"
BPE_PREFIX="${REPO_ROOT}/tokenizers/libritts_bpe_500"
SPEAKER_BANK="${SPEAKER_BANK:-/work/dankker0900/dataset/processed_svae_unified/speaker_ecapa_avg.pt}"

export HF_HOME=/work/dankker0900/hf_home
export HF_DATASETS_CACHE=/work/dankker0900/hf_datasets
export TRANSFORMERS_CACHE=/work/dankker0900/hf_models
export TORCH_HOME=/work/dankker0900/torch_home
export PYTHONPATH="/work/dankker0900/bvfm/bvfm_speech/Semantic-VAE:${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "${REPO_ROOT}/tokenizers"

if [[ ! -f "${BPE_PREFIX}.model" ]]; then
  python "${REPO_ROOT}/scripts/train_bpe_tokenizer.py" \
    --manifest "$TRAIN_MANIFEST" \
    --output-prefix "$BPE_PREFIX" \
    --vocab-size 500
fi

if [[ "${PRECOMPUTE_SPK_BANK:-1}" == "1" && ! -f "$SPEAKER_BANK" ]]; then
  python "${REPO_ROOT}/scripts/precompute_ecapa_speaker_bank.py" \
    --out "$SPEAKER_BANK" \
    --top-k-spk "${TOP_K_SPK:-1000}" \
    --max-utts-per-spk "${SPK_BANK_MAX_UTTS_PER_SPK:-80}"
fi

EXTRA_ARGS=(--speaker-emb-path "$SPEAKER_BANK")
if [[ -n "${BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${DEMO_EVERY:-}" ]]; then
  EXTRA_ARGS+=(--demo-every "$DEMO_EVERY")
fi
if [[ -n "${COMPILE_ENABLE:-}" ]]; then
  EXTRA_ARGS+=(--compile-enable "$COMPILE_ENABLE")
fi

echo "[INFO] config=$CONFIG_PATH bpe=${BPE_PREFIX}.model"
python "${REPO_ROOT}/train.py" --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
