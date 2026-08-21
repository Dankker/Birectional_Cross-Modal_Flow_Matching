#!/bin/bash
#SBATCH --job-name=svae_recon
#SBATCH --account=MST114566
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -p normal,normal2
#SBATCH -o /work/dankker0900/dataset/logs/%x_%j.out
#SBATCH -e /work/dankker0900/dataset/logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hm1326abc.ee13@nycu.edu.tw

set -euo pipefail

module purge
module load miniconda3/24.11.1

export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

eval "$(conda shell.bash hook)"
conda activate biflow_fix2

REPO_ROOT="/work/dankker0900/Semantic-VAE"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] python=$(which python)"
python - <<'PY'
import torch
print("[INFO] cuda_available=", torch.cuda.is_available())
print("[INFO] cuda_device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[INFO] cuda_device=", torch.cuda.get_device_name(0))
PY

python "${REPO_ROOT}/examples/example.py"

echo "[INFO] recon written under ${REPO_ROOT}/examples"
