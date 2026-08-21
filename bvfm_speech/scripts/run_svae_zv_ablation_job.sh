#!/usr/bin/env bash
#SBATCH --job-name=svae_zv_plot
#SBATCH --account=MST114566
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/dankker0900/bvfm/bvfm_speech/logs/svae_zv_plot_%j.out
#SBATCH --error=/work/dankker0900/bvfm/bvfm_speech/logs/svae_zv_plot_%j.err

set -euo pipefail

REPO_ROOT="/work/dankker0900/bvfm/bvfm_speech"
mkdir -p "${REPO_ROOT}/logs"

cd "${REPO_ROOT}"
/home/dankker0900/.conda/envs/biflow_fix2/bin/python scripts/visualize_svae_zv_ablation.py \
  --ckpt-dir "${REPO_ROOT}/checkpoints/ckpt_joint_svae_zeroshot_norm" \
  --checkpoint latest.pt \
  --out-dir "${REPO_ROOT}/svae_latent_visualizations/zv_ablation" \
  --solver heun \
  --nfe 20 \
  --prior-temp 0.0 \
  --style-temp 0.0 \
  --seed 20260821
