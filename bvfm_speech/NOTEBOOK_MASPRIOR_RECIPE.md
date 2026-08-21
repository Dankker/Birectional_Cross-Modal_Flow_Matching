# Notebook MAS-prior recipe

This repo is a copy of `biflow_repo_cutmanifest_latent_vi` with a hybrid
config: it keeps the stronger notebook MAS-prior training losses, but uses the
new `processed_unified` data flow for full-length duration / ASR supervision.

Key defaults in `configs/cutmanifest_notebook_unified_clean.json`:

- Uses `processed_unified` manifests.
- Uses full-TTS teacher for duration learning.
- Uses full-ASR CTC auxiliary supervision.
- Enables SpeechT5 residual adapter.
- Disables length predictor and mel refiner, so TTS quality reflects the base VF.
- Uses DiT VF `hidden=1024`, `depth=12`, `heads=16`.
- Uses stochastic acoustic prior NLL with `w_prior=0.5`.
- Enables forward and backward rollout endpoint loss with `w_end=0.3`.
- Uses CTC weights `w_ctc_T=0.4`, `w_ctc_hat=0.8`.
- Enables acoustic prior NLL with `w_prior=0.5`, matching the notebook's effective prior loss.
- Disables latent-VI additions: style latent, canonical KL/NLL, SSL hidden/zc loss.

Run with:

```bash
sbatch scripts/run_h100_cutmanifest_singlevf.sh
```

## Exact notebook port

`train.py` is still the current structured training code with a hybrid config.
For a direct notebook-code path, use:

```bash
sbatch scripts/run_h100_notebook_exact.sh
```

This runs `train_notebook_masprior_exact.py`, which is extracted from the main
training cell of `/work/dankker0900/toy_big_masprior_cut_full.ipynb` with only
workspace paths and environment overrides changed. It keeps the notebook's
length predictor, refiner, acoustic prior NLL, fwd/bwd rollout losses, and save
format.
