# BVFM Speech

This branch contains the speech BVFM implementation based on Semantic-VAE
latents. The speech endpoint is not a BigVGAN mel spectrogram; it is a
40 Hz, 64-dimensional Semantic-VAE latent sequence.

## Bundled Components

- `biflow/`: BVFM model, checkpointing, data, and utility code.
- `train.py`: main training entrypoint.
- `scripts/`: SLURM launchers, evaluation tools, and one-shot TTS inference.
- `configs/`: experiment and inference JSON configs.
- `Semantic-VAE/`: local Semantic-VAE code used for latent extraction and audio
  reconstruction.
- `checkpoints/`: ignored local placement for BVFM checkpoints.
- `Semantic-VAE/ckpts/`: ignored local placement for Semantic-VAE weights.

The checkpoint directories are intentionally ignored by Git. Deployment
weights live in the separate Hugging Face model repository. Set
`BVFM_WEIGHTS_ROOT=/path/to/downloaded/model-repo` before inference.

## Main Config

```bash
python train.py --config configs/cutmanifest_svae_latent.json
```

The relevant SVAE fields are:

```json
{
  "cache": {
    "speech_backend": "svae",
    "semantic_vae_root": "./Semantic-VAE",
    "semantic_vae_ckpt": "${BVFM_WEIGHTS_ROOT}/speech/semantic_vae_1000k",
    "svae_dim": 64,
    "svae_sample_rate": 16000,
    "svae_hop_size": 400
  }
}
```

Some legacy variable names still contain `mel` or `bigvgan`; in this branch
they refer to the SVAE latent/audio decode path unless the config explicitly
selects another backend.

## One-Shot TTS

```bash
python scripts/infer_tts_one.py \
  --config configs/infer_tts_one_zeroshot_norm.json
```

The config controls the checkpoint path, text, reference audio, seed, solver,
and output directory.

## Notes

Generated demos, logs, caches, and large model weights are excluded from normal
Git tracking.
