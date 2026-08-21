# Speech Checkpoints

Speech deployment weights are distributed through the separate Hugging Face
model repository:

- `speech/bvfm_speech_step299999_inference.pt`
- `speech/merged_config.json`

The deployment checkpoint contains inference modules and checkpoint metadata,
without optimizer state, periodic snapshots, logs, or evaluation outputs.

The Semantic-VAE decoder checkpoint is packaged separately under:

```text
speech/semantic_vae_1000k
```

Set `BVFM_WEIGHTS_ROOT` to the downloaded model repository, or pass
`--ckpt-dir` and `--checkpoint` explicitly.
