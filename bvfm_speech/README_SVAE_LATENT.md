# Semantic-VAE Latent Bi-Flow Recipe

This repo variant replaces the old 100-bin BigVGAN mel speech space with the
Semantic-VAE continuous latent space.

## Speech Space

- Old target: normalized BigVGAN mel, roughly `[T_93Hz, 100]`.
- New target: Semantic-VAE latent, `[T_40Hz, 64]`.
- The training code still uses legacy field names like `mel_len`,
  `ctx_mel_start`, and `core_mel_end`; in this repo they mean Semantic-VAE
  latent frames.

## Build Data

```bash
sbatch /work/dankker0900/bvfm/bvfm_speech/scripts/run_h100_build_svae_unified_manifests.sh
```

Outputs:

- `/work/dankker0900/dataset/processed_svae_unified`
- `/work/dankker0900/dataset/processed_svae_test`

Each row contains `svae_latent_path`, `svae_len`, `speech_dim=64`, and
40 Hz word/duration spans.

## Train

```bash
sbatch /work/dankker0900/bvfm/bvfm_speech/scripts/run_h100_svae_latent.sh
```

Main config:

```text
/work/dankker0900/bvfm/bvfm_speech/configs/cutmanifest_svae_latent.json
```

Important config fields:

- `cache.speech_backend = "svae"`
- `cache.svae_dim = 64`
- `cache.svae_sample_rate = 16000`
- `cache.svae_hop_size = 400`
- `runtime.load_bigvgan_model = true` is a legacy option name; with
  `speech_backend = "svae"` it loads the Semantic-VAE decoder for demos.

## Rationale

The goal is to avoid generating off-manifold mel spectrograms that BigVGAN
cannot vocode cleanly. The flow now learns:

```text
z_c <-> z_svae
```

instead of:

```text
z_c <-> mel
```

Demo audio is decoded by Semantic-VAE, not BigVGAN.
