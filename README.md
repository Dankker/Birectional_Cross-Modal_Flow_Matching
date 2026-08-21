# Bidirectional Variational Flow Matching

This repository collects the two BVFM research branches: **text-to-image / image-to-text (T2I/I2T)** and **automatic speech recognition / text-to-speech (ASR/TTS)**.

* `bvfm_image/`: image/text BVFM implementation.
* `bvfm_speech/`: speech/text SVAE-BVFM implementation.

## Model Weights

Pretrained BVFM checkpoints and required model assets are available on Hugging Face:

**[🤗 Bidirectional Cross-Modal Flow Matching — Hugging Face](https://huggingface.co/Dankker0900/Birectional_Cross-Modal_Flow_Matching)**

The Hugging Face repository contains the deployment checkpoints for both the **image/text** and **speech/text** branches.

Large runtime artifacts are kept out of this GitHub tree. Generated demos,
logs, runs, and model binaries belong in local ignored directories or in the
separate Hugging Face model repository.

## Layout

```text
bvfm/
  bvfm_image/    image branch: configs, jobs, scripts, image BVFM package
  bvfm_speech/   speech SVAE/BVFM branch: biflow package, Semantic-VAE, configs, scripts
```

## Image Branch

```bash
cd bvfm_image
bash jobs/smoke_test.sh
```

Download the image model package from Hugging Face, then point the code at the
downloaded model-repository root:

```bash
export BVFM_WEIGHTS_ROOT=/path/to/bvfm_huggingface
```

The deployment package contains:

* `image/bvfm_image_step40000.pt`
* `image/FlowTiTok_512.bin`

Training-only initializers remain separate.

## Speech Branch

```bash
cd bvfm_speech
python train.py --config configs/cutmanifest_svae_latent.json
```

The speech deployment package is stored in the same Hugging Face model
repository:

* `speech/bvfm_speech_step299999_inference.pt`
* `speech/merged_config.json`
* `speech/semantic_vae_1000k/`

The speech branch uses Semantic-VAE latents and decodes generated latents with
the bundled Semantic-VAE checkpoint. It is not a BigVGAN-mel BVFM branch; any
remaining `bigvgan` names are inherited from older option names or files inside
the Semantic-VAE implementation.

Checkpoint binaries are intentionally ignored by Git and are not duplicated in
this source repository.

## Transport Error Analysis

Submit the two independent GPU jobs:

```bash
cd bvfm_image
sbatch jobs/eval_transport_error.sh

cd ../bvfm_speech
sbatch scripts/run_h100_transport_error.sh
```

The defaults use 128 sorted COCO validation pairs and 100 LibriTTS test
utterances. Both use the one-sided `z_v` prior mean (`ZV_TEMPERATURE=0`) for a
deterministic main comparison. Override `SAMPLES`, `STEPS`, checkpoint/data
paths, and output paths through the environment variables exposed by each job.

After both jobs finish, create the four-panel figure and CSV exports on CPU:

```bash
cd ..
python scripts/plot_transport_error.py
```

Default outputs are written under `runs/transport_error/`. The speech curve is
an oracle-MAS-aligned latent analysis: the paired real SVAE target determines
the frame topology. It is intentionally distinct from deployment TTS with
predicted durations. The speech `without_zv` arm is a true projector bypass
(`style_e=None` at every vector-field call), not the older zero-vector
ablation.
