# Bidirectional Variational Flow Matching

This repository collects the two research branches of **Bidirectional Variational Flow Matching (BVFM)**:

* **Text-to-Image / Image-to-Text (T2I/I2T)**
* **Automatic Speech Recognition / Text-to-Speech (ASR/TTS)**

BVFM studies bidirectional cross-modal generation with a shared flow-matching framework, where the same vector field can be integrated in opposite directions for different cross-modal tasks.

The repository is organized into two main workspaces:

* `bvfm_image/`: image/text BVFM implementation.
* `bvfm_speech/`: speech/text SVAE-BVFM implementation.

Large runtime artifacts are kept out of this GitHub tree. Generated demos,
logs, runs, datasets, and model binaries should be stored in local ignored
directories or in the separate Hugging Face model repository.

## Model Weights

Pretrained BVFM checkpoints and the required model assets are available on Hugging Face:

**[🤗 Bidirectional Cross-Modal Flow Matching — Hugging Face](https://huggingface.co/Dankker0900/Birectional_Cross-Modal_Flow_Matching)**

The Hugging Face repository contains deployment checkpoints for both the
**image/text** and **speech/text** branches.

## Datasets

The BVFM experiments use the following public datasets.

### Image–Text

* **[MS COCO](https://cocodataset.org/)**

  * Used for paired image-caption training and evaluation.
  * Used by the T2I and I2T experiments.

### Automatic Speech Recognition

* **[LibriSpeech](https://www.openslr.org/12/)**

  * Training: `train-clean-100`
  * Evaluation: `test-clean`, `test-other`

### Text-to-Speech

* **[LibriTTS](https://www.openslr.org/60/)**

  * Training: `train-clean-100`
  * Evaluation: `test-clean`

The datasets are not redistributed in this repository. Please download them
from their official sources and configure the corresponding local paths in the
configuration files or job scripts of each branch.

## Layout

```text
bvfm/
  bvfm_image/    image branch: configs, jobs, scripts, image BVFM package
  bvfm_speech/   speech SVAE/BVFM branch: biflow package, Semantic-VAE, configs, scripts
```

## Image Branch

Move to the image branch and run the smoke test:

```bash
cd bvfm_image
bash jobs/smoke_test.sh
```

Download the image model package from Hugging Face, then point the code at the
downloaded model-repository root:

```bash
export BVFM_WEIGHTS_ROOT=/path/to/bvfm_huggingface
```

The image deployment package contains:

* `image/bvfm_image_step40000.pt`
* `image/FlowTiTok_512.bin`

Training-only initializers remain separate from the deployment package.

## Speech Branch

Move to the speech branch and start training with:

```bash
cd bvfm_speech
python train.py --config configs/cutmanifest_svae_latent.json
```

The speech deployment package is stored in the same Hugging Face model
repository:

* `speech/bvfm_speech_step299999_inference.pt`
* `speech/merged_config.json`
* `speech/semantic_vae_1000k/`

The speech branch uses **Semantic-VAE** latents and decodes generated latents
with the bundled Semantic-VAE checkpoint.

It is not a BigVGAN-mel BVFM branch. Any remaining `bigvgan` names are inherited
from older option names or files inside the Semantic-VAE implementation.

Checkpoint binaries are intentionally ignored by Git and are not duplicated in
this source repository.

## Transport Error Analysis

The repository also contains scripts for analyzing the effect of the
variational latent `z_v` on bidirectional transport.

Submit the two independent GPU jobs:

```bash
cd bvfm_image
sbatch jobs/eval_transport_error.sh

cd ../bvfm_speech
sbatch scripts/run_h100_transport_error.sh
```

The default settings use:

* **Image–Text:** 128 sorted COCO validation pairs.
* **Speech–Text:** 100 LibriTTS test utterances.

Both experiments use the one-sided `z_v` prior mean

```text
ZV_TEMPERATURE=0
```

for the deterministic main comparison.

The number of samples, integration steps, checkpoint paths, dataset paths, and
output paths can be overridden through the environment variables exposed by
the corresponding job scripts.

After both GPU jobs finish, generate the four-panel transport-error figure and
CSV exports on CPU:

```bash
cd ..
python scripts/plot_transport_error.py
```

Default outputs are written under:

```text
runs/transport_error/
```

For the speech branch, the transport-error curve is an
**oracle-MAS-aligned latent analysis**: the paired real SVAE target determines
the frame topology. This setting is intentionally distinct from deployment TTS,
where durations are obtained from the duration predictor.

The speech `without_zv` arm is a true projector bypass:

```text
style_e=None
```

at every vector-field call, rather than the older zero-vector ablation.
