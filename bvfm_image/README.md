# BVFM Image

This repository contains the cleaned image/text implementation of
Bidirectional Variational Flow Matching on FlowTok-XL. A single shared vector
field is integrated from text to image (`0 -> 1`) or image to text (`1 -> 0`);
there is no task or direction embedding.

The variational components are:

```text
q(z_v | z_text, z_image)
p_text(z_v | z_text)
p_image(z_v | z_image)
v_theta(z_t, t, z_v)
```

## Selected model

The ready-to-use checkpoint is distributed through the separate Hugging Face
model repository as:

```text
image/bvfm_image_step40000.pt
```

It contains the shared field, both inference priors, full-pair posterior, and
AR caption decoder. Set `BVFM_WEIGHTS_ROOT` to the downloaded model-repository
root. Alternatively, place the files under ignored local `checkpoints/` and
`assets/` directories or override `--checkpoint`/`CKPT` and
`FLOWTITOK_CKPT`.

## Repository layout

```text
bvfm_image/
├── bvfm_image/              # shared runtime and training implementation
├── configs/bvfm_image_xl.py # single canonical configuration
├── scripts/
│   ├── train.py
│   ├── infer_t2i.py
│   ├── infer_i2t.py
│   └── export_trajectories.py
├── jobs/                    # Slurm entry points
├── assets/                  # released FlowTok/FlowTiTok + decoder warm start
└── checkpoints/             # selected joint checkpoint location
```

## Environment

The cluster jobs use the verified environment:

```bash
module load miniconda3/24.11.1
conda activate biflow_fix2
# Optional when using a separate virtual environment:
export VENV_PATH=/path/to/venv
source "$VENV_PATH/bin/activate"
```

Binary model files under `assets/` and `checkpoints/` are ignored by Git.
Place them according to the README files in those directories.

Set the COCO root before training or exporting trajectories:

```bash
export COCO_ROOT=/path/to/coco
```

Expected dataset layout:

```text
$COCO_ROOT/train2017/
$COCO_ROOT/val2017/
$COCO_ROOT/annotations/captions_train2017.json
$COCO_ROOT/annotations/captions_val2017.json
```

## T2I inference

The evaluated setting uses the mean of `p_text(z_v | z_text)`:

```bash
cd /path/to/bvfm_image
PROMPT="Three teddy bears sitting together" \
OUT_DIR="$PWD/runs/teddy" \
sbatch jobs/infer_t2i.sh
```

Sample velocity modes instead:

```bash
PROMPT="A corgi wearing sunglasses on a beach" \
ZV_MODE=sample SAMPLES=8 ZV_TEMPERATURE=1.0 \
OUT_DIR="$PWD/runs/corgi_samples" \
sbatch jobs/infer_t2i.sh
```

The Python entry point also accepts repeated `--prompt` arguments or a
newline-delimited `--prompt-file`.

## I2T inference

Caption one image:

```bash
IMAGE_PATH=/path/to/image.png \
OUT_JSON="$PWD/runs/caption.json" \
sbatch jobs/infer_i2t.sh
```

Caption a directory:

```bash
IMAGE_DIR=/path/to/images \
OUT_JSON="$PWD/runs/captions.json" \
sbatch jobs/infer_i2t.sh
```

Use `ZV_MODE=none`, `mean`, or `sample` for ablations. The default `mean`
matches the reported COCO metrics.

## Training

Start a new shared-BVFM run from released FlowTok-XL and the decoder warm
start:

```bash
cd /path/to/bvfm_image
COCO_ROOT=/path/to/coco \
RESUME=none OUT_DIR="$PWD/runs/train_v1" \
sbatch jobs/train.sh
```

Resume automatically from `$OUT_DIR/latest.pt`:

```bash
COCO_ROOT=/path/to/coco OUT_DIR="$PWD/runs/train_v1" \
sbatch jobs/train.sh
```

Optional overrides include `NGPU`, `BATCH_SIZE`, and `N_STEPS`.

Training writes:

- `latest.pt`: resumable state with optimizer;
- `final.pt`: optimizer-free deployment candidate;
- `step*.pt`: periodic optimizer-free snapshots;
- `best_proxy.pt`: best caption score that also passes the inexpensive
  velocity-drift gate. This is not automatically the best task-level joint
  checkpoint; select the paper checkpoint using T2I FID/CLIP and I2T metrics.

## Generated-image round-trip trajectories

This reproduces the 16-pair T2I-to-I2T PCA figure. T2I does not show paired
COCO image targets. The with-`z_v` images are decoded, re-encoded, and then
fed to both reverse arms under identical inputs.

```bash
sbatch jobs/export_trajectories.sh
```

The TikZ-ready CSV contains only:

```text
panel,sample_id,point_type,step_index,time,pc1,pc2
```

## Smoke test

Run imports, checkpoint loading, one T2I generation, and reverse captioning on
a compute node:

```bash
sbatch jobs/smoke_test.sh
```
