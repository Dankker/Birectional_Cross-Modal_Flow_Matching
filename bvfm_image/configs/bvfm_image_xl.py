"""FlowTok-XL shared, direction-free BVFM configuration."""

import os
from dataclasses import dataclass

import ml_collections


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_ROOT = os.environ.get(
    "COCO_ROOT", os.path.join(REPO_ROOT, "data", "coco"))
BVFM_WEIGHTS_ROOT = os.environ.get("BVFM_WEIGHTS_ROOT")
IMAGE_WEIGHTS_DIR = os.environ.get(
    "BVFM_IMAGE_WEIGHTS_DIR",
    os.path.join(BVFM_WEIGHTS_ROOT, "image") if BVFM_WEIGHTS_ROOT else "",
)


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


MODEL = Args(
    learn_sigma=False,
    channels=16,
    use_t2i=True,
    clip_dim=768,
    num_clip_token=77,
    gradient_checking=False,
    cfg_indicator=0.10,
    noising_type="none",
    noising_scale=0.1,
    use_t2t_temperature=True,
    use_task_condition=False,
    use_task_adapters=False,
    use_bvfm_condition=True,
    bvfm_latent_dim=128,
    bvfm_adapter_rank=128,
    bvfm_adapter_scale=1.0,
    textVAE=Args(
        num_blocks=6,
        hidden_dim=1024,
        num_attention_heads=8,
        dropout_prob=0.1,
        clip_loss_weight=0.0,
        align_quantized=False,
        use_pretrained=False,
        freeze_encoder=True,
    ),
)


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.nnet_path = os.environ.get(
        "FLOWTOK_CKPT", os.path.join(REPO_ROOT, "assets", "FlowTok-XL.pth"))
    config.tokenizer_checkpoint = os.environ.get(
        "FLOWTITOK_CKPT",
        os.path.join(IMAGE_WEIGHTS_DIR, "FlowTiTok_512.bin")
        if IMAGE_WEIGHTS_DIR
        else os.path.join(REPO_ROOT, "assets", "FlowTiTok_512.bin"))
    config.decoder_init = os.environ.get(
        "DECODER_INIT",
        os.path.join(REPO_ROOT, "assets", "decoder_init.pt"))

    config.dataset = d(resize_shorter_edge=512, crop_size=512)
    config.vq_model = d(
        deterministic=False,
        token_size=16,
        vit_enc_model_size="base",
        vit_dec_model_size="large",
        vit_enc_patch_size=16,
        vit_dec_patch_size=16,
        num_latent_tokens=77,
        is_legacy=False,
        use_rmsnorm=False,
        use_swiglu=True,
        scale_factor=1.0143,
    )
    config.nnet = d(name="flowtok-xl", model_args=MODEL)
    config.decoder_type = "ar"
    config.decoder = d(
        latent_dim=16,
        d_model=768,
        depth=6,
        num_heads=8,
        d_ff=3072,
        vocab_size=49408,
        seq_len=77,
        dropout=0.1,
    )
    config.bvfm = d(
        latent_dim=128,
        hidden_dim=256,
        dropout=0.1,
        logvar_bias=-2.0,
    )
    config.data = d(
        train_images_dir=os.path.join(COCO_ROOT, "train2017"),
        train_captions=os.path.join(
            COCO_ROOT, "annotations", "captions_train2017.json"),
        val_images_dir=os.path.join(COCO_ROOT, "val2017"),
        val_captions=os.path.join(
            COCO_ROOT, "annotations", "captions_val2017.json"),
    )
    config.train = d(
        seed=1234,
        n_steps=40_000,
        batch_size_per_gpu=20,
        num_workers=8,
        lr_vf=1e-4,
        lr_variational=1e-4,
        lr_decoder=3e-5,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        warmup_steps=1_000,
        grad_clip=1.0,
        w_bvfm=1.0,
        bvfm_beta=0.3,
        posterior_temperature=1.0,
        kl_start=500,
        kl_anneal_steps=5_000,
        w_t2i_distill=1.0,
        distill_every=2,
        distill_batch=8,
        distill_prior_temperature=1.0,
        startup_max_relative_drift=1e-2,
        w_teacher_ce=0.5,
        w_endpoint_ce=1.0,
        endpoint_ce_start=500,
        endpoint_ce_every=1,
        endpoint_ce_batch=8,
        w_endpoint_latent=2.0,
        endpoint_latent_start=1_000,
        endpoint_latent_every=4,
        endpoint_latent_batch=4,
        endpoint_steps=20,
        endpoint_cosine_weight=0.25,
        rollout_prior_temperature=1.0,
        log_every=50,
        eval_every=1_000,
        save_every=2_000,
        snapshot_every=10_000,
        output_dir=os.environ.get(
            "BVFM_OUTPUT_DIR", os.path.join(REPO_ROOT, "runs", "train")),
    )
    config.diag = d(
        n_val_captions=1_024,
        n_val_images=128,
        n_val_fm_pairs=32,
        n_print_samples=6,
        i2t_steps=20,
        t2i_steps=20,
        t2i_cfg=2.0,
        t2i_prompts=4,
        t2i_every=1_000,
        pca_every=1_000,
        pca_pairs=32,
        pca_example_index=1,
        pca_samples=32,
        pca_temperature=1.0,
        pca_steps=20,
        pca_t2i_cfg=2.0,
        prior_temperature=0.0,
        max_t2i_relative_velocity_drift=0.02,
        min_i2t_bleu4_for_best=0.10,
    )
    return config
