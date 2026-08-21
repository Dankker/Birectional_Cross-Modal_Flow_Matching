import json


DEFAULT_CONFIG = {
    "tokenizer": {
        "type": "char",
        "model_path": None,
    },
    "paths": {
        "cut_manifest": "/work/dankker0900/dataset/cut_manifests/cut_manifest_all.jsonl",
        "aligned_manifest": "/work/dankker0900/dataset/align/train_manifest_aligned.jsonl",
        "max_cut_rows": None,
    },
    "cache": {
        "mel_dir": "/work/dankker0900/biflow_repo_cutmanifest_full/cache/mels",
        "speecht5_dir": "/work/dankker0900/biflow_repo_cutmanifest_full/cache/speecht5",
        "wav_cache_max_items": 64,
        "mel_cache_max_items": 512,
        "text_hidden_cache_max_items": 2048,
        "stats_max_unique_wavs": 5000,
        "gpu_mel_cache": True,
        "gpu_text_cache": True,
        "gpu_mel_preload": True,
        "gpu_text_preload": True,
        "gpu_mel_cache_dtype": "float32",
        "gpu_text_cache_dtype": "bfloat16",
        "gpu_mel_cache_limit_gib": None,
        "gpu_text_cache_limit_gib": 8,
        "gpu_cache_reserve_gib": 64.0,
    },
    "data": {
        "batch_size": 8,
        "batch_size_tts": None,
        "batch_size_asr": None,
        "ds_factor": 1,
        "ds_align": 1,
        "top_k_spk": 1000,
        "target_spks": None,
        "force_text_normalize": True,
        "use_dataloader": False,
        "loader_num_workers": 4,
        "loader_pin_memory": True,
        "loader_persistent_workers": True,
        "loader_prefetch_factor": 2,
        "dataset_mel_worker_cache_max_items": 128,
        "enable_length_bucket": False,
        "num_length_buckets": 12,
    },
    "model": {
        "E_spk": 64,
        "spk_scale": 0.2,
        "spk_drop_rate": 0.3,
        "speaker_cond_type": "table",
        "speaker_emb_path": None,
        "speaker_emb_l2_normalize": True,
        "speaker_emb_missing": "error",
        "speaker_emb_trainable": False,
        "speaker_delta_scale": 0.1,
        "speaker_cond_layernorm": True,
        "vf_use_speaker_cond": True,
        "asr_vf_use_speaker_cond": None,
        "asr_use_spk_cond": False,
        "asr_spk_scale": 1.0,
        "asr_spk_unknown": "zero",
        "asr_use_style_cond": False,
        "asr_style_use_mean": True,
        "asr_style_temp": 0.0,
        "asr_style_detach": True,
        "use_tts_source_cond": False,
        "tts_source_cond_hidden": 128,
        "tts_source_cond_scale": 1.0,
        "use_tts_style_latent": False,
        "tts_style_dim": 64,
        "tts_style_hidden": 256,
        "tts_style_dropout": 0.1,
        "tts_style_source_scale": 1.0,
        "tts_style_post_mode": "speech",
        "tts_style_prior_type": "standard_normal",
        "tts_style_prior_canonical_detach": True,
        "tts_style_prior_logvar_bias": None,
        "tts_style_into_source": False,
        "use_true_canonical_latent": False,
        "canonical_dim": 192,
        "canonical_hidden": 256,
        "canonical_dropout": 0.1,
        "use_vf_canonical_text_cond": True,
        "st5_layer_idx": -2,
        "text_encoder_type": "speecht5",
        "text_encoder_dim": 384,
        "text_encoder_layers": 6,
        "text_encoder_heads": 6,
        "text_encoder_ff_mult": 4,
        "text_encoder_conv_ksize": 5,
        "text_encoder_dropout": 0.1,
        "text_encoder_max_len": 1024,
        "use_adapter": True,
        "adapter_bottleneck": 192,
        "adapter_dropout": 0.1,
        "bigvgan_name": "nvidia/bigvgan_24khz_100band",
        "vf_hidden": 1024,
        "vf_depth": 12,
        "vf_heads": 16,
        "vf_dropout": 0.0,
        "vf_max_len": 4096,
        "dur_hidden": 256,
        "dur_dropout": 0.5,
        "use_len_predictor": False,
        "len_hidden": 192,
        "ctc_head_type": "baseline",
        "ctc_hidden": 384,
        "ctc_layers": 2,
        "ctc_ksize": 5,
        "ctc_lstm_hidden": 384,
        "ctc_lstm_layers": 2,
        "ctc_dropout": 0.1,
        "att_decoder_hidden": 384,
        "att_decoder_layers": 4,
        "att_decoder_heads": 6,
        "att_decoder_ff_mult": 4.0,
        "att_decoder_dropout": 0.1,
        "att_decoder_max_len": 512,
        "use_refiner": False,
        "ref_hidden": 512,
        "ref_blocks": 8,
        "ref_ksize": 7,
        "ref_dropout": 0.1,
    },
    "loss": {
        "mel_floor": -11.5,
        "mel_ceil": 2.0,
        "w_end": 0.30,
        "enable_fwd_end_loss": True,
        "w_end_fwd": None,
        "w_end_bwd": None,
        "w_ctc_hat": 0.8,
        "w_ctc_T": 0.4,
        "w_ctc_start": 200,
        "enable_ctc_dur": False,
        "w_ctc_dur": 0.1,
        "ctc_dur_start": 200,
        "enable_att_decoder": False,
        "w_att_decoder": 0.0,
        "att_decoder_start": 200,
        "att_decoder_anneal_steps": 10000,
        "att_decoder_label_smoothing": 0.1,
        "att_decoder_detach_input": False,
        "w_tts_mel": 1.3,
        "w_ref": 0.5,
        "w_delta": 0.0,
        "w_mel_range": 0.0,
        "w_align_start": 0,
        "w_dur": 0.1,
        "w_len": 0.05,
        "mas_temp": 1.0,
        "mas_mode": "hard",
        "mas_mix_alpha": 0.3,
        "fwd_prior_mode": "mas",
        "fwd_prior_mix_alpha": 0.5,
        "fwd_anchor_mode": "mean",
        "fwd_anchor_mix_alpha": 0.5,
        "use_full_tts_teacher": False,
        "w_prior": 0.5,
        "enable_acoustic_prior_nll": True,
        "enable_canonical_nll": False,
        "w_canonical_nll": 0.0,
        "canonical_nll_start": 1000,
        "canonical_nll_anneal_steps": 10000,
        "prior_loss_mode": "gaussian_nll",
        "prior_mu_loss_type": "smooth_l1",
        "w_prior_mu": 1.0,
        "w_prior_var": 0.05,
        "w_prior_nll": 0.1,
        "prior_fixed_logvar": -2.0,
        "prior_var_reg_target": -2.0,
        "w_tts_style_kl": 0.01,
        "tts_style_kl_start": 1000,
        "tts_style_kl_anneal_steps": 10000,
        "enable_full_asr_ctc_aux": False,
        "w_ctc_full": 0.2,
        "full_asr_ctc_aux_start": 1000,
        "full_asr_ctc_aux_every": 4,
        "full_asr_ctc_aux_batch_size": 1,
        "use_stat_match": True,
        "w_stat": 0.05,
        "use_stft": False,
        "w_stft": 0.2,
        "enable_vf_lip": True,
        "vf_lip_start": 1000,
        "w_vf_lip": 0.2,
        "vf_lip_L_hi": 1.0,
        "vf_lip_sigma": 0.01,
        "vf_lip_every": 1,
        "vf_lip_print_every": 200,
        "enable_bilip_diag": True,
        "diag_every": 500,
        "amp_sigma": 0.01,
    },
    "train": {
        "seed": 0,
        "total_steps": 150000,
        "lr_all": 1e-4,
        "lr_schedule": "warmup_cosine",
        "lr_warmup_steps": 1000,
        "lr_min_scale": 0.1,
        "save_every_steps": 1000,
        "keep_last_k": 3,
        "resume_from": None,
        "use_ema": False,
        "ema_decay": 0.9999,
        "log_every": 200,
        "perf_log_every": 200,
        "debug_every": 1000,
        "grad_clip": 1.0,
    },
    "infer": {
        "ode_steps_eval": 8,
        "ode_steps_endloss": 4,
        "full_asr_chunk_core": 256,
        "full_asr_chunk_ctx": 96,
        "full_asr_use_euler": True,
        "demo_cfg_scale": 1.3,
        "demo_prior_temp": 0.0,
        "demo_plot_trajectory": False,
        "demo_trajectory_asr_realization_plot": False,
        "demo_trajectory_asr_realization_speakers": 4,
        "demo_trajectory_asr_realization_styles": 8,
        "demo_every": 1000,
    },
    "runtime": {
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "matmul_precision": "high",
        "compile_enable": False,
        "compile_mode": "max-autotune-no-cudagraphs",
        "compile_dynamic": True,
        "compile_vf": True,
        "compile_refiner": False,
        "compile_ctc_head": False,
    },
    "io": {
        "demo_dir": "/work/dankker0900/biflow_repo_cutmanifest_full/demos_bigvgan_cutmanifest_full",
        "ckpt_dir": "/work/dankker0900/biflow_repo_cutmanifest_full/ckpt_joint_cutmanifest_full",
    },
}


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path=None):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        deep_update(cfg, loaded)
    return cfg


def apply_overrides(cfg, args):
    if args.cut_manifest is not None:
        cfg["paths"]["cut_manifest"] = args.cut_manifest
    if args.aligned_manifest is not None:
        cfg["paths"]["aligned_manifest"] = args.aligned_manifest
    if args.max_cut_rows is not None:
        cfg["paths"]["max_cut_rows"] = args.max_cut_rows
    if args.target_spks:
        cfg["data"]["target_spks"] = args.target_spks
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.total_steps is not None:
        cfg["train"]["total_steps"] = args.total_steps
    if args.lr_all is not None:
        cfg["train"]["lr_all"] = args.lr_all
    if args.lr_warmup_steps is not None:
        cfg["train"]["lr_warmup_steps"] = args.lr_warmup_steps
    if args.ckpt_dir is not None:
        cfg["io"]["ckpt_dir"] = args.ckpt_dir
    if args.demo_dir is not None:
        cfg["io"]["demo_dir"] = args.demo_dir
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.save_every_steps is not None:
        cfg["train"]["save_every_steps"] = args.save_every_steps
    if args.keep_last_k is not None:
        cfg["train"]["keep_last_k"] = args.keep_last_k
    if args.resume_from is not None:
        cfg["train"]["resume_from"] = args.resume_from
    if getattr(args, "demo_every", None) is not None:
        cfg["infer"]["demo_every"] = args.demo_every
    if getattr(args, "load_bigvgan_model", None) is not None:
        cfg["runtime"]["load_bigvgan_model"] = args.load_bigvgan_model == "true"
    if args.use_ema is not None:
        cfg["train"]["use_ema"] = args.use_ema == "true"
    if args.ema_decay is not None:
        cfg["train"]["ema_decay"] = args.ema_decay
    if args.compile_enable is not None:
        cfg["runtime"]["compile_enable"] = args.compile_enable == "true"
    if getattr(args, "matmul_precision", None) is not None:
        cfg["runtime"]["matmul_precision"] = args.matmul_precision
    if args.gpu_mel_cache is not None:
        cfg["cache"]["gpu_mel_cache"] = args.gpu_mel_cache == "true"
    if args.gpu_text_cache is not None:
        cfg["cache"]["gpu_text_cache"] = args.gpu_text_cache == "true"
    if args.gpu_mel_preload is not None:
        cfg["cache"]["gpu_mel_preload"] = args.gpu_mel_preload == "true"
    if args.gpu_text_preload is not None:
        cfg["cache"]["gpu_text_preload"] = args.gpu_text_preload == "true"
    if args.gpu_mel_cache_limit_gib is not None:
        cfg["cache"]["gpu_mel_cache_limit_gib"] = args.gpu_mel_cache_limit_gib
    if args.gpu_text_cache_limit_gib is not None:
        cfg["cache"]["gpu_text_cache_limit_gib"] = args.gpu_text_cache_limit_gib
    if args.gpu_cache_reserve_gib is not None:
        cfg["cache"]["gpu_cache_reserve_gib"] = args.gpu_cache_reserve_gib
    if getattr(args, "speaker_emb_path", None) is not None:
        cfg["model"]["speaker_emb_path"] = args.speaker_emb_path
    return cfg
