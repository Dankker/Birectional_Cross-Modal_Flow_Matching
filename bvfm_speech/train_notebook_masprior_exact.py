#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================================================
# Joint Training (Cut-manifest version)
# - Unified ASR/TTS with cut_manifest_all.jsonl
# - TTS branch uses cut_type == "tts"
# - ASR branch uses cut_type == "asr"
# - Keep full TTS / ASR demo with aligned full manifest
#
# REQUIRE:
#   pip install transformers torchaudio librosa scipy soundfile bigvgan
#   BigVGAN repo must be importable for meldataset.get_mel_spectrogram
# =========================================================

import os
import random
import json
import math
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl_rows(path, max_rows=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] JSON parse failed at line {line_idx}: {e}")
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def save_wav(path, wav, sr=24000):
    import soundfile as sf
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu()
        if wav.dim() == 3:
            wav = wav[0]
        if wav.dim() == 2 and wav.shape[0] == 1:
            wav = wav[0]
        wav = wav.float().numpy()
    else:
        wav = np.asarray(wav, dtype=np.float32)
    sf.write(path, wav, sr)
    print("wrote", path)


def extract_speaker_id_from_path(path: str):
    """LibriTTS common: .../<spk>/<chapter>/<utt>.wav"""
    p = path.replace("\\", "/")
    parts = p.split("/")
    for seg in parts:
        if seg.isdigit() and len(seg) >= 3:
            return seg
    m = re.search(r"/(\d{3,})/", p)
    if m:
        return m.group(1)
    return None


# -------------------------
# Char tokenizer (for CTC targets)
# -------------------------

class CharTokenizer:
    def __init__(self):
        self.stoi = {"<pad>": 0, "<unk>": 1, "<blank>": 2}
        self.itos = ["<pad>", "<unk>", "<blank>"]

    def build(self, texts):
        for s in texts:
            for ch in s:
                if ch not in self.stoi:
                    self.stoi[ch] = len(self.itos)
                    self.itos.append(ch)

    def encode(self, s):
        return [self.stoi.get(ch, 1) for ch in s]


def ctc_greedy_decode(logits_bkv, input_len, blank_id=2):
    preds = logits_bkv.argmax(dim=-1)  # [B,K]
    outs = []
    for b in range(preds.shape[0]):
        T = int(input_len[b])
        seq = preds[b, :T].tolist()
        out = []
        prev = None
        for x in seq:
            if x == blank_id:
                prev = x
                continue
            if prev is not None and x == prev:
                prev = x
                continue
            out.append(x)
            prev = x
        outs.append(out)
    return outs


def ids_to_text(ids, itos):
    chars = []
    for i in ids:
        if i < len(itos) and itos[i] not in ["<pad>", "<blank>"]:
            chars.append("?" if itos[i] == "<unk>" else itos[i])
    return "".join(chars)


# -------------------------
# Masked losses / stats
# -------------------------

def masked_mse(a, b, mask):
    mask = mask.float()
    diff2 = (a - b) ** 2
    diff2 = diff2 * mask.unsqueeze(-1)
    denom = mask.sum() * a.shape[-1] + 1e-8
    return diff2.sum() / denom


def masked_l1(a, b, mask):
    mask = mask.float()
    diff = (a - b).abs() * mask.unsqueeze(-1)
    denom = mask.sum() * a.shape[-1] + 1e-8
    return diff.sum() / denom


def masked_mse_per_sample(a, b, mask, eps=1e-8):
    m = mask.float().unsqueeze(-1)  # [B,K,1]
    diff2 = ((a - b) ** 2) * m
    num = diff2.sum(dim=(1, 2))
    denom = (m.sum(dim=(1, 2)) * a.shape[-1]).clamp_min(1.0)
    return num / (denom + eps)


def masked_mean_std(z, mask):
    m = mask.float().unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    mean = (z * m).sum(dim=1) / denom
    var = ((z - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
    std = torch.sqrt(var + 1e-6)
    return mean, std


def time_delta_bkd(x_bkd):
    return x_bkd[:, 1:, :] - x_bkd[:, :-1, :]


def mel_range_penalty(mel_bkD, maskK, mel_min=-11.5, mel_max=2.0):
    m = maskK.float().unsqueeze(-1)
    hi = F.relu(mel_bkD - mel_max) * m
    lo = F.relu(mel_min - mel_bkD) * m
    denom = m.sum() * mel_bkD.shape[-1] + 1e-8
    return (hi.sum() + lo.sum()) / denom


def masked_gaussian_nll(z, mu, logvar, mask):
    logvar = logvar.clamp(-6.0, 1.5)
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (((z - mu) ** 2) * inv_var + logvar)
    nll = nll * mask.float().unsqueeze(-1)
    denom = mask.float().sum() * z.shape[-1] + 1e-8
    return nll.sum() / denom


# =========================================================
# VF Lipschitz Upper-Bound (Finite Difference on v)
# =========================================================

def vf_lip_fd_ratio(vf, z_base, t, maskK, cfg_flag, spk_e, sigma=0.01, eps=1e-8):
    delta = torch.randn_like(z_base) * float(sigma)
    delta = delta * maskK.float().unsqueeze(-1)

    v0 = vf(z_base, t, maskK, cfg_flag=cfg_flag, spk_e=spk_e)
    v1 = vf(z_base + delta, t, maskK, cfg_flag=cfg_flag, spk_e=spk_e)

    num = torch.sqrt(masked_mse_per_sample(v1, v0, maskK, eps=eps) + eps)  # [B]
    den = torch.sqrt(masked_mse_per_sample(z_base + delta, z_base, maskK, eps=eps) + eps)  # [B]
    return num / (den + eps)


# =========================================================
# Bi-Lipschitz Diagnostics
# =========================================================

def masked_pair_l2(z_i, z_j, m_i, m_j, eps=1e-8):
    m = (m_i & m_j).float().unsqueeze(-1)
    diff2 = ((z_i - z_j) ** 2) * m
    denom = m.sum() * z_i.shape[-1] + eps
    return torch.sqrt(diff2.sum() / denom + eps)


def batch_pair_ratios(x_in, x_out, maskK, eps=1e-8):
    B = x_in.shape[0]
    ratios = []
    for i in range(B):
        for j in range(i + 1, B):
            din  = masked_pair_l2(x_in[i],  x_in[j],  maskK[i], maskK[j], eps=eps)
            dout = masked_pair_l2(x_out[i], x_out[j], maskK[i], maskK[j], eps=eps)
            ratios.append((dout / (din + eps)).detach())
    if len(ratios) == 0:
        return None
    return torch.stack(ratios)


def summarize_ratios(name, r: torch.Tensor):
    r = r.flatten()
    r_sorted, _ = torch.sort(r)
    n = r_sorted.numel()

    def q(p):
        idx = int(round((n - 1) * p))
        idx = max(0, min(idx, n - 1))
        return float(r_sorted[idx].item())

    print(
        f"[BI-LIP] {name} "
        f"min={float(r_sorted[0]):.4g} p10={q(0.10):.4g} p50={q(0.50):.4g} p90={q(0.90):.4g} max={float(r_sorted[-1]):.4g}"
    )


def amp_once(map_fn, x, maskK, sigma=0.001, eps=1e-8):
    noise = torch.randn_like(x) * sigma
    noise = noise * maskK.float().unsqueeze(-1)
    y0 = map_fn(x)
    y1 = map_fn(x + noise)
    num = torch.sqrt(masked_mse(y1, y0, maskK) + eps)
    den = torch.sqrt(masked_mse(x + noise, x, maskK) + eps)
    return float((num / (den + eps)).item())


# =========================================================
# MAS (Monotonic Alignment Search) - SAFE FP32
# =========================================================

def monotonic_alignment_search(log_probs, maskK, maskL, neg_inf=-1e4):
    device = log_probs.device
    B, K, L = log_probs.shape

    lp = log_probs.float()
    lp = lp.masked_fill(~maskK[:, :, None], neg_inf)
    lp = lp.masked_fill(~maskL[:, None, :], neg_inf)

    path = torch.full((B, K, L), neg_inf, device=device, dtype=torch.float32)
    path[:, 0, 0] = lp[:, 0, 0]

    for k in range(1, K):
        prev_same = path[:, k - 1, :]
        prev_move = F.pad(prev_same, (1, 0), value=neg_inf)[:, :-1]
        path[:, k, :] = torch.maximum(prev_same, prev_move) + lp[:, k, :]

    attn = torch.zeros((B, K, L), device=device, dtype=torch.float32)

    path_cpu = path.detach().cpu().numpy()
    maskK_cpu = maskK.detach().cpu().numpy()
    maskL_cpu = maskL.detach().cpu().numpy()

    for b in range(B):
        t_len = int(maskK_cpu[b].sum())
        l_len = int(maskL_cpu[b].sum())
        if t_len <= 0 or l_len <= 0:
            continue
        i = t_len - 1
        j = l_len - 1
        while i >= 0 and j >= 0:
            attn[b, i, j] = 1.0
            if i == 0:
                break
            stay = path_cpu[b, i - 1, j]
            move = path_cpu[b, i - 1, j - 1] if j > 0 else -1e18
            if j > 0 and move > stay:
                j -= 1
            i -= 1

    attn = attn * maskK[:, :, None].float() * maskL[:, None, :].float()
    return attn


def gaussian_mas_score(z_bkd, mu_bld, logvar_bld, maskK, maskL, score_temp=1.0):
    """
    z_bkd      : [B,K,D] normalized speech frames
    mu_bld     : [B,L,D] normalized token prior mean
    logvar_bld : [B,L,D] normalized token prior log-variance
    return     : [B,K,L] score for MAS
    """
    z = z_bkd[:, :, None, :]                          # [B,K,1,D]
    mu = mu_bld[:, None, :, :]                        # [B,1,L,D]
    logvar = logvar_bld[:, None, :, :].clamp(-6.0, 1.5)
    inv_var = torch.exp(-logvar)

    score = -0.5 * ((((z - mu) ** 2) * inv_var) + logvar).mean(dim=-1)  # [B,K,L]
    score = score * float(score_temp)

    score = score.masked_fill(~maskK[:, :, None], -1e4)
    score = score.masked_fill(~maskL[:, None, :], -1e4)
    return score


# =========================================================
# Anti-alias Downsample (stable, mask-safe)
# =========================================================

_AA_POS_KERNEL_CACHE = {}

def _make_gaussian_kernel1d(ksize: int, sigma: float, device):
    assert ksize % 2 == 1
    x = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize // 2)
    h = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    h = h / (h.sum() + 1e-12)
    return h


def downsample_time_bkd(z_bkd, mask_bk, factor: int):
    """
    Anti-aliased downsample along time axis for masked sequences.
    z_bkd: [B,K,D]
    mask_bk: [B,K] bool
    """
    assert factor >= 1
    if factor == 1:
        k_list = mask_bk.sum(dim=1).tolist()
        return z_bkd, mask_bk, k_list

    B, K, D = z_bkd.shape
    device = z_bkd.device

    K_ds = int(math.ceil(K / factor))
    K_pad = K_ds * factor
    if K_pad != K:
        z_pad = torch.zeros(B, K_pad, D, device=device, dtype=z_bkd.dtype)
        m_pad = torch.zeros(B, K_pad, device=device, dtype=torch.bool)
        z_pad[:, :K] = z_bkd
        m_pad[:, :K] = mask_bk
    else:
        z_pad = z_bkd
        m_pad = mask_bk

    if factor == 2:
        key = ("binom5", device)
        if key not in _AA_POS_KERNEL_CACHE:
            h = torch.tensor([1, 4, 6, 4, 1], device=device, dtype=torch.float32)
            h = h / h.sum()
            _AA_POS_KERNEL_CACHE[key] = h
        h = _AA_POS_KERNEL_CACHE[key]
    else:
        ksize = int(2 * factor * 3 + 1)
        ksize = max(7, min(ksize, 31))
        sigma = 0.6 * factor
        key = ("gauss", factor, ksize, float(sigma), device)
        if key not in _AA_POS_KERNEL_CACHE:
            _AA_POS_KERNEL_CACHE[key] = _make_gaussian_kernel1d(ksize, sigma, device)
        h = _AA_POS_KERNEL_CACHE[key]

    ksize = int(h.numel())
    pad = ksize // 2

    z = z_pad.float()
    m = m_pad.float()

    x = (z * m.unsqueeze(-1)).transpose(1, 2)                  # [B,D,K_pad]
    w = h.view(1, 1, ksize).repeat(D, 1, 1)                    # [D,1,ksize]
    num = F.conv1d(x, w, stride=factor, padding=pad, groups=D) # [B,D,K_ds]
    den = F.conv1d(m.unsqueeze(1), h.view(1, 1, ksize), stride=factor, padding=pad)  # [B,1,K_ds]

    den_safe = den.clamp_min(1e-3)
    z_ds = (num / den_safe).transpose(1, 2).to(dtype=z_bkd.dtype)  # [B,K_ds,D]

    mask_ds = (den.squeeze(1) > 0.5)
    k_list_ds = mask_ds.sum(dim=1).tolist()
    return z_ds, mask_ds, k_list_ds


# =========================================================
# Multi-Resolution STFT Loss (waveform)
# =========================================================

def _stft_loss_1(wav_hat, wav_gt, fft_size, hop, win, eps=1e-7):
    window = torch.hann_window(win, device=wav_hat.device, dtype=wav_hat.dtype)
    Sh = torch.stft(wav_hat, n_fft=fft_size, hop_length=hop, win_length=win,
                    window=window, center=True, return_complex=True)
    Sg = torch.stft(wav_gt,  n_fft=fft_size, hop_length=hop, win_length=win,
                    window=window, center=True, return_complex=True)
    Mh = Sh.abs()
    Mg = Sg.abs()
    sc = (Mg - Mh).norm(p='fro') / (Mg.norm(p='fro') + eps)
    lm = (torch.log(Mg + eps) - torch.log(Mh + eps)).abs().mean()
    return sc + lm


def mrstft_loss(wav_hat, wav_gt, eps=1e-7):
    cfgs = [
        (1024, 256, 1024),
        (2048, 512, 2048),
        ( 512, 128,  512),
    ]
    loss = 0.0
    for (fft_size, hop, win) in cfgs:
        loss = loss + _stft_loss_1(wav_hat, wav_gt, fft_size, hop, win, eps=eps)
    return loss / len(cfgs)


# =========================================================
# Modules
# =========================================================

class ResidualAdapter(nn.Module):
    def __init__(self, dim, bottleneck=192, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bottleneck)
        self.fc2 = nn.Linear(bottleneck, dim)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.ln(x)
        h = F.silu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class FastSpeech2DurationPredictor(nn.Module):
    def __init__(self, D=80, hidden=256, ksize=3, dropout=0.5):
        super().__init__()
        pad = ksize // 2
        self.conv1 = nn.Conv1d(D, hidden, kernel_size=ksize, padding=pad)
        self.ln1   = nn.LayerNorm(hidden)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=ksize, padding=pad)
        self.ln2   = nn.LayerNorm(hidden)
        self.drop2 = nn.Dropout(dropout)

        self.proj  = nn.Linear(hidden, 1)

    def forward(self, h_tok, maskL):
        x = h_tok.transpose(1, 2)
        x = self.conv1(x).transpose(1, 2)
        x = F.relu(x)
        x = self.ln1(x)
        x = self.drop1(x)

        x = x.transpose(1, 2)
        x = self.conv2(x).transpose(1, 2)
        x = F.relu(x)
        x = self.ln2(x)
        x = self.drop2(x)

        log_dur = self.proj(x).squeeze(-1)
        log_dur = log_dur * maskL.float()
        return log_dur


class LengthPredictor(nn.Module):
    def __init__(self, D=80, hidden=192):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_tok, maskL):
        w = maskL.float()
        pooled = (h_tok * w.unsqueeze(-1)).sum(dim=1)
        k = self.mlp(pooled).squeeze(-1)
        return F.softplus(k) + 1.0


class FrameCTCConvHead(nn.Module):
    def __init__(self, V, D=80, hidden=256, layers=4, ksize=5):
        super().__init__()
        self.inp = nn.Linear(D, hidden)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel_size=ksize, padding=ksize // 2)
            for _ in range(layers)
        ])
        self.out = nn.Linear(hidden, V)

    def forward(self, z, maskK):
        x = self.inp(z)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = F.gelu(conv(x)) + x
        x = x.transpose(1, 2)
        return self.out(x)


def masked_mean_pool(x_bld, mask_bl, eps=1e-6):
    if mask_bl is None:
        return x_bld.mean(dim=1)
    w = mask_bl.float().unsqueeze(-1)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (x_bld * w).sum(dim=1) / (denom + eps)


class AdaLNFiLM(nn.Module):
    def __init__(self, channels, cond_dim, zero_init=True):
        super().__init__()
        self.ln = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.to_ss = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * channels),
        )
        if zero_init:
            nn.init.zeros_(self.to_ss[-1].weight)
            nn.init.zeros_(self.to_ss[-1].bias)

    def forward(self, h_bct, c_bc):
        h_btc = h_bct.transpose(1, 2)
        h_btc = self.ln(h_btc)
        shift, scale = self.to_ss(c_bc).chunk(2, dim=1)
        h_btc = h_btc * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h_btc.transpose(1, 2)


class CondResBlock1D(nn.Module):
    def __init__(self, ch, cond_dim, ksize=5, dilation=1, dropout=0.0):
        super().__init__()
        pad = (ksize // 2) * dilation
        self.film1 = AdaLNFiLM(ch, cond_dim, zero_init=True)
        self.c1 = nn.Conv1d(ch, ch, ksize, padding=pad, dilation=dilation)
        self.film2 = AdaLNFiLM(ch, cond_dim, zero_init=True)
        self.c2 = nn.Conv1d(ch, ch, ksize, padding=pad, dilation=dilation)
        self.drop = float(dropout)

    def forward(self, x_bct, c_bc):
        h = self.film1(x_bct, c_bc)
        h = F.gelu(self.c1(h))
        h = F.dropout(h, p=self.drop, training=self.training)
        h = self.film2(x_bct + h, c_bc)
        h = self.c2(h)
        h = F.dropout(h, p=self.drop, training=self.training)
        return x_bct + h


class TextCondPostNet(nn.Module):
    def __init__(self, D=80, hidden=512, cond_dim=768, ksize=5, n_layers=5, dropout=0.2):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_ch  = D if i == 0 else hidden
            out_ch = D if i == n_layers - 1 else hidden
            self.convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=ksize, padding=ksize // 2))

        self.films = nn.ModuleList([
            AdaLNFiLM(hidden, cond_dim, zero_init=True) for _ in range(n_layers - 1)
        ])

        nn.init.zeros_(self.convs[-1].weight)
        nn.init.zeros_(self.convs[-1].bias)

    def forward(self, y_bkd, c_bc):
        x = y_bkd.transpose(1, 2)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs) - 1:
                x = self.films[i](x, c_bc)
                x = torch.tanh(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.transpose(1, 2)


class TextCondRefiner1xResidualPostNet(nn.Module):
    """
    1x refiner: same-length residual + postnet.
    """
    def __init__(self, D=80, hidden=512, cond_dim=768, n_blocks=8, ksize=7, dropout=0.1):
        super().__init__()
        self.D = D
        self.hidden = hidden
        self.cond_dim = cond_dim

        self.inp = nn.Conv1d(D, hidden, kernel_size=1)
        dilas = [1, 2, 4, 8, 16, 1, 2, 4][:n_blocks]
        self.blocks = nn.ModuleList([
            CondResBlock1D(hidden, cond_dim, ksize=ksize, dilation=d, dropout=dropout)
            for d in dilas
        ])
        self.out = nn.Conv1d(hidden, D, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.postnet = TextCondPostNet(D=D, hidden=hidden, cond_dim=cond_dim, ksize=5, n_layers=5, dropout=0.2)

    def forward(self, z_bkd, cond=None, cond_mask=None):
        B, K, D = z_bkd.shape
        device = z_bkd.device

        if cond is None:
            c = torch.zeros(B, self.cond_dim, device=device, dtype=z_bkd.dtype)
        else:
            if cond.dim() == 3:
                c = masked_mean_pool(cond, cond_mask).to(device=device, dtype=z_bkd.dtype)
            else:
                c = cond.to(device=device, dtype=z_bkd.dtype)

        x = z_bkd.transpose(1, 2)   # [B,D,K]
        h = F.gelu(self.inp(x))
        for blk in self.blocks:
            h = blk(h, c)
        res = self.out(h).transpose(1, 2)
        y = z_bkd + res
        y = y + self.postnet(y, c)
        return y


class SpeakerTable(nn.Module):
    def __init__(self, n_spk: int, E: int, scale: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding(n_spk, E)
        self.ln = nn.LayerNorm(E)
        self.scale = float(scale)
        nn.init.zeros_(self.emb.weight)

    def forward(self, spk_id: torch.LongTensor):
        e = self.emb(spk_id)
        e = self.ln(e) * self.scale
        return e


class TextPriorHead(nn.Module):
    """
    Token-wise stochastic prior in NORMALIZED mel space.
    """
    def __init__(self, in_dim, hidden=256, out_dim=80, logvar_bias=-2.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden, out_dim)
        self.logvar = nn.Linear(hidden, out_dim)
        nn.init.constant_(self.logvar.bias, float(logvar_bias))

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-6.0, 1.5)
        return mu, logvar


# =========================================================
# DiT Vector Field (single VF) + CFG + Speaker conditioning
# =========================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SDPASelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, cos, sin, key_padding_mask=None):
        B, K, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, K, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, K), device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(B, K, C)
        return self.out_proj(y)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SDPASelfAttention(hidden_size, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, cos, sin, key_padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(attn_in, cos, sin, key_padding_mask=key_padding_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTVectorField(nn.Module):
    def __init__(self, D=80, E_spk=16, hidden=512, depth=8, n_heads=8, dropout=0.0, max_len=4096):
        super().__init__()
        self.D = D
        self.E_spk = E_spk
        self.in_proj = nn.Linear(D, hidden)
        self.t_embedder = TimestepEmbedder(hidden)

        self.cfg_embedder = nn.Embedding(2, hidden)
        nn.init.zeros_(self.cfg_embedder.weight)

        self.spk_proj = nn.Sequential(
            nn.Linear(E_spk, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        head_dim = hidden // n_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_len)

        self.blocks = nn.ModuleList([DiTBlock(hidden, n_heads, mlp_ratio=4.0, dropout=dropout) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden, D)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for blk in self.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, z, t, maskK, cfg_flag=None, spk_e=None):
        B, K, _ = z.shape
        x = self.in_proj(z)
        c = self.t_embedder(t)

        if cfg_flag is None:
            cfg_flag = torch.ones(B, dtype=torch.long, device=z.device)
        c = c + self.cfg_embedder(cfg_flag)

        if spk_e is None:
            spk_e = torch.zeros(B, self.E_spk, device=z.device, dtype=z.dtype)
        c = c + self.spk_proj(spk_e.to(dtype=c.dtype))

        cos, sin = self.rope(x, K)
        key_padding_mask = ~maskK
        for blk in self.blocks:
            x = blk(x, c, cos, sin, key_padding_mask=key_padding_mask)
        return self.final_layer(x, c)


# =========================================================
# Integrators (Euler/Heun)
# =========================================================

@torch.no_grad()
def heun_integrate(vf, z0, maskK, steps=30, direction=+1, cfg_scale=1.0, spk_e=None):
    z = z0
    dt = direction * (1.0 / steps)
    t0 = 0.0 if direction == +1 else 1.0
    B = z.shape[0]
    device = z.device

    if spk_e is None:
        spk_e = torch.zeros(B, vf.E_spk, device=device, dtype=z.dtype)

    cfg_cond = torch.ones(B, dtype=torch.long, device=device)
    cfg_un   = torch.zeros(B, dtype=torch.long, device=device)
    spk_zero = torch.zeros_like(spk_e)

    def v_eval(z_now, t_now, cfg_flag, spk):
        t_tensor = torch.full((B,), float(t_now), device=device)
        return vf(z_now, t_tensor, maskK, cfg_flag=cfg_flag, spk_e=spk)

    def v_mix(z_now, t_now):
        if cfg_scale == 1.0:
            is_un = (spk_e.abs().sum(dim=1) < 1e-8)
            cfg = torch.where(is_un, cfg_un, cfg_cond)
            spk = torch.where(is_un[:, None], spk_zero, spk_e)
            return v_eval(z_now, t_now, cfg, spk)

        v_c = v_eval(z_now, t_now, cfg_cond, spk_e)
        v_u = v_eval(z_now, t_now, cfg_un,   spk_zero)
        return v_u + cfg_scale * (v_c - v_u)

    for i in range(steps):
        t = t0 + i * dt
        t_next = t + dt
        k1 = v_mix(z, t)
        z_pred = z + dt * k1
        k2 = v_mix(z_pred, t_next)
        z = z + dt * 0.5 * (k1 + k2)
    return z


def euler_integrate_grad(vf, z0, maskK, steps=10, direction=+1, cfg_flag_value=1, spk_e=None):
    z = z0
    dt = direction * (1.0 / steps)
    t0 = 0.0 if direction == +1 else 1.0
    B = z.shape[0]
    device = z.device

    if spk_e is None:
        spk_e = torch.zeros(B, vf.E_spk, device=device, dtype=z.dtype)
    cfg_flag = torch.full((B,), int(cfg_flag_value), dtype=torch.long, device=device)

    for i in range(steps):
        t = t0 + i * dt
        t_tensor = torch.full((B,), float(t), device=device)
        v = vf(z, t_tensor, maskK, cfg_flag=cfg_flag, spk_e=spk_e)
        z = z + dt * v
    return z


@torch.no_grad()
def euler_integrate(vf, z0, maskK, steps=10, direction=+1, cfg_flag_value=1, spk_e=None):
    z = z0
    dt = direction * (1.0 / steps)
    t0 = 0.0 if direction == +1 else 1.0
    B = z.shape[0]
    device = z.device

    if spk_e is None:
        spk_e = torch.zeros(B, vf.E_spk, device=device, dtype=z.dtype)
    cfg_flag = torch.full((B,), int(cfg_flag_value), dtype=torch.long, device=device)

    for i in range(steps):
        t = t0 + i * dt
        t_tensor = torch.full((B,), float(t), device=device)
        v = vf(z, t_tensor, maskK, cfg_flag=cfg_flag, spk_e=spk_e)
        z = z + dt * v
    return z


# =========================================================
# Frozen SpeechT5 text encoder wrapper (layer-select)
# =========================================================

class FrozenSpeechT5TextEncoder(nn.Module):
    def __init__(self, model_name="microsoft/speecht5_tts", device="cuda", layer_idx=-1):
        super().__init__()
        print(f"Loading Frozen {model_name} ...")
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.layer_idx = int(layer_idx)
        for p in self.model.parameters():
            p.requires_grad = False

        H = None
        for attr in ["hidden_size", "d_model", "encoder_hidden_size"]:
            if hasattr(self.model.config, attr):
                H = int(getattr(self.model.config, attr))
                break
        self.hidden_size = H if H is not None else 768
        try:
            self.pad_id = int(self.processor.tokenizer.pad_token_id)
        except Exception:
            self.pad_id = 1
        print(f"SpeechT5 hidden: {self.hidden_size} pad: {self.pad_id} layer_idx={self.layer_idx}")

    @torch.no_grad()
    def forward(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        try:
            if hasattr(self.model, "speecht5") and hasattr(self.model.speecht5, "encoder"):
                enc = self.model.speecht5.encoder
                out = enc(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hs = out.hidden_states
                h = hs[self.layer_idx] if hs is not None else out.last_hidden_state
                return h.float(), attention_mask.bool()
        except Exception:
            pass

        try:
            B = input_ids.shape[0]
            mel_bins = int(getattr(self.model.config, "num_mel_bins", 80))
            decoder_input_values = torch.zeros((B, 1, mel_bins), device=self.device)
            spk_dim = int(getattr(self.model.config, "speaker_embedding_dim", 512))
            speaker_embeddings = torch.zeros((B, spk_dim), device=self.device)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_values=decoder_input_values,
                speaker_embeddings=speaker_embeddings,
                output_hidden_states=True,
                return_dict=True,
            )
            if hasattr(out, "encoder_outputs") and out.encoder_outputs is not None:
                eo = out.encoder_outputs
                if hasattr(eo, "hidden_states") and eo.hidden_states is not None:
                    h = eo.hidden_states[self.layer_idx]
                    return h.float(), attention_mask.bool()
                if hasattr(eo, "last_hidden_state"):
                    return eo.last_hidden_state.float(), attention_mask.bool()
            if hasattr(out, "encoder_last_hidden_state") and out.encoder_last_hidden_state is not None:
                return out.encoder_last_hidden_state.float(), attention_mask.bool()
        except Exception as e:
            raise RuntimeError(f"Could not extract SpeechT5 text encoder hidden states: {repr(e)}")

        raise RuntimeError("Could not extract SpeechT5 text encoder hidden states (no valid path).")


# =========================================================
# Demo helper: durations -> hard expand to frames
# =========================================================

def durations_to_int_and_fixsum(dur_float, maskL, K_target):
    d = dur_float[0].detach().float().cpu()
    m = maskL[0].detach().cpu()
    L = d.numel()
    dur_int = torch.zeros(L, dtype=torch.long)
    for i in range(L):
        if not bool(m[i]):
            dur_int[i] = 0
        else:
            dur_int[i] = max(1, int(torch.round(d[i]).item()))
    s = int(dur_int.sum().item())
    if s <= 0:
        for i in range(L):
            if bool(m[i]):
                dur_int[i] = 1
        s = int(dur_int.sum().item())
    last = None
    for i in range(L - 1, -1, -1):
        if bool(m[i]):
            last = i
            break
    if last is None:
        return dur_int, s
    diff = K_target - s
    dur_int[last] = max(1, int(dur_int[last].item() + diff))
    return dur_int, int(dur_int.sum().item())


# =========================================================
# MAIN
# =========================================================

def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ------------ USER SETTINGS ------------
    cut_manifest = os.environ.get("CUT_MANIFEST", r"/work/dankker0900/dataset/cut_manifests/cut_manifest_all.jsonl")
    aligned_manifest = os.environ.get("ALIGNED_MANIFEST", r"/work/dankker0900/dataset/align/train_manifest_aligned.jsonl")
    max_cut_rows = None if os.environ.get("MAX_CUT_ROWS", "") == "" else int(os.environ["MAX_CUT_ROWS"])

    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    batch_size_tts = batch_size
    batch_size_asr = batch_size

    ds_factor = 1
    ds_align = 1

    TOP_K_SPK = 1000
    TARGET_SPKS = None   # or None to use top speakers from cut manifest

    E_spk = 64
    spk_scale = 0.2
    spk_drop_rate = 0.3

    st5_layer_idx = -2
    use_adapter = True
    adapter_bottleneck = 192
    adapter_dropout = 0.1

    bigvgan_name = "nvidia/bigvgan_24khz_100band"

    mel_floor = -11.5
    mel_ceil  =  2.0

    total_steps = int(os.environ.get("TOTAL_STEPS", "150000"))
    lr_all = 1e-4

    ode_steps_eval = 8
    ode_steps_endloss = 4

    demo_cfg_scale = 1.3
    demo_prior_temp = 0.0

    w_end = 0.30
    w_ctc_hat = 0.8
    w_ctc_T = 0.4
    w_ctc_start = 200

    w_tts_mel = 1.3
    w_ref = 0.5
    w_delta = 0.0
    w_mel_range = 0.0

    w_align_start = 0
    w_dur = 0.1
    w_len = 0.05
    mas_temp = 1.0

    w_prior = 0.5

    use_stat_match = True
    w_stat = 0.05

    use_stft = False
    w_stft = 0.2
    wav_cache_max_items = 64

    stats_max_unique_wavs = 5000
    mel_cache_max_items = 512

    demo_every = int(os.environ.get("DEMO_EVERY", "1000"))
    demo_dir = os.environ.get("DEMO_DIR", r"/work/dankker0900/biflow_repo_cutmanifest_notebook_masprior/demos_bigvgan_notebook_exact")
    os.makedirs(demo_dir, exist_ok=True)

    enable_bilip_diag = True
    diag_every = 500
    amp_sigma = 0.01

    enable_vf_lip = False
    vf_lip_start = 5000
    w_vf_lip = 0.2
    vf_lip_L_hi = 1.0
    vf_lip_sigma = 0.01
    vf_lip_every = 1
    vf_lip_print_every = 200

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # -------------------------
    # BigVGAN load
    # -------------------------
    bigvgan_model = None
    bigvgan_h = None
    try:
        import bigvgan
        from meldataset import get_mel_spectrogram
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        for p in bigvgan_model.parameters():
            p.requires_grad_(False)
        bigvgan_h = bigvgan_model.h
        print("Loaded BigVGAN:", bigvgan_name)
    except Exception as e:
        print("WARNING: could not load BigVGAN:", repr(e))
        bigvgan_model = None
        bigvgan_h = None

    assert bigvgan_h is not None, "BigVGAN mel config not available."

    sampling_rate = int(bigvgan_h.sampling_rate)
    num_mels = int(bigvgan_h.num_mels)
    hop_size = int(bigvgan_h.hop_size)
    D_mel = num_mels

    import librosa

    # -------------------------
    # Read cut manifest
    # -------------------------
    cut_rows_all = read_jsonl_rows(cut_manifest, max_rows=max_cut_rows)
    assert len(cut_rows_all) > 0, "cut_manifest_all.jsonl is empty or not found."

    spk_count = {}
    for r in cut_rows_all:
        spk = str(r.get("speaker", ""))
        if len(spk) == 0:
            continue
        spk_count[spk] = spk_count.get(spk, 0) + 1
    assert len(spk_count) > 0, "No speaker field found in cut manifest."

    if TARGET_SPKS is None:
        spk_sorted = sorted(spk_count.items(), key=lambda kv: kv[1], reverse=True)
        target_spks = [s for (s, c) in spk_sorted[:TOP_K_SPK]]
    else:
        target_spks = list(TARGET_SPKS)

    target_spk_set = set(target_spks)
    cut_rows = [r for r in cut_rows_all if str(r.get("speaker", "")) in target_spk_set]
    assert len(cut_rows) > 0, "No cut rows left after speaker filtering."

    print("Selected speakers:", target_spks)
    print(f"Filtered cut rows: {len(cut_rows)} / {len(cut_rows_all)}")

    tts_pool = [r for r in cut_rows if r.get("cut_type", "") == "tts"]
    asr_pool = [r for r in cut_rows if r.get("cut_type", "") == "asr"]

    assert len(tts_pool) > 0, "tts_pool is empty."
    assert len(asr_pool) > 0, "asr_pool is empty."

    print("tts_pool =", len(tts_pool))
    print("asr_pool =", len(asr_pool))

    spk_list = sorted(list({str(r["speaker"]) for r in cut_rows}))
    spk2id = {s: i for i, s in enumerate(spk_list)}
    n_spk = len(spk_list)
    print("n_spk =", n_spk, "spk_list(head) =", spk_list[:10])

    # -------------------------
    # Full aligned manifest for demo / eval
    # -------------------------
    aligned_rows_all = read_jsonl_rows(aligned_manifest, max_rows=None)
    assert len(aligned_rows_all) > 0, "aligned_manifest.jsonl is empty or not found."

    aligned_rows = []
    for r in aligned_rows_all:
        spk = extract_speaker_id_from_path(r["wav"])
        if spk in target_spk_set:
            aligned_rows.append(r)

    assert len(aligned_rows) > 0, "No aligned full rows left after speaker filtering."
    print("aligned_rows(for demo) =", len(aligned_rows))

    # -------------------------
    # CTC char vocab from cut texts
    # -------------------------
    texts_all = list({
        str(r.get("text_norm", r.get("text", ""))).strip()
        for r in cut_rows
        if len(str(r.get("text_norm", r.get("text", ""))).strip()) > 0
    })

    tok = CharTokenizer()
    tok.build(texts_all)
    Vt = len(tok.itos)
    UNK_ID = tok.stoi["<unk>"]
    BLANK_ID = tok.stoi["<blank>"]
    print("char vocab size:", Vt, "blank_id:", BLANK_ID)

    alpha_sample_rows = tts_pool if len(tts_pool) <= 20000 else random.sample(tts_pool, 20000)
    ratios = []
    for r in alpha_sample_rows:
        txt = str(r.get("text_norm", r.get("text", ""))).strip()
        ratios.append(float(r["cut_mel_len"]) / max(len(txt), 1))
    alpha_K = float(np.mean(ratios)) if len(ratios) > 0 else 1.0
    print("alpha_K ≈ mean(cut_mel_len/len(text)) =", alpha_K)

    # -------------------------
    # Frozen SpeechT5
    # -------------------------
    st5 = FrozenSpeechT5TextEncoder(
        model_name="microsoft/speecht5_tts",
        device=device,
        layer_idx=st5_layer_idx
    )
    H_text = st5.hidden_size

    # -------------------------
    # Full wav / mel caches
    # -------------------------
    wav_cache = OrderedDict()
    mel_cache = OrderedDict()

    def load_wav_full_cached(path: str):
        nonlocal wav_cache
        if path in wav_cache:
            wav_cache.move_to_end(path)
            return wav_cache[path]

        wav_np, _ = librosa.load(path, sr=sampling_rate, mono=True)
        rem = wav_np.shape[0] % hop_size
        if rem != 0:
            wav_np = wav_np[:-rem]
        wav_np = librosa.util.normalize(wav_np) * 0.95
        wav_np = wav_np.astype(np.float32)

        wav_cache[path] = wav_np
        wav_cache.move_to_end(path)
        while len(wav_cache) > int(wav_cache_max_items):
            wav_cache.popitem(last=False)
        return wav_np

    def load_logmel_full_cached(path: str):
        nonlocal mel_cache
        if path in mel_cache:
            mel_cache.move_to_end(path)
            return mel_cache[path]

        wav_np = load_wav_full_cached(path)
        wav = torch.FloatTensor(wav_np).unsqueeze(0).to(device)
        mel = get_mel_spectrogram(wav, bigvgan_h)      # [1, D, K]
        mel_cpu = mel[0].detach().cpu().float().transpose(0, 1).contiguous()  # [K, D]

        mel_cache[path] = mel_cpu
        mel_cache.move_to_end(path)
        while len(mel_cache) > int(mel_cache_max_items):
            mel_cache.popitem(last=False)
        return mel_cpu

    # -------------------------
    # GLOBAL mu/std on log-mel frames
    # -------------------------
    print("Computing global mel mean/std from unique parent wavs ...")
    unique_wavs = list(OrderedDict.fromkeys([r["parent_wav"] for r in cut_rows]).keys())
    if stats_max_unique_wavs is not None and len(unique_wavs) > stats_max_unique_wavs:
        random.shuffle(unique_wavs)
        unique_wavs = unique_wavs[:stats_max_unique_wavs]

    count = 0
    mean = np.zeros((D_mel,), dtype=np.float64)
    M2   = np.zeros((D_mel,), dtype=np.float64)

    for wav_path in unique_wavs:
        x = load_logmel_full_cached(wav_path).numpy().astype(np.float64)   # [K,D]
        if x.size == 0:
            continue
        b_n = x.shape[0]
        b_mean = x.mean(axis=0)
        b_var = x.var(axis=0, ddof=0)
        if count == 0:
            mean = b_mean
            M2 = b_var * b_n
            count = b_n
        else:
            delta = b_mean - mean
            tot = count + b_n
            mean = mean + delta * (b_n / tot)
            M2 = M2 + b_var * b_n + (delta * delta) * (count * b_n / tot)
            count = tot

    var = M2 / max(count, 1)
    std = np.sqrt(var + 1e-8)
    std = np.maximum(std, 0.2)

    mu_g = torch.tensor(mean, dtype=torch.float32).view(1, 1, D_mel)
    std_g = torch.tensor(std, dtype=torch.float32).view(1, 1, D_mel)

    mu_b = mu_g.to(device)
    std_b = std_g.to(device)

    print("global mu/std ready:", tuple(mu_b.shape), tuple(std_b.shape))

    # -------------------------
    # Cut-batch sampler
    # -------------------------
    def sample_cut_rows(pool, bs):
        if len(pool) >= bs:
            return random.sample(pool, bs)
        return random.choices(pool, k=bs)

    def build_batch_from_cut_rows(batch_rows):
        z_list = []
        K_list = []
        texts_b = []
        spk_ids = []
        wav_paths = []
        starts = []
        ends = []

        for row in batch_rows:
            wav_path = row["parent_wav"]
            s = int(row["cut_start_mel"])
            e = int(row["cut_end_mel"])
            txt = str(row.get("text_norm", row.get("text", ""))).strip()
            spk = str(row["speaker"])

            mel_full = load_logmel_full_cached(wav_path)  # [K,D] cpu
            K0 = int(mel_full.shape[0])

            s = max(0, min(s, K0 - 1))
            e = max(s + 1, min(e, K0))
            mel_seg = mel_full[s:e].float()

            z_list.append(mel_seg)
            K_list.append(int(mel_seg.shape[0]))
            texts_b.append(txt)
            spk_ids.append(int(spk2id[spk]))
            wav_paths.append(wav_path)
            starts.append(s)
            ends.append(e)

        B = len(z_list)
        Kmax = int(max(K_list))
        zS_log = torch.zeros(B, Kmax, D_mel, dtype=torch.float32)
        maskK = torch.zeros(B, Kmax, dtype=torch.bool)

        for b in range(B):
            K = int(z_list[b].shape[0])
            zS_log[b, :K] = z_list[b]
            maskK[b, :K] = True

        return (
            zS_log.to(device),                  # [B,K,D] log-mel
            maskK.to(device),                   # [B,K]
            K_list,                             # list[int]
            texts_b,                            # list[str]
            torch.tensor(spk_ids, dtype=torch.long, device=device),
            wav_paths,                          # list[str]
            starts,                             # list[int]
            ends                                # list[int]
        )

    # -------------------------
    # Models
    # -------------------------
    adapter = ResidualAdapter(H_text, bottleneck=adapter_bottleneck, dropout=adapter_dropout).to(device) if use_adapter else None
    text_prior = TextPriorHead(in_dim=H_text, hidden=256, out_dim=D_mel, logvar_bias=-2.0).to(device)
    dur_pred = FastSpeech2DurationPredictor(D=H_text, hidden=256, ksize=3, dropout=0.5).to(device)
    len_pred = LengthPredictor(D=H_text, hidden=192).to(device)

    spk_table = SpeakerTable(n_spk=n_spk, E=E_spk, scale=spk_scale).to(device)
    vf = DiTVectorField(D=D_mel, E_spk=E_spk, hidden=1024, depth=12, n_heads=16, dropout=0.0, max_len=4096).to(device)

    mel_refiner = TextCondRefiner1xResidualPostNet(
        D=D_mel,
        hidden=512,
        cond_dim=H_text,
        n_blocks=8,
        ksize=7,
        dropout=0.1
    ).to(device)

    text_ctc_head = FrameCTCConvHead(V=Vt, D=D_mel, hidden=256, layers=4, ksize=5).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    opt_params = (
        list(text_prior.parameters()) +
        list(dur_pred.parameters()) +
        list(len_pred.parameters()) +
        list(spk_table.parameters()) +
        list(vf.parameters()) +
        list(text_ctc_head.parameters()) +
        list(mel_refiner.parameters())
    )
    if adapter is not None:
        opt_params += list(adapter.parameters())

    opt = torch.optim.AdamW(opt_params, lr=lr_all)

    print("\n=== Single-VF CUT-MANIFEST version ===")
    print(f"[RATE] ds_factor(model)={ds_factor}  ds_align(MAS)={ds_align}")
    print(f"[SPEAKER] n_spk={n_spk} E_spk={E_spk} spk_scale={spk_scale} spk_drop_rate={spk_drop_rate}")
    print(f"[SpeechT5] layer_idx={st5_layer_idx} adapter={use_adapter}")
    print(f"[MAS] Gaussian score temp={mas_temp}")
    print(f"[PRIOR] stochastic prior w_prior={w_prior}")
    print(f"[STAT] use={use_stat_match} w_stat={w_stat}")
    print(f"[STFT] use={use_stft} w_stft={w_stft} (refiner-only via detach)")
    print(f"[VF-LIP-FD] enable={enable_vf_lip} start={vf_lip_start} w={w_vf_lip} L_hi={vf_lip_L_hi} sigma={vf_lip_sigma}")

    # =========================================================
    # Helper fns for training
    # =========================================================
    def encode_text_batch(texts):
        with torch.no_grad():
            h_st5, maskL = st5(texts)

        h_enc = h_st5
        if adapter is not None:
            h_enc = adapter(h_enc)

        mu_tok, logvar_tok = text_prior(h_enc)
        return h_enc, maskL, mu_tok, logvar_tok

    def build_local_prior_batch(
        zS_log, maskK, h_enc, maskL, mu_tok, logvar_tok,
        supervise_dur_len=False
    ):
        zS = (zS_log - mu_b) / std_b

        zS_align, maskK_align, _ = downsample_time_bkd(zS, maskK, ds_align)

        with torch.no_grad():
            score = gaussian_mas_score(
                zS_align.float(),
                mu_tok.float(),
                logvar_tok.float(),
                maskK_align,
                maskL,
                score_temp=mas_temp
            )
            attn = monotonic_alignment_search(score, maskK_align, maskL, neg_inf=-1e4)

        mu_align = torch.bmm(attn, mu_tok.float())
        logvar_align = torch.bmm(attn, logvar_tok.float())

        B, Kmax, D = zS.shape
        mu_full = torch.zeros(B, Kmax, D, device=device, dtype=zS.dtype)
        logvar_full = torch.zeros(B, Kmax, D, device=device, dtype=zS.dtype)

        for b in range(B):
            K0 = int(maskK[b].sum().item())
            Ka = int(maskK_align[b].sum().item())
            if Ka <= 0 or K0 <= 0:
                continue
            mu_up = mu_align[b, :Ka].repeat_interleave(ds_align, dim=0)[:K0]
            lv_up = logvar_align[b, :Ka].repeat_interleave(ds_align, dim=0)[:K0]
            mu_full[b, :K0] = mu_up.to(dtype=zS.dtype)
            logvar_full[b, :K0] = lv_up.to(dtype=zS.dtype)

        eps = torch.randn_like(mu_full)
        zT_sample = mu_full + torch.exp(0.5 * logvar_full.clamp(-6.0, 1.5)) * eps
        zT_sample = zT_sample * maskK.float().unsqueeze(-1)
        zT_mean = mu_full

        loss_dur = torch.tensor(0.0, device=device)
        loss_len = torch.tensor(0.0, device=device)

        if supervise_dur_len:
            h_dp = h_enc.detach()
            log_dur_pred = dur_pred(h_dp, maskL)
            k_pred = len_pred(h_dp, maskL)

            dur_gt_align = attn.sum(dim=1).float()
            K_tar = maskK.sum(dim=1).float()
            K_align_cnt = maskK_align.sum(dim=1).float()
            K_est_full = (K_align_cnt * float(ds_align)).clamp_min(1.0)
            scale = (K_tar / K_est_full).unsqueeze(1)
            dur_gt_full = dur_gt_align * float(ds_align) * scale

            log_dur_gt_full = torch.log(dur_gt_full + 1.0)
            denomL = maskL.float().sum().clamp_min(1.0)

            loss_dur = (((log_dur_pred - log_dur_gt_full) ** 2) * maskL.float()).sum() / denomL

            dur_pred_lin = (torch.exp(log_dur_pred) - 1.0) * maskL.float()
            dur_sum = dur_pred_lin.sum(dim=1)
            loss_dur = loss_dur + 0.5 * ((dur_sum - K_tar) / (K_tar + 1e-6)).abs().mean()

            loss_len = (((k_pred / (K_tar + 1e-6)) - 1.0) ** 2).mean()

        return zS, attn, mu_full, logvar_full, zT_sample, zT_mean, loss_dur, loss_len

    def build_ctc_targets_from_texts(texts):
        targets = []
        t_lens = []
        for txt in texts:
            ids = tok.encode(txt)
            if len(ids) == 0:
                ids = [UNK_ID]
            t = torch.tensor(ids, dtype=torch.long)
            targets.append(t)
            t_lens.append(len(ids))
        targets = torch.cat(targets, dim=0).to(device)
        target_lengths = torch.tensor(t_lens, dtype=torch.long, device=device)
        return targets, target_lengths

    # =========================================================
    # Full-length ASR eval
    # =========================================================
    @torch.no_grad()
    def asr_eval_full_one_sample(step_tag: str, idx: int):
        item = aligned_rows[idx]

        wav_path = item["wav"]
        text = str(item.get("text_norm", item.get("text_raw", ""))).strip()

        zS_log_cpu = load_logmel_full_cached(wav_path)   # [K, D]
        K0 = int(zS_log_cpu.shape[0])

        zS_raw_full = zS_log_cpu.to(device).unsqueeze(0)
        maskK_full = torch.ones(1, K0, device=device, dtype=torch.bool)

        zS_full = (zS_raw_full - mu_b) / std_b

        zT_hat = heun_integrate(
            vf, zS_full, maskK_full,
            steps=ode_steps_eval,
            direction=-1,
            cfg_scale=1.0,
            spk_e=None
        )

        logits_hat = text_ctc_head(zT_hat, maskK_full)
        decoded = ctc_greedy_decode(logits_hat, [K0], blank_id=BLANK_ID)[0]
        hyp = ids_to_text(decoded, tok.itos)

        spk_name = extract_speaker_id_from_path(wav_path)
        print(f"\n[ASR-FULL @ {step_tag}] (spkGT={spk_name})\nGT : {text}\nHYP: {hyp}")

    # =========================================================
    # TTS demo
    # =========================================================
    @torch.no_grad()
    def tts_demo_from_text(step_tag: str, text: str, spk_pick: str = None):
        if bigvgan_model is None:
            print("[DEMO] bigvgan_model is None, skip vocoding.")
            return

        if spk_pick is None:
            spk_pick = random.choice(spk_list)
        spk_id = torch.tensor([spk2id[spk_pick]], device=device, dtype=torch.long)
        spk_e = spk_table(spk_id)

        h_st5, m_st5 = st5([text])
        h_enc = h_st5
        if adapter is not None:
            h_enc = adapter(h_enc)

        mu_tok, logvar_tok = text_prior(h_enc)
        maskL = m_st5

        k_pred_full_raw = int(max(16, round(float(len_pred(h_enc, maskL).item()))))
        L_valid = int(maskL.sum().item())
        limit = getattr(vf.rope, "max_seq_len", 4096)
        k_pred_full = min(max(k_pred_full_raw, L_valid), limit)

        log_dur = dur_pred(h_enc, maskL)
        dur = (torch.exp(log_dur) - 1.0) * maskL.float()
        dur_int, _ = durations_to_int_and_fixsum(dur, maskL, k_pred_full)

        mu_feats = []
        lv_feats = []
        for i in range(mu_tok.shape[1]):
            if not bool(maskL[0, i]):
                continue
            d = int(dur_int[i].item())
            mu_feats.append(mu_tok[:, i:i+1, :].repeat(1, d, 1))
            lv_feats.append(logvar_tok[:, i:i+1, :].repeat(1, d, 1))

        if len(mu_feats) == 0:
            zT_mean = mu_tok[:, :1, :].repeat(1, k_pred_full, 1)
            zT_logvar = logvar_tok[:, :1, :].repeat(1, k_pred_full, 1)
        else:
            zT_mean = torch.cat(mu_feats, dim=1)[:, :k_pred_full]
            zT_logvar = torch.cat(lv_feats, dim=1)[:, :k_pred_full]

        if demo_prior_temp > 0:
            eps = torch.randn_like(zT_mean)
            zT0 = zT_mean + demo_prior_temp * torch.exp(0.5 * zT_logvar) * eps
        else:
            zT0 = zT_mean

        maskK = torch.ones(1, zT0.shape[1], device=device, dtype=torch.bool)

        zS_pred = heun_integrate(
            vf, zT0, maskK,
            steps=ode_steps_eval,
            direction=+1,
            cfg_scale=demo_cfg_scale,
            spk_e=spk_e.to(dtype=zT0.dtype)
        )

        zS_ref = mel_refiner(zS_pred, cond=h_enc, cond_mask=maskL)

        mel_log = (zS_ref * std_b + mu_b).float()
        mel_log_clamped = mel_log.clamp(mel_floor, mel_ceil)
        mel_for_vocoder = mel_log_clamped.transpose(1, 2).contiguous()

        print(f"[TTS-DEMO {step_tag}] spk={spk_pick} L={L_valid} K={k_pred_full} cfg={demo_cfg_scale} prior_temp={demo_prior_temp}")

        mel_base = os.path.join(demo_dir, f"tts_demo_{step_tag}_spk{spk_pick}")
        mel_img = mel_log_clamped[0].detach().cpu().numpy().T
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_img, aspect="auto", origin="lower", vmin=mel_floor, vmax=mel_ceil)
        plt.tight_layout()
        plt.savefig(mel_base + "_mel.png", dpi=150)
        plt.close()

        wav_hat = bigvgan_model(mel_for_vocoder).squeeze(1)
        out_path = mel_base + ".wav"
        save_wav(out_path, wav_hat.unsqueeze(1), sr=sampling_rate)

    # =========================================================
    # Training loop
    # =========================================================
    ratios_last = None
    step = 0

    for step in range(total_steps):
        tts_rows = sample_cut_rows(tts_pool, batch_size_tts)
        asr_rows = sample_cut_rows(asr_pool, batch_size_asr)

        (
            zS_tts_log, maskK_tts, K_list_tts,
            texts_tts, spk_ids_tts,
            wav_paths_tts, starts_tts, ends_tts
        ) = build_batch_from_cut_rows(tts_rows)

        (
            zS_asr_log, maskK_asr, K_list_asr,
            texts_asr, spk_ids_asr,
            wav_paths_asr, starts_asr, ends_asr
        ) = build_batch_from_cut_rows(asr_rows)

        B_tts = zS_tts_log.shape[0]
        B_asr = zS_asr_log.shape[0]

        opt.zero_grad(set_to_none=True)

        zt_for_lip = None
        t_for_lip = None
        mask_for_lip = None
        cfg_for_lip = None
        spk_for_lip = None
        loss_vf_lip = torch.tensor(0.0, device=device)
        loss_stft = torch.tensor(0.0, device=device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            h_enc_tts, maskL_tts, mu_tok_tts, logvar_tok_tts = encode_text_batch(texts_tts)
            h_enc_asr, maskL_asr, mu_tok_asr, logvar_tok_asr = encode_text_batch(texts_asr)

            (
                zS_tts, attn_tts, mu_tts, logvar_tts,
                zT_tts_sample, zT_tts_mean,
                loss_dur_tts, loss_len_tts
            ) = build_local_prior_batch(
                zS_tts_log, maskK_tts,
                h_enc_tts, maskL_tts,
                mu_tok_tts, logvar_tok_tts,
                supervise_dur_len=True
            )

            (
                zS_asr, attn_asr, mu_asr, logvar_asr,
                zT_asr_sample, zT_asr_mean,
                _, _
            ) = build_local_prior_batch(
                zS_asr_log, maskK_asr,
                h_enc_asr, maskL_asr,
                mu_tok_asr, logvar_tok_asr,
                supervise_dur_len=False
            )

            spk_e_tts_match = spk_table(spk_ids_tts)

            drop_mask = (torch.rand(B_tts, device=device) < spk_drop_rate)
            if B_tts < 2:
                drop_mask[:] = False

            perm = torch.randperm(B_tts, device=device)
            if B_tts > 1 and torch.all(perm == torch.arange(B_tts, device=device)):
                perm = torch.roll(perm, shifts=1)

            cfg_flag_fm = torch.where(
                drop_mask,
                torch.zeros(B_tts, dtype=torch.long, device=device),
                torch.ones(B_tts, dtype=torch.long, device=device)
            )

            zS_fm_tgt = torch.where(drop_mask[:, None, None], zS_tts[perm], zS_tts)
            spk_e_fm = torch.where(drop_mask[:, None], torch.zeros_like(spk_e_tts_match), spk_e_tts_match)

            t = torch.rand(B_tts, device=device)
            t_ = t.view(B_tts, 1, 1)
            zt = (1 - t_) * zT_tts_sample + t_ * zS_fm_tgt
            U  = zS_fm_tgt - zT_tts_sample

            U_hat = vf(
                zt, t, maskK_tts,
                cfg_flag=cfg_flag_fm,
                spk_e=spk_e_fm.to(dtype=zt.dtype)
            )
            loss_fm = masked_mse(U_hat, U, maskK_tts)

            zt_for_lip = zt.detach()
            t_for_lip = t.detach()
            mask_for_lip = maskK_tts
            cfg_for_lip = cfg_flag_fm
            spk_for_lip = spk_e_fm.detach()

            loss_stat = torch.tensor(0.0, device=device)
            if use_stat_match:
                muT, stdT = masked_mean_std(zT_tts_mean, maskK_tts)
                muS, stdS = masked_mean_std(zS_tts, maskK_tts)
                loss_stat = (muT - muS).abs().mean() + (stdT - stdS).abs().mean()

            zS_end_tts = euler_integrate_grad(
                vf, zT_tts_mean, maskK_tts,
                steps=ode_steps_endloss,
                direction=+1,
                cfg_flag_value=1,
                spk_e=spk_e_tts_match.to(dtype=zT_tts_mean.dtype)
            )

            loss_end_fwd = masked_mse(zS_end_tts, zS_tts, maskK_tts)

            mel_end = (zS_end_tts * std_b + mu_b)
            mel_gt  = zS_tts_log

            loss_tts_mel = masked_l1(mel_end, mel_gt, maskK_tts)

            dt_pred = time_delta_bkd(mel_end)
            dt_gt   = time_delta_bkd(mel_gt)
            mask_dt = maskK_tts[:, 1:] & maskK_tts[:, :-1]
            loss_delta = masked_l1(dt_pred, dt_gt, mask_dt)

            loss_range = mel_range_penalty(
                mel_end.float(), maskK_tts,
                mel_min=mel_floor, mel_max=mel_ceil
            )

            zS_ref = mel_refiner(
                zS_end_tts,
                cond=h_enc_tts,
                cond_mask=maskL_tts
            )
            mel_ref = (zS_ref * std_b + mu_b)
            loss_ref = masked_l1(mel_ref, mel_gt, maskK_tts)

            loss_prior = masked_gaussian_nll(zS_tts, mu_tts, logvar_tts, maskK_tts)

            zT_end_asr = euler_integrate_grad(
                vf, zS_asr, maskK_asr,
                steps=ode_steps_endloss,
                direction=-1,
                cfg_flag_value=0,
                spk_e=None
            )

            loss_end_bwd = masked_mse(zT_end_asr, zT_asr_mean, maskK_asr)

            logits_T   = text_ctc_head(zT_asr_mean, maskK_asr)
            logits_hat = text_ctc_head(zT_end_asr,  maskK_asr)

            logp_T   = F.log_softmax(logits_T,   dim=-1).transpose(0, 1)
            logp_hat = F.log_softmax(logits_hat, dim=-1).transpose(0, 1)

            input_lengths = torch.tensor(K_list_asr, dtype=torch.long, device=device)
            targets, target_lengths = build_ctc_targets_from_texts(texts_asr)

            loss_ctc_T   = ctc_loss_fn(logp_T,   targets, input_lengths, target_lengths)
            loss_ctc_hat = ctc_loss_fn(logp_hat, targets, input_lengths, target_lengths)

            w_ctc = 0.0 if step < w_ctc_start else 1.0
            w_align = 0.0 if step < w_align_start else 1.0

            loss_end = loss_end_fwd + loss_end_bwd

            loss = (
                loss_fm
                + w_end * loss_end
                + w_tts_mel * loss_tts_mel
                + w_delta * loss_delta
                + w_ref * loss_ref
                + w_prior * loss_prior
                + w_mel_range * loss_range
                + (w_stat * loss_stat if use_stat_match else 0.0)
                + w_ctc * (w_ctc_T * loss_ctc_T + w_ctc_hat * loss_ctc_hat)
                + w_align * (w_dur * loss_dur_tts + w_len * loss_len_tts)
            )

        if enable_vf_lip and (step >= vf_lip_start) and (step % vf_lip_every == 0) and (zt_for_lip is not None):
            with torch.amp.autocast('cuda', enabled=False):
                ratios_now = vf_lip_fd_ratio(
                    vf,
                    zt_for_lip.float(),
                    t_for_lip.float(),
                    mask_for_lip,
                    cfg_for_lip,
                    spk_for_lip.float(),
                    sigma=vf_lip_sigma
                )
                loss_vf_lip = F.relu(ratios_now - float(vf_lip_L_hi)).mean()
                ratios_last = ratios_now.detach()
            loss = loss + float(w_vf_lip) * loss_vf_lip

        if use_stft and (bigvgan_model is not None):
            K_stft = 256
            stft_one_sample = True

            with torch.amp.autocast('cuda', enabled=False):
                zS_end_det = zS_end_tts.float().detach()
                h_enc_det  = h_enc_tts.float().detach()
                zS_ref_full = mel_refiner(
                    zS_end_det,
                    cond=h_enc_det,
                    cond_mask=maskL_tts
                )

            b_list = [0] if stft_one_sample else list(range(zS_ref_full.shape[0]))
            stft_acc = 0.0
            denom = 0

            for b in b_list:
                Kb = int(maskK_tts[b].sum().item())
                if Kb < 8:
                    continue

                Kseg2 = min(K_stft, Kb)
                s0 = 0 if Kb <= Kseg2 else int(np.random.randint(0, Kb - Kseg2 + 1))
                e0 = s0 + Kseg2

                with torch.amp.autocast('cuda', enabled=False):
                    mel_hat_seg = (zS_ref_full[b:b+1, s0:e0] * std_b.float() + mu_b.float())
                    mel_hat_seg = mel_hat_seg.clamp(mel_floor, mel_ceil).transpose(1, 2).contiguous()

                with torch.amp.autocast('cuda', enabled=True):
                    wav_hat = bigvgan_model(mel_hat_seg).squeeze()
                wav_hat = wav_hat.float()

                wav_np = load_wav_full_cached(wav_paths_tts[b])
                s_frame = int(starts_tts[b]) + int(s0)
                e_frame = int(starts_tts[b]) + int(e0)
                s_samp = s_frame * hop_size
                e_samp = e_frame * hop_size
                wav_gt_np = wav_np[s_samp:e_samp]

                if wav_gt_np.size <= 64 or wav_hat.numel() <= 64:
                    continue

                wav_gt = torch.tensor(wav_gt_np, device=device, dtype=torch.float32)

                Tm = min(wav_hat.numel(), wav_gt.numel())
                if Tm <= 64:
                    continue

                stft_acc = stft_acc + mrstft_loss(wav_hat[:Tm], wav_gt[:Tm])
                denom += 1

            loss_stft = (stft_acc / max(denom, 1))
            loss = loss + float(w_stft) * loss_stft

        if torch.isnan(loss):
            print("[FATAL] loss is NaN at step", step)
            break

        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        torch.nn.utils.clip_grad_norm_(vf.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(text_prior.parameters(), 1.0)
        if adapter is not None:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dur_pred.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(len_pred.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(text_ctc_head.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(spk_table.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(mel_refiner.parameters(), 1.0)

        scaler.step(opt)
        scaler.update()

        if step % 1000 == 0:
            with torch.no_grad():
                kt0 = int(maskK_tts[0].sum().item())
                ka0 = int(maskK_asr[0].sum().item())

                zs_min = float(zS_tts[0, :kt0].min().item()) if kt0 > 0 else 0.0
                zs_max = float(zS_tts[0, :kt0].max().item()) if kt0 > 0 else 0.0
                zt_min = float(zT_tts_mean[0, :kt0].min().item()) if kt0 > 0 else 0.0
                zt_max = float(zT_tts_mean[0, :kt0].max().item()) if kt0 > 0 else 0.0

                mel_min_v = float(mel_end[0, :kt0].min().item()) if kt0 > 0 else 0.0
                mel_max_v = float(mel_end[0, :kt0].max().item()) if kt0 > 0 else 0.0

            print(
                f"[DBG step{step:05d}] "
                f"TTS_K0={kt0} ASR_K0={ka0} "
                f"zS_tts[min,max]=[{zs_min:.2f},{zs_max:.2f}] "
                f"zT_tts_mean[min,max]=[{zt_min:.2f},{zt_max:.2f}] "
                f"mel_end[min,max]=[{mel_min_v:.2f},{mel_max_v:.2f}] "
                f"stat={float(loss_stat.item()) if use_stat_match else 0.0:.4f}"
            )

        if (step % vf_lip_print_every == 0) and (ratios_last is not None):
            r = ratios_last.float()
            print(
                f"[VF-LIP] L_hi={vf_lip_L_hi} "
                f"min={r.min().item():.4g} "
                f"p50={r.median().item():.4g} "
                f"p90={torch.quantile(r, 0.9).item():.4g} "
                f"max={r.max().item():.4g}"
            )

        if step % 200 == 0:
            with torch.no_grad():
                fwd = masked_mse(zS_end_tts, zS_tts, maskK_tts).item()
                bwd = masked_mse(zT_end_asr, zT_asr_mean, maskK_asr).item()

                decoded_ids = ctc_greedy_decode(
                    logits_hat.detach(),
                    K_list_asr,
                    blank_id=BLANK_ID
                )

                muT, stdT = masked_mean_std(zT_tts_mean, maskK_tts)
                muS, stdS = masked_mean_std(zS_tts, maskK_tts)
                print("[CHK-TTS] mean|muT-muS|", (muT - muS).abs().mean().item(),
                      " mean|stdT-stdS|", (stdT - stdS).abs().mean().item())

            lip_str = ""
            if enable_vf_lip and (step >= vf_lip_start) and (step % vf_lip_every == 0):
                if ratios_last is not None:
                    rr = ratios_last.float().cpu().numpy().tolist()
                    rr_sorted = sorted(rr)
                    p50 = rr_sorted[len(rr_sorted) // 2]
                    p90 = rr_sorted[int(round(0.9 * (len(rr_sorted) - 1)))]
                    lip_str = f" lip {loss_vf_lip.item():.4f} Lp50 {p50:.3f} Lp90 {p90:.3f}"
                else:
                    lip_str = f" lip {loss_vf_lip.item():.4f}"

            stft_str = f" stft {loss_stft.item():.4f}" if use_stft else ""
            print(
                f"[JOINT-CUT] step {step:05d} loss {loss.item():.6f} "
                f"fm {loss_fm.item():.6f} stat {loss_stat.item() if use_stat_match else 0.0:.4f} "
                f"end {loss_end.item():.4f} vf_lip {loss_vf_lip.item():.4f} prior {loss_prior.item():.4f} "
                f"tts {loss_tts_mel.item():.4f} ref {loss_ref.item():.4f} d {loss_delta.item():.4f} "
                f"ctcT {loss_ctc_T.item():.4f} ctcH {loss_ctc_hat.item():.4f} "
                f"dur {loss_dur_tts.item():.4f} len {loss_len_tts.item():.4f} "
                f"fwd {fwd:.4f} bwd {bwd:.4f}"
                f"{lip_str}{stft_str}"
            )

            for i in range(min(2, B_tts)):
                print(f"  [TTS {i}] spk={spk_list[int(spk_ids_tts[i])]} TXT : {texts_tts[i]}")
            for i in range(min(2, B_asr)):
                print(f"  [ASR {i}] spk={spk_list[int(spk_ids_asr[i])]} GT  : {texts_asr[i]}")
                print(f"           HYP : {ids_to_text(decoded_ids[i], tok.itos)}")

        if enable_bilip_diag and diag_every and (step % diag_every == 0) and (batch_size_tts >= 2):
            with torch.no_grad():
                spk_e_cond = spk_e_tts_match.to(dtype=zT_tts_mean.dtype)
                zS_map = euler_integrate(vf, zT_tts_mean, maskK_tts, steps=ode_steps_endloss, direction=+1, cfg_flag_value=1, spk_e=spk_e_cond)
                zT_map = euler_integrate(vf, zS_tts, maskK_tts, steps=ode_steps_endloss, direction=-1, cfg_flag_value=0, spk_e=None)

                r_fwd = batch_pair_ratios(zT_tts_mean, zS_map, maskK_tts)
                r_bwd = batch_pair_ratios(zS_tts, zT_map, maskK_tts)
                if r_fwd is not None:
                    summarize_ratios("FWD (Tmean->S, CUT)", r_fwd)
                if r_bwd is not None:
                    summarize_ratios("BWD (S->Tmean, CUT)", r_bwd)

                f_amp = amp_once(
                    lambda xx: euler_integrate(vf, xx, maskK_tts, steps=ode_steps_endloss, direction=+1, cfg_flag_value=1, spk_e=spk_e_cond),
                    zT_tts_mean, maskK_tts, sigma=amp_sigma
                )
                b_amp = amp_once(
                    lambda xx: euler_integrate(vf, xx, maskK_tts, steps=ode_steps_endloss, direction=-1, cfg_flag_value=0, spk_e=None),
                    zS_tts, maskK_tts, sigma=amp_sigma
                )
                print(f"[AMP] fwd_amp={f_amp:.4g} bwd_amp={b_amp:.4g}")

        if (demo_every is not None) and (demo_every > 0) and (step % demo_every == 0):
            step_tag = f"step{step:05d}"

            try:
                tts_demo_from_text(step_tag, texts_tts[0], spk_pick=random.choice(spk_list))
            except Exception as e:
                print(f"[WARN] TTS demo failed at {step_tag}: {repr(e)}")

            try:
                idx_eval = int(np.random.randint(0, len(aligned_rows)))
                asr_eval_full_one_sample(step_tag, idx_eval)
            except Exception as e:
                print(f"[WARN] ASR full eval failed at {step_tag}: {repr(e)}")

    # =========================================================
    # Save checkpoint
    # =========================================================
    ckpt_dir = os.environ.get("CKPT_DIR", r"/work/dankker0900/biflow_repo_cutmanifest_notebook_masprior/ckpt_joint_notebook_exact")
    os.makedirs(ckpt_dir, exist_ok=True)

    config = dict(
        cut_manifest=str(cut_manifest),
        aligned_manifest=str(aligned_manifest),
        max_cut_rows=max_cut_rows,
        batch_size=int(batch_size),
        batch_size_tts=int(batch_size_tts),
        batch_size_asr=int(batch_size_asr),
        total_steps=int(total_steps),
        lr_all=float(lr_all),
        ds_factor=int(ds_factor),
        ds_align=int(ds_align),
        ode_steps_eval=int(ode_steps_eval),
        ode_steps_endloss=int(ode_steps_endloss),
        demo_cfg_scale=float(demo_cfg_scale),
        demo_prior_temp=float(demo_prior_temp),
        mel_range=dict(floor=float(mel_floor), ceil=float(mel_ceil)),
        w_end=float(w_end),
        w_ctc_hat=float(w_ctc_hat),
        w_ctc_T=float(w_ctc_T),
        w_ctc_start=int(w_ctc_start),
        w_tts_mel=float(w_tts_mel),
        w_ref=float(w_ref),
        w_delta=float(w_delta),
        w_mel_range=float(w_mel_range),
        w_align_start=int(w_align_start),
        w_dur=float(w_dur),
        w_len=float(w_len),
        mas_temp=float(mas_temp),
        w_prior=float(w_prior),
        use_stat_match=bool(use_stat_match),
        w_stat=float(w_stat),
        stft=dict(use=bool(use_stft), w=float(w_stft), refiner_only=True),
        speakers=dict(
            top_k=int(TOP_K_SPK),
            target_spks=list(target_spks),
            spk_list=list(spk_list),
            spk2id=spk2id,
            n_spk=int(n_spk),
            E_spk=int(E_spk),
            spk_scale=float(spk_scale),
            spk_drop_rate=float(spk_drop_rate),
        ),
        adapter=dict(use=bool(use_adapter), bottleneck=int(adapter_bottleneck), dropout=float(adapter_dropout)),
        D_mel=int(D_mel),
        Vt=int(Vt),
        alpha_K=float(alpha_K),
        UNK_ID=int(UNK_ID),
        BLANK_ID=int(BLANK_ID),
        bigvgan=dict(
            name=str(bigvgan_name),
            sampling_rate=int(sampling_rate),
            num_mels=int(num_mels),
            hop_size=int(hop_size),
        ),
        demo_every=demo_every,
        demo_dir=str(demo_dir),
        stats_max_unique_wavs=stats_max_unique_wavs,
        mel_cache_max_items=int(mel_cache_max_items),
        vf_lip_fd=dict(
            enable=bool(enable_vf_lip),
            start=int(vf_lip_start),
            every=int(vf_lip_every),
            w=float(w_vf_lip),
            L_hi=float(vf_lip_L_hi),
            sigma=float(vf_lip_sigma),
            print_every=int(vf_lip_print_every),
        )
    )

    state = dict(
        step=int(step),
        config=config,
        adapter=(adapter.state_dict() if adapter is not None else None),
        text_prior=text_prior.state_dict(),
        dur_pred=dur_pred.state_dict(),
        len_pred=len_pred.state_dict(),
        spk_table=spk_table.state_dict(),
        vf=vf.state_dict(),
        text_ctc_head=text_ctc_head.state_dict(),
        mel_refiner=mel_refiner.state_dict(),
        mu_g=mu_g.detach().cpu().float(),
        std_g=std_g.detach().cpu().float(),
        tok_stoi=tok.stoi,
        tok_itos=tok.itos,
        opt=opt.state_dict(),
        scaler=scaler.state_dict() if (use_amp and scaler is not None) else None,
    )

    ckpt_path = os.path.join(ckpt_dir, "joint_singleVF_CUTmanifest.pt")
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    torch.save(state, ckpt_path)
    torch.save(state, latest_path)

    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n[OK] Training done. Saved to:", ckpt_dir)
    print(" -", ckpt_path)
    print(" -", latest_path)


if __name__ == "__main__":
    main()