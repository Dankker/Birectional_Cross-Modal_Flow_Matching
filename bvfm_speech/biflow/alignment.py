import math

import torch
import torch.nn.functional as F


_AA_POS_KERNEL_CACHE = {}


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


def monotonic_alignment_posterior(log_probs, maskK, maskL, neg_inf=-1e4):
    device = log_probs.device
    B, K, L = log_probs.shape
    lp = log_probs.float()
    lp = lp.masked_fill(~maskK[:, :, None], neg_inf)
    lp = lp.masked_fill(~maskL[:, None, :], neg_inf)

    post = torch.zeros((B, K, L), device=device, dtype=torch.float32)
    neg_inf_t = torch.tensor(float("-inf"), device=device, dtype=torch.float32)

    for b in range(B):
        Kb = int(maskK[b].sum().item())
        Lb = int(maskL[b].sum().item())
        if Kb <= 0 or Lb <= 0 or Kb < Lb:
            continue

        alpha = torch.full((Kb, Lb), neg_inf_t.item(), device=device, dtype=torch.float32)
        alpha[0, 0] = lp[b, 0, 0]
        for k in range(1, Kb):
            j_min = max(0, Lb - (Kb - k))
            j_max = min(k, Lb - 1)
            for j in range(j_min, j_max + 1):
                if j == 0:
                    prev = alpha[k - 1, 0]
                else:
                    prev = torch.logaddexp(alpha[k - 1, j], alpha[k - 1, j - 1])
                alpha[k, j] = lp[b, k, j] + prev

        log_z = alpha[Kb - 1, Lb - 1]
        if not torch.isfinite(log_z):
            continue

        beta = torch.full((Kb, Lb), neg_inf_t.item(), device=device, dtype=torch.float32)
        beta[Kb - 1, Lb - 1] = 0.0
        for k in range(Kb - 2, -1, -1):
            j_min = max(0, Lb - (Kb - k))
            j_max = min(k, Lb - 1)
            for j in range(j_max, j_min - 1, -1):
                stay = beta[k + 1, j] + lp[b, k + 1, j]
                if j + 1 < Lb:
                    move = beta[k + 1, j + 1] + lp[b, k + 1, j + 1]
                    beta[k, j] = torch.logaddexp(stay, move)
                else:
                    beta[k, j] = stay

        gamma = torch.exp(alpha + beta - log_z)
        post[b, :Kb, :Lb] = gamma

    post = post * maskK[:, :, None].float() * maskL[:, None, :].float()
    denom = post.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    post = post / denom
    post = post * maskK[:, :, None].float()
    return post

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
