import torch
import torch.nn.functional as F


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

def sample_q_beta_time(B, device, beta=0.0):
    beta = float(beta)
    if beta <= 0.0:
        return torch.rand(B, device=device)
    beta = max(0.0, min(beta, 1.0))
    t = torch.rand(B, device=device)
    r = torch.rand(B, device=device)
    edge = r < beta
    edge_lr = torch.rand(B, device=device) < 0.5
    t = torch.where(edge & edge_lr, torch.zeros_like(t), t)
    t = torch.where(edge & (~edge_lr), torch.ones_like(t), t)
    return t

def vf_lip_fd_ratio(vf, z_base, t, maskK, cfg_flag, spk_e, style_e=None, text_cond=None, sigma=0.01, eps=1e-8):
    delta = torch.randn_like(z_base) * float(sigma)
    delta = delta * maskK.float().unsqueeze(-1)

    v0 = vf(z_base, t, maskK, cfg_flag=cfg_flag, spk_e=spk_e, style_e=style_e, text_cond=text_cond)
    v1 = vf(z_base + delta, t, maskK, cfg_flag=cfg_flag, spk_e=spk_e, style_e=style_e, text_cond=text_cond)

    num = torch.sqrt(masked_mse_per_sample(v1, v0, maskK, eps=eps) + eps)  # [B]
    den = torch.sqrt(masked_mse_per_sample(z_base + delta, z_base, maskK, eps=eps) + eps)  # [B]
    return num / (den + eps)

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

def hidden_ssl_cosine_loss(pred_bkd, pred_mask, tgt_bkd, tgt_mask, eps=1e-8):
    losses = []
    B = pred_bkd.shape[0]
    for b in range(B):
        kp = int(pred_mask[b].long().sum().item())
        kt = int(tgt_mask[b].long().sum().item())
        if kp <= 1 or kt <= 1:
            continue
        pred = pred_bkd[b:b+1, :kp].transpose(1, 2)
        pred = F.interpolate(pred, size=kt, mode="linear", align_corners=False).transpose(1, 2)[0]
        tgt = tgt_bkd[b, :kt]
        cos = F.cosine_similarity(pred, tgt, dim=-1, eps=eps)
        losses.append(1.0 - cos.mean())
    if len(losses) == 0:
        return pred_bkd.new_tensor(0.0)
    return torch.stack(losses).mean()

def masked_cosine_loss(a, b, mask, eps=1e-8):
    m = mask.float()
    cos = F.cosine_similarity(a, b, dim=-1, eps=eps)
    loss = (1.0 - cos) * m
    denom = m.sum().clamp_min(1.0)
    return loss.sum() / denom

def masked_mean_pool(x_bld, mask_bl, eps=1e-6):
    if mask_bl is None:
        return x_bld.mean(dim=1)
    w = mask_bl.float().unsqueeze(-1)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (x_bld * w).sum(dim=1) / (denom + eps)
