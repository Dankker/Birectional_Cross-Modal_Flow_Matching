import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResidualAdapter(nn.Module):
    def __init__(self, dim, bottleneck=192, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bottleneck)
        self.fc2 = nn.Linear(bottleneck, dim)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, mask=None):
        h = self.ln(x)
        h = F.silu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class CanonicalTextEncoderBlock(nn.Module):
    def __init__(self, dim, n_heads=8, ff_mult=4, conv_ksize=5, dropout=0.1):
        super().__init__()
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        self.ln_conv = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=conv_ksize,
            padding=conv_ksize // 2,
            groups=dim,
        )
        self.conv_pw = nn.Linear(dim, dim)
        self.drop_conv = nn.Dropout(dropout)

        ff_hidden = int(dim * ff_mult)
        self.ln_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        key_padding_mask = None if mask is None else ~mask.bool()
        h = self.ln_attn(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop_attn(h)
        if mask is not None:
            x = x * mask.float().unsqueeze(-1)

        h = self.ln_conv(x)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(self.conv_pw(h))
        x = x + self.drop_conv(h)
        if mask is not None:
            x = x * mask.float().unsqueeze(-1)

        x = x + self.ff(self.ln_ff(x))
        if mask is not None:
            x = x * mask.float().unsqueeze(-1)
        return x


class CanonicalTextEncoder(nn.Module):
    """
    Trainable contextual adaptor on top of frozen SpeechT5 hidden states.
    It starts as an identity mapping through zero-initialized output delta,
    then learns a canonical text representation for the downstream prior.
    """
    def __init__(
        self,
        dim,
        layers=4,
        n_heads=8,
        ff_mult=4,
        conv_ksize=5,
        dropout=0.1,
        residual_scale=1.0,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.in_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            CanonicalTextEncoderBlock(
                dim=dim,
                n_heads=n_heads,
                ff_mult=ff_mult,
                conv_ksize=conv_ksize,
                dropout=dropout,
            )
            for _ in range(int(layers))
        ])
        self.out_norm = nn.LayerNorm(dim)
        self.delta = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x, mask=None):
        h = self.in_norm(x)
        if mask is not None:
            h = h * mask.float().unsqueeze(-1)
        for block in self.blocks:
            h = block(h, mask=mask)
        delta = self.drop(self.delta(self.out_norm(h)))
        y = x + self.residual_scale * delta
        if mask is not None:
            y = y * mask.float().unsqueeze(-1)
        return y


class TrainableTokenTextEncoder(nn.Module):
    """Trainable token encoder for a fixed char/BPE vocabulary."""

    def __init__(
        self,
        vocab_size,
        dim=384,
        layers=6,
        n_heads=6,
        ff_mult=4,
        conv_ksize=5,
        dropout=0.1,
        max_len=1024,
        padding_idx=0,
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.token_embed = nn.Embedding(int(vocab_size), int(dim), padding_idx=int(padding_idx))
        self.pos_embed = nn.Embedding(self.max_len, int(dim))
        self.drop = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList([
            CanonicalTextEncoderBlock(
                dim=int(dim),
                n_heads=int(n_heads),
                ff_mult=int(ff_mult),
                conv_ksize=int(conv_ksize),
                dropout=float(dropout),
            )
            for _ in range(int(layers))
        ])
        self.out_norm = nn.LayerNorm(int(dim))
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        if self.token_embed.padding_idx is not None:
            with torch.no_grad():
                self.token_embed.weight[self.token_embed.padding_idx].zero_()

    def forward(self, token_ids, mask=None):
        B, L = token_ids.shape
        if L > self.max_len:
            raise ValueError(f"Text token length {L} exceeds text_encoder_max_len={self.max_len}")
        positions = torch.arange(L, device=token_ids.device).view(1, L).expand(B, L)
        h = self.drop(self.token_embed(token_ids) + self.pos_embed(positions))
        if mask is not None:
            h = h * mask.float().unsqueeze(-1)
        for block in self.blocks:
            h = block(h, mask=mask)
        h = self.out_norm(h)
        if mask is not None:
            h = h * mask.float().unsqueeze(-1)
        return h

class FastSpeech2DurationPredictor(nn.Module):
    def __init__(self, D=80, hidden=256, ksize=3, dropout=0.5, cond_dim=0):
        super().__init__()
        pad = ksize // 2
        self.conv1 = nn.Conv1d(D, hidden, kernel_size=ksize, padding=pad)
        self.ln1   = nn.LayerNorm(hidden)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=ksize, padding=pad)
        self.ln2   = nn.LayerNorm(hidden)
        self.drop2 = nn.Dropout(dropout)

        self.proj  = nn.Linear(hidden, 1)
        self.cond_dim = int(max(0, cond_dim))
        self.cond_proj = None
        if self.cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(self.cond_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            nn.init.zeros_(self.cond_proj[-1].weight)
            nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(self, h_tok, maskL, cond=None):
        x = h_tok.transpose(1, 2)
        x = self.conv1(x).transpose(1, 2)
        if self.cond_proj is not None and cond is not None:
            x = x + self.cond_proj(cond).unsqueeze(1)
        x = F.relu(x)
        x = self.ln1(x)
        x = self.drop1(x)

        x = x.transpose(1, 2)
        x = self.conv2(x).transpose(1, 2)
        if self.cond_proj is not None and cond is not None:
            x = x + self.cond_proj(cond).unsqueeze(1)
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
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h_tok * w.unsqueeze(-1)).sum(dim=1) / denom
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
        x = x * maskK.float().unsqueeze(-1)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = F.gelu(conv(x)) + x
        x = x.transpose(1, 2)
        x = x * maskK.float().unsqueeze(-1)
        return self.out(x)

class BaselineCTCHead(nn.Module):
    """
    A stronger ASR-baseline-style CTC head:
    frame projection + residual conv frontend + BiLSTM encoder + linear CTC classifier.
    """
    def __init__(self, V, D=80, hidden=384, conv_layers=2, ksize=5, lstm_hidden=384, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(D, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel_size=ksize, padding=ksize // 2)
            for _ in range(conv_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(conv_layers)])
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=(dropout if lstm_layers > 1 else 0.0),
            bidirectional=True,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(2 * lstm_hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, V),
        )

    def forward(self, z, maskK):
        lengths = maskK.long().sum(dim=1).clamp_min(1)
        x = self.inp(z)
        x = x * maskK.float().unsqueeze(-1)

        h = x.transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            y = conv(h).transpose(1, 2)
            y = norm(y)
            y = F.gelu(y)
            x = x + y
            x = x * maskK.float().unsqueeze(-1)
            h = x.transpose(1, 2)

        packed = pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=z.shape[1])
        x = x * maskK.float().unsqueeze(-1)
        return self.out(x)


class ZipformerFeedForward(nn.Module):
    def __init__(self, hidden, ff_mult=4.0, dropout=0.1):
        super().__init__()
        ff_hidden = int(hidden * float(ff_mult))
        self.net = nn.Sequential(
            nn.Linear(hidden, ff_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ZipformerConvModule(nn.Module):
    def __init__(self, hidden, kernel_size=31, dropout=0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.norm = nn.LayerNorm(hidden)
        self.pointwise_in = nn.Linear(hidden, 2 * hidden)
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden,
        )
        self.depthwise_norm = nn.LayerNorm(hidden)
        self.pointwise_out = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.norm(x)
        h = F.glu(self.pointwise_in(h), dim=-1)
        if mask is not None:
            h = h * mask.float().unsqueeze(-1)
        h = self.depthwise(h.transpose(1, 2)).transpose(1, 2)
        h = F.silu(self.depthwise_norm(h))
        h = self.drop(self.pointwise_out(h))
        if mask is not None:
            h = h * mask.float().unsqueeze(-1)
        return h


class ZipformerLiteBlock(nn.Module):
    """
    Lightweight Zipformer/Conformer-style ASR block for latent CTC heads.
    It keeps the repository's [B, K, D] + mask interface and avoids the
    full icefall dependency stack while preserving the useful ASR ingredients:
    macaron FFNs, global self-attention, and local depthwise convolution.
    """
    def __init__(self, hidden, heads=8, ff_mult=4.0, conv_kernel=31, dropout=0.1):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        self.ff1_norm = nn.LayerNorm(hidden)
        self.ff1 = ZipformerFeedForward(hidden, ff_mult=ff_mult, dropout=dropout)
        self.attn_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ZipformerConvModule(hidden, kernel_size=conv_kernel, dropout=dropout)
        self.ff2_norm = nn.LayerNorm(hidden)
        self.ff2 = ZipformerFeedForward(hidden, ff_mult=ff_mult, dropout=dropout)
        self.final_norm = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        mask_f = None if mask is None else mask.float().unsqueeze(-1)
        key_padding_mask = None if mask is None else ~mask.bool()

        x = x + 0.5 * self.ff1(self.ff1_norm(x))
        if mask_f is not None:
            x = x * mask_f

        h = self.attn_norm(x)
        h, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.attn_drop(h)
        if mask_f is not None:
            x = x * mask_f

        x = x + self.conv(x, mask=mask)
        if mask_f is not None:
            x = x * mask_f

        x = x + 0.5 * self.ff2(self.ff2_norm(x))
        x = self.final_norm(x)
        if mask_f is not None:
            x = x * mask_f
        return x


class ZipformerCTCHead(nn.Module):
    """
    Latent-level Zipformer-style CTC head.

    This is intended for zC/zT_hat ASR paths, not raw log-mel frontend
    replacement.  The middle stack optionally runs at a lower frame rate and is
    merged back to the full-rate stream before CTC projection.
    """
    def __init__(
        self,
        V,
        D=80,
        hidden=384,
        layers=4,
        heads=6,
        ff_mult=4.0,
        ksize=31,
        dropout=0.1,
        downsample_factor=2,
    ):
        super().__init__()
        layers = int(max(1, layers))
        hidden = int(hidden)
        heads = int(heads)
        if hidden % heads != 0:
            raise ValueError(f"ctc hidden={hidden} must be divisible by ctc heads={heads}")

        mid_layers = max(1, layers // 2)
        side_layers = max(0, layers - mid_layers)
        pre_layers = side_layers // 2
        post_layers = side_layers - pre_layers
        if layers >= 3 and pre_layers == 0:
            pre_layers = 1
            mid_layers = max(1, mid_layers - 1)

        self.downsample_factor = int(max(1, downsample_factor))
        self.inp = nn.Sequential(
            nn.Linear(D, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.pre_blocks = nn.ModuleList([
            ZipformerLiteBlock(hidden, heads=heads, ff_mult=ff_mult, conv_kernel=ksize, dropout=dropout)
            for _ in range(pre_layers)
        ])
        self.mid_blocks = nn.ModuleList([
            ZipformerLiteBlock(hidden, heads=heads, ff_mult=ff_mult, conv_kernel=ksize, dropout=dropout)
            for _ in range(mid_layers)
        ])
        self.post_blocks = nn.ModuleList([
            ZipformerLiteBlock(hidden, heads=heads, ff_mult=ff_mult, conv_kernel=ksize, dropout=dropout)
            for _ in range(post_layers)
        ])

        if self.downsample_factor > 1:
            k = 2 * self.downsample_factor - 1
            self.downsample = nn.Conv1d(
                hidden,
                hidden,
                kernel_size=k,
                stride=self.downsample_factor,
                padding=self.downsample_factor - 1,
            )
            self.down_norm = nn.LayerNorm(hidden)
            self.merge = nn.Sequential(
                nn.LayerNorm(2 * hidden),
                nn.Linear(2 * hidden, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            nn.init.zeros_(self.merge[-1].weight)
            nn.init.zeros_(self.merge[-1].bias)
        else:
            self.downsample = None
            self.down_norm = None
            self.merge = None

        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, V),
        )

    @staticmethod
    def _downsample_mask(mask, factor, target_len):
        if factor <= 1:
            return mask
        B, K = mask.shape
        pad = (-K) % factor
        if pad:
            mask = torch.cat([mask, mask.new_zeros(B, pad)], dim=1)
        mask = mask.view(B, -1, factor).any(dim=-1)
        if mask.shape[1] < target_len:
            pad = target_len - mask.shape[1]
            mask = torch.cat([mask, mask.new_zeros(B, pad)], dim=1)
        return mask[:, :target_len]

    def forward(self, z, maskK):
        x = self.inp(z)
        x = x * maskK.float().unsqueeze(-1)
        for block in self.pre_blocks:
            x = block(x, maskK)

        if self.downsample is not None and x.shape[1] > 1:
            K = x.shape[1]
            low = self.downsample(x.transpose(1, 2)).transpose(1, 2)
            low_mask = self._downsample_mask(maskK, self.downsample_factor, low.shape[1])
            low = self.down_norm(low) * low_mask.float().unsqueeze(-1)
            for block in self.mid_blocks:
                low = block(low, low_mask)
            low_up = F.interpolate(
                low.transpose(1, 2),
                size=K,
                mode="nearest",
            ).transpose(1, 2)
            x = x + self.merge(torch.cat([x, low_up], dim=-1))
            x = x * maskK.float().unsqueeze(-1)
        else:
            for block in self.mid_blocks:
                x = block(x, maskK)

        for block in self.post_blocks:
            x = block(x, maskK)
        return self.out(x)


class AttentionCTCDecoder(nn.Module):
    """
    Teacher-forced attention decoder for CTC encoder representations.

    This is intended as an auxiliary AED loss beside CTC.  The CTC vocabulary is
    kept unchanged; callers can append extra SOS/EOS ids to the decoder vocab.
    """
    def __init__(
        self,
        vocab_size,
        encoder_dim,
        d_model=384,
        layers=4,
        heads=6,
        ff_mult=4.0,
        dropout=0.1,
        pad_id=0,
        max_len=512,
    ):
        super().__init__()
        d_model = int(d_model)
        heads = int(heads)
        if d_model % heads != 0:
            raise ValueError(f"decoder hidden={d_model} must be divisible by decoder heads={heads}")
        self.pad_id = int(pad_id)
        self.max_len = int(max_len)
        self.memory_proj = nn.Sequential(
            nn.Linear(int(encoder_dim), d_model),
            nn.LayerNorm(d_model),
        )
        self.token_embed = nn.Embedding(int(vocab_size), d_model, padding_idx=self.pad_id)
        self.pos_embed = nn.Embedding(self.max_len, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=int(d_model * float(ff_mult)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(max(1, layers)))
        self.out = nn.Linear(d_model, int(vocab_size))

    def forward(self, memory, memory_mask, decoder_input_ids):
        B, T = decoder_input_ids.shape
        if T > self.max_len:
            raise ValueError(f"decoder target length {T} exceeds max_len={self.max_len}")
        memory = self.memory_proj(memory)
        memory = memory * memory_mask.float().unsqueeze(-1)
        pos = torch.arange(T, device=decoder_input_ids.device).unsqueeze(0)
        tgt = self.token_embed(decoder_input_ids) + self.pos_embed(pos)
        tgt_key_padding_mask = decoder_input_ids.eq(self.pad_id)
        memory_key_padding_mask = ~memory_mask.bool()
        causal_mask = torch.triu(
            torch.ones(T, T, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out(h)


class ADMAHiddenCTCHead(nn.Module):
    """
    A-DMA-style hidden CTC projector/compressor.
    It consumes an intermediate DiT hidden state, applies a lightweight
    temporal projector/compressor, and outputs frame-wise vocabulary logits.
    """
    def __init__(self, V, in_dim, hidden=384, ksize=5, dropout=0.1, pool_factors=(1, 1)):
        super().__init__()
        self.pool_factors = tuple(int(max(1, x)) for x in pool_factors)

        self.inp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList()
        for _ in self.pool_factors:
            self.blocks.append(nn.ModuleDict(dict(
                conv=nn.Conv1d(hidden, hidden, kernel_size=ksize, padding=ksize // 2),
                norm=nn.LayerNorm(hidden),
            )))

        self.out = nn.Linear(hidden, V)

    def _downsample_mask(self, maskK, factor):
        if factor <= 1:
            return maskK
        B, K = maskK.shape
        K_new = (K + factor - 1) // factor
        pad = K_new * factor - K
        if pad > 0:
            maskK = F.pad(maskK, (0, pad), value=False)
        maskK = maskK.view(B, K_new, factor)
        return maskK.any(dim=-1)

    def forward(self, h, maskK):
        x = self.inp(h)
        x = x * maskK.float().unsqueeze(-1)
        mask_now = maskK

        for blk, factor in zip(self.blocks, self.pool_factors):
            y = blk["conv"](x.transpose(1, 2)).transpose(1, 2)
            y = blk["norm"](y)
            y = F.gelu(y)
            x = x + y
            x = x * mask_now.float().unsqueeze(-1)

            if factor > 1:
                x = F.avg_pool1d(x.transpose(1, 2), kernel_size=factor, stride=factor, ceil_mode=True).transpose(1, 2)
                mask_now = self._downsample_mask(mask_now, factor)
                x = x * mask_now.float().unsqueeze(-1)

        logits = self.out(x)
        return logits, mask_now

class ADMASpeechAlignMLP(nn.Module):
    """
    More A-DMA-like speech-guided alignment head for late DiT hidden states.
    It first projects DiT hidden states into a speech-alignment hidden space,
    refines them with residual Conv1d blocks, and finally projects them into
    the SSL teacher feature space. Sequence-length alignment is still handled
    later via linear interpolation to the teacher frame length.
    """
    def __init__(self, in_dim, out_dim, hidden=512, pool_factors=(1, 1), ksize=5, dropout=0.1):
        super().__init__()
        self.pool_factors = tuple(int(x) for x in pool_factors)
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList()
        for _pf in self.pool_factors:
            self.blocks.append(nn.ModuleDict(dict(
                conv=nn.Conv1d(hidden, hidden, kernel_size=ksize, padding=ksize // 2),
                norm=nn.LayerNorm(hidden),
                drop=nn.Dropout(dropout),
            )))
        self.out = nn.Linear(hidden, out_dim)

    def _downsample_mask(self, maskK, factor):
        if factor <= 1:
            return maskK
        B, K = maskK.shape
        K_new = (K + factor - 1) // factor
        pad = K_new * factor - K
        if pad > 0:
            maskK = F.pad(maskK, (0, pad), value=False)
        maskK = maskK.view(B, K_new, factor)
        return maskK.any(dim=-1)

    def forward(self, h, maskK):
        x = self.inp(h)
        x = x * maskK.float().unsqueeze(-1)
        mask_now = maskK
        for blk, factor in zip(self.blocks, self.pool_factors):
            y = blk["conv"](x.transpose(1, 2)).transpose(1, 2)
            y = blk["norm"](y)
            y = F.gelu(y)
            y = blk["drop"](y)
            x = (x + y) * mask_now.float().unsqueeze(-1)
            if factor > 1:
                x = F.avg_pool1d(x.transpose(1, 2), kernel_size=factor, stride=factor, ceil_mode=True).transpose(1, 2)
                mask_now = self._downsample_mask(mask_now, factor)
                x = x * mask_now.float().unsqueeze(-1)
        return self.out(x), mask_now

class TextHiddenProjectorHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, n_blocks=2, ksize=5, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                conv=nn.Conv1d(hidden, hidden, kernel_size=ksize, padding=ksize // 2),
                norm=nn.LayerNorm(hidden),
                drop=nn.Dropout(dropout),
            ))
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, z, maskK):
        mask = maskK.float().unsqueeze(-1)
        x = self.inp(z) * mask
        for blk in self.blocks:
            y = blk["conv"](x.transpose(1, 2)).transpose(1, 2)
            y = blk["norm"](y)
            y = F.gelu(y)
            y = blk["drop"](y)
            x = (x + y) * mask
        return self.out(x)

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

def masked_mean_pool(x_bld, mask_bl, eps=1e-6):
    if mask_bl is None:
        return x_bld.mean(dim=1)
    w = mask_bl.float().unsqueeze(-1)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (x_bld * w).sum(dim=1) / (denom + eps)


def masked_mean_std_pool(x_bld, mask_bl, eps=1e-6):
    mean = masked_mean_pool(x_bld, mask_bl, eps=eps)
    if mask_bl is None:
        var = ((x_bld - mean.unsqueeze(1)) ** 2).mean(dim=1)
        return mean, torch.sqrt(var.clamp_min(eps))
    w = mask_bl.float().unsqueeze(-1)
    denom = w.sum(dim=1).clamp_min(1.0)
    var = (((x_bld - mean.unsqueeze(1)) ** 2) * w).sum(dim=1) / (denom + eps)
    return mean, torch.sqrt(var.clamp_min(eps))


class SourceStatsConditioner(nn.Module):
    def __init__(self, D=80, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * D, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, source_delta, maskK):
        mean, std = masked_mean_std_pool(source_delta, maskK)
        return self.net(torch.cat([mean, std], dim=-1))

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


class SpeakerConditioner(nn.Module):
    """
    Speaker conditioning module for closed-set TTS.

    mode="table" keeps the previous learned speaker-ID table behavior.
    mode="ecapa"/"xvector"/"pretrained" projects a frozen pretrained speaker
    embedding bank into the TTS speaker-conditioning dimension.
    mode="*_plus_table" adds a small learned residual table on top.
    """
    TABLE_MODES = {"table", "id", "speaker_id"}
    PRETRAINED_MODES = {"pretrained", "ecapa", "xvector"}
    PRETRAINED_PLUS_TABLE_MODES = {
        "pretrained_plus_table",
        "ecapa_plus_table",
        "xvector_plus_table",
    }

    def __init__(
        self,
        n_spk: int,
        E: int,
        scale: float = 0.5,
        mode: str = "table",
        pretrained_emb: torch.Tensor | None = None,
        pretrained_trainable: bool = False,
        delta_scale: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.n_spk = int(n_spk)
        self.E = int(E)
        self.scale = float(scale)
        self.mode = str(mode).lower()
        self.delta_scale = float(delta_scale)
        self.use_layernorm = bool(use_layernorm)

        valid_modes = self.TABLE_MODES | self.PRETRAINED_MODES | self.PRETRAINED_PLUS_TABLE_MODES
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported speaker conditioner mode={mode!r}")

        self.uses_pretrained = self.mode in (self.PRETRAINED_MODES | self.PRETRAINED_PLUS_TABLE_MODES)
        self.uses_table = self.mode in (self.TABLE_MODES | self.PRETRAINED_PLUS_TABLE_MODES)

        if self.uses_pretrained:
            if pretrained_emb is None:
                raise ValueError(f"speaker conditioner mode={mode!r} requires pretrained_emb")
            if pretrained_emb.ndim != 2:
                raise ValueError(
                    f"pretrained_emb must be 2-D [n_spk, dim], got shape={tuple(pretrained_emb.shape)}"
                )
            if int(pretrained_emb.shape[0]) != self.n_spk:
                raise ValueError(
                    f"pretrained_emb n_spk mismatch: bank={int(pretrained_emb.shape[0])} n_spk={self.n_spk}"
                )
            pretrained_emb = pretrained_emb.detach().float()
            if pretrained_trainable:
                self.pretrained_emb = nn.Parameter(pretrained_emb)
            else:
                self.register_buffer("pretrained_emb", pretrained_emb, persistent=True)
            self.pretrained_proj = nn.Linear(int(pretrained_emb.shape[1]), self.E)
        else:
            self.pretrained_emb = None
            self.pretrained_proj = None

        self.emb = nn.Embedding(self.n_spk, self.E) if self.uses_table else None
        if self.emb is not None:
            nn.init.zeros_(self.emb.weight)

        self.ln = nn.LayerNorm(self.E) if self.use_layernorm else nn.Identity()

    def forward(self, spk_id: torch.LongTensor):
        if self.uses_pretrained:
            raw = self.pretrained_emb[spk_id].to(dtype=self.pretrained_proj.weight.dtype)
            e = self.pretrained_proj(raw)
        else:
            e = self.emb(spk_id)

        if self.mode in self.PRETRAINED_PLUS_TABLE_MODES:
            e = e + self.delta_scale * self.emb(spk_id).to(dtype=e.dtype)

        e = self.ln(e) * self.scale
        return e

    def from_pretrained_embedding(self, raw_emb: torch.Tensor):
        """
        Project externally extracted speaker embeddings into the same conditioning
        space as the pretrained speaker bank.

        This is used for zero-shot/reference-audio conditioning. The learned
        speaker-ID residual table is intentionally not added because a new
        reference speaker may not exist in the closed-set table.
        """
        if not self.uses_pretrained or self.pretrained_proj is None:
            raise RuntimeError("from_pretrained_embedding requires a pretrained speaker conditioner mode")
        raw = raw_emb.to(device=self.pretrained_proj.weight.device, dtype=self.pretrained_proj.weight.dtype)
        e = self.pretrained_proj(raw)
        e = self.ln(e) * self.scale
        return e

    def extra_repr(self):
        parts = [
            f"mode={self.mode}",
            f"n_spk={self.n_spk}",
            f"E={self.E}",
            f"scale={self.scale}",
        ]
        if self.mode in self.PRETRAINED_PLUS_TABLE_MODES:
            parts.append(f"delta_scale={self.delta_scale}")
        return ", ".join(parts)

class StyleEncoder(nn.Module):
    def __init__(self, D=80, hidden=256, out_dim=128, conv_layers=3, ksize=5, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(D, hidden)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel_size=ksize, padding=ksize // 2)
            for _ in range(conv_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(conv_layers)])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z, maskK):
        mask = maskK.float().unsqueeze(-1)
        x = self.inp(z) * mask
        h = x.transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            y = conv(h).transpose(1, 2)
            y = norm(y)
            y = F.gelu(y)
            y = self.drop(y)
            x = (x + y) * mask
            h = x.transpose(1, 2)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (x * mask).sum(dim=1) / denom
        return self.out(pooled)

class TTSStylePosterior(nn.Module):
    """
    Utterance-level TTS-only posterior q(u | ...).

    mode="speech" keeps the legacy q(u | s) posterior. mode="path" uses
    q(u | s, z_t, t, spk), which matches the variational-FM path latent while
    still producing one global u per utterance.
    """
    def __init__(self, D=80, spk_dim=64, latent_dim=64, hidden=256, dropout=0.1, mode="speech"):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"speech", "path"}:
            raise ValueError(f"Unsupported TTSStylePosterior mode: {mode}")
        self.mode = mode
        self.D = int(D)
        self.spk_dim = int(spk_dim)
        self.latent_dim = int(latent_dim)
        in_dim = 2 * self.D
        if self.mode == "path":
            in_dim = 4 * self.D + self.spk_dim + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden, self.latent_dim)
        self.logvar = nn.Linear(hidden, self.latent_dim)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -4.0)

    def forward(self, z, maskK, z_t=None, t=None, spk_e=None):
        mean, std = masked_mean_std_pool(z, maskK)
        parts = [mean, std]
        if self.mode == "path":
            if z_t is None or t is None or spk_e is None:
                raise ValueError("mode='path' requires z_t, t, and spk_e")
            zt_mean, zt_std = masked_mean_std_pool(z_t, maskK)
            if t.dim() == 0:
                t = t.view(1).expand(z.shape[0])
            t_feat = t.to(device=z.device, dtype=z.dtype).view(z.shape[0], -1)[:, :1]
            spk = spk_e.to(device=z.device, dtype=z.dtype)
            parts.extend([zt_mean, zt_std, t_feat, spk])
        h = self.net(torch.cat(parts, dim=-1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)


class TTSStylePairPosterior(nn.Module):
    """
    Training-only full-information posterior q(u | z_s, z_c, r).

    This encoder sees both endpoints and the speaker condition, then produces one
    utterance-level realization latent. Inference uses the one-sided students
    instead: p(u | z_c, r) for TTS and q(u | z_s) for ASR.
    """
    def __init__(self, s_dim=80, c_dim=80, spk_dim=64, latent_dim=64, hidden=256, dropout=0.1):
        super().__init__()
        self.s_dim = int(s_dim)
        self.c_dim = int(c_dim)
        self.spk_dim = int(spk_dim)
        self.latent_dim = int(latent_dim)
        in_dim = 2 * self.s_dim + 2 * self.c_dim + self.spk_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden, self.latent_dim)
        self.logvar = nn.Linear(hidden, self.latent_dim)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -4.0)

    def forward(self, z_s, mask_s, z_c, mask_c, spk_e):
        s_mean, s_std = masked_mean_std_pool(z_s, mask_s)
        c_mean, c_std = masked_mean_std_pool(z_c, mask_c)
        spk = spk_e.to(device=z_s.device, dtype=z_s.dtype)
        h = self.net(torch.cat([s_mean, s_std, c_mean, c_std, spk], dim=-1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)


class TTSStylePrior(nn.Module):
    """
    Speaker-conditioned prior p(u | r). The zero initialization makes the
    initial prior close to N(0, exp(-4) I), so enabling the module starts as a
    near no-op and the KL can be annealed in safely.
    """
    def __init__(self, spk_dim=64, latent_dim=64, hidden=256, dropout=0.1):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(spk_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden, self.latent_dim)
        self.logvar = nn.Linear(hidden, self.latent_dim)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -4.0)

    def forward(self, spk_e):
        h = self.net(spk_e)
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)


class TTSStyleCanonicalPrior(nn.Module):
    """
    Conditional utterance-style prior p(u | z_c, r).

    z_c is pooled to an utterance-level summary so this prior models global
    style/prosody tendencies conditioned on content and speaker, without
    receiving frame-level alignment details directly.
    """
    def __init__(
        self,
        c_dim=80,
        spk_dim=64,
        latent_dim=64,
        hidden=256,
        dropout=0.1,
        logvar_bias=0.0,
    ):
        super().__init__()
        self.c_dim = int(c_dim)
        self.spk_dim = int(spk_dim)
        self.latent_dim = int(latent_dim)
        in_dim = 2 * self.c_dim + self.spk_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden, self.latent_dim)
        self.logvar = nn.Linear(hidden, self.latent_dim)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, float(logvar_bias))

    def forward(self, z_c, maskK, spk_e):
        mean, std = masked_mean_std_pool(z_c, maskK)
        spk = spk_e.to(device=z_c.device, dtype=z_c.dtype)
        h = self.net(torch.cat([mean, std, spk], dim=-1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)


class TTSStyleToSource(nn.Module):
    """
    Projects TTS-only style latent u and speaker condition r into a frame-level
    source bias P_u(u, r) used by g(c,u,r)=c+P_u(u,r).
    """
    def __init__(self, latent_dim=64, spk_dim=64, out_dim=80, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + spk_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u, spk_e):
        return self.net(torch.cat([u, spk_e.to(dtype=u.dtype)], dim=-1))


class CanonicalToSource(nn.Module):
    """
    Maps frame-level canonical latent c to the 80D FM source state y0.
    The FM ODE still runs in 80D; this module keeps canonical semantics out of
    the mel/source state while giving TTS a learnable source realization.
    """
    def __init__(self, c_dim=192, spk_dim=64, style_dim=0, out_dim=80, hidden=256, dropout=0.1):
        super().__init__()
        self.c_dim = int(c_dim)
        self.spk_dim = int(spk_dim)
        self.style_dim = int(max(0, style_dim))
        in_dim = self.c_dim + self.spk_dim + self.style_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, c_frame, maskK, spk_e=None, style_e=None):
        B, K, _ = c_frame.shape
        parts = [c_frame]
        if spk_e is None:
            spk = torch.zeros(B, self.spk_dim, device=c_frame.device, dtype=c_frame.dtype)
        else:
            spk = spk_e.to(device=c_frame.device, dtype=c_frame.dtype)
        parts.append(spk.unsqueeze(1).expand(B, K, self.spk_dim))
        if self.style_dim > 0:
            if style_e is None:
                style = torch.zeros(B, self.style_dim, device=c_frame.device, dtype=c_frame.dtype)
            else:
                style = style_e.to(device=c_frame.device, dtype=c_frame.dtype)
            parts.append(style.unsqueeze(1).expand(B, K, self.style_dim))
        y0 = self.net(torch.cat(parts, dim=-1))
        return y0 * maskK.float().unsqueeze(-1)


class SourceToCanonical(nn.Module):
    """
    Maps backward FM source endpoint y0_hat back into canonical latent space.
    CTC and canonical NLL operate on this output when true canonical latents are
    enabled.
    """
    def __init__(self, in_dim=80, c_dim=192, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, c_dim),
        )

    def forward(self, y0_frame, maskK):
        c = self.net(y0_frame)
        return c * maskK.float().unsqueeze(-1)


class CanonicalPosterior(nn.Module):
    """
    Frame-level Gaussian posterior q(c | s) on top of the backward canonical
    endpoint. It is initialized as a near-delta around the input endpoint so
    switching from point NLL to KL starts close to the previous behavior.
    """
    def __init__(self, dim=80, hidden=256, dropout=0.1, logvar_bias=-4.0):
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.delta_mu = nn.Linear(hidden, self.dim)
        self.logvar = nn.Linear(hidden, self.dim)
        nn.init.zeros_(self.delta_mu.weight)
        nn.init.zeros_(self.delta_mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, float(logvar_bias))

    def forward(self, c_point, maskK):
        h = self.net(c_point)
        mask = maskK.float().unsqueeze(-1)
        mu = (c_point + self.delta_mu(h)) * mask
        logvar = self.logvar(h).clamp(-8.0, 4.0) * mask
        return mu, logvar


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
        self.hidden_size = int(hidden_size)
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
        self.spk_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.style_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.spk_adaLN_modulation[-1].weight)
        nn.init.zeros_(self.spk_adaLN_modulation[-1].bias)
        nn.init.zeros_(self.style_adaLN_modulation[-1].weight)
        nn.init.zeros_(self.style_adaLN_modulation[-1].bias)

    def forward(self, x, c, cos, sin, key_padding_mask=None, spk_c=None, style_c=None):
        mod = self.adaLN_modulation(c)
        if spk_c is not None:
            mod = mod + self.spk_adaLN_modulation(spk_c)
        if style_c is not None:
            mod = mod + self.style_adaLN_modulation(style_c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
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
        self.spk_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.style_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c, spk_c=None, style_c=None):
        mod = self.adaLN_modulation(c)
        if spk_c is not None:
            mod = mod + self.spk_adaLN_modulation(spk_c)
        if style_c is not None:
            mod = mod + self.style_adaLN_modulation(style_c)
        shift, scale = mod.chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

class DiTVectorField(nn.Module):
    def __init__(
        self,
        D=80,
        E_spk=16,
        style_dim=0,
        text_cond_dim=0,
        hidden=512,
        depth=8,
        n_heads=8,
        dropout=0.0,
        max_len=4096,
        hidden_tap_index=3,
        condition_injection="legacy",
    ):
        super().__init__()
        self.D = D
        self.E_spk = E_spk
        self.style_dim = int(max(0, style_dim))
        self.text_cond_dim = int(max(0, text_cond_dim))
        self.hidden = hidden
        self.depth = depth
        self.hidden_tap_index = int(max(0, min(depth - 1, hidden_tap_index)))
        self.condition_injection = str(condition_injection).lower()
        if self.condition_injection not in {"legacy", "separate_adaln"}:
            raise ValueError(
                f"Unsupported condition_injection={condition_injection!r}; "
                "expected 'legacy' or 'separate_adaln'."
            )
        self.in_proj = nn.Linear(D, hidden)
        self.t_embedder = TimestepEmbedder(hidden)

        self.cfg_embedder = nn.Embedding(2, hidden)
        nn.init.zeros_(self.cfg_embedder.weight)

        self.spk_proj = nn.Sequential(
            nn.Linear(E_spk, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.style_proj = None
        if self.style_dim > 0:
            self.style_proj = nn.Sequential(
                nn.Linear(self.style_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            nn.init.zeros_(self.style_proj[-1].weight)
            nn.init.zeros_(self.style_proj[-1].bias)
        self.text_cond_proj = None
        if self.text_cond_dim > 0:
            self.text_cond_proj = nn.Sequential(
                nn.Linear(self.text_cond_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            nn.init.zeros_(self.text_cond_proj[-1].weight)
            nn.init.zeros_(self.text_cond_proj[-1].bias)

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
            nn.init.constant_(blk.spk_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.spk_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(blk.style_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.style_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.spk_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.spk_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.style_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.style_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if self.text_cond_proj is not None:
            nn.init.constant_(self.text_cond_proj[-1].weight, 0)
            nn.init.constant_(self.text_cond_proj[-1].bias, 0)
        if self.style_proj is not None:
            nn.init.constant_(self.style_proj[-1].weight, 0)
            nn.init.constant_(self.style_proj[-1].bias, 0)

    def forward(self, z, t, maskK, cfg_flag=None, spk_e=None, style_e=None, text_cond=None, return_hidden=False, hidden_tap_index=None):
        B, K, _ = z.shape
        x = self.in_proj(z)
        if self.text_cond_proj is not None and text_cond is not None:
            if text_cond.shape[0] != B or text_cond.shape[1] != K:
                raise ValueError(
                    f"text_cond shape {tuple(text_cond.shape[:2])} does not match latent shape {(B, K)}"
                )
            x = x + self.text_cond_proj(text_cond.to(dtype=x.dtype))
            x = x * maskK.float().unsqueeze(-1)
        c = self.t_embedder(t)

        if cfg_flag is None:
            cfg_flag = torch.ones(B, dtype=torch.long, device=z.device)
        c = c + self.cfg_embedder(cfg_flag)

        if not getattr(self, "direct_speaker_cond", True):
            spk_e = None
        if spk_e is None:
            spk_e = torch.zeros(B, self.E_spk, device=z.device, dtype=z.dtype)
        spk_c = self.spk_proj(spk_e.to(dtype=c.dtype))
        style_c = None
        if self.style_proj is not None and style_e is not None:
            style_c = self.style_proj(style_e.to(dtype=c.dtype))
        if self.condition_injection == "legacy":
            c = c + spk_c
            if style_c is not None:
                c = c + style_c
            spk_c = None
            style_c = None

        cos, sin = self.rope(x, K)
        key_padding_mask = ~maskK
        tap_idx = self.hidden_tap_index if hidden_tap_index is None else int(hidden_tap_index)
        tap_idx = max(0, min(tap_idx, len(self.blocks) - 1))
        hidden_tap = None

        for i, blk in enumerate(self.blocks):
            x = blk(x, c, cos, sin, key_padding_mask=key_padding_mask, spk_c=spk_c, style_c=style_c)
            if i == tap_idx:
                hidden_tap = x

        pred = self.final_layer(x, c, spk_c=spk_c, style_c=style_c)
        if return_hidden:
            if hidden_tap is None:
                hidden_tap = x
            return pred, hidden_tap
        return pred

def heun_integrate(vf, z0, maskK, steps=30, direction=+1, cfg_scale=1.0, spk_e=None, style_e=None, text_cond=None):
    z = z0
    dt = direction * (1.0 / steps)
    t0 = 0.0 if direction == +1 else 1.0
    B = z.shape[0]
    device = z.device

    if spk_e is None:
        spk_e = torch.zeros(B, vf.E_spk, device=device, dtype=z.dtype)
    if getattr(vf, "style_dim", 0) > 0 and style_e is None:
        style_e = torch.zeros(B, vf.style_dim, device=device, dtype=z.dtype)

    cfg_cond = torch.ones(B, dtype=torch.long, device=device)
    cfg_un   = torch.zeros(B, dtype=torch.long, device=device)
    spk_zero = torch.zeros_like(spk_e)
    style_zero = torch.zeros_like(style_e) if style_e is not None else None
    text_zero = torch.zeros_like(text_cond) if text_cond is not None else None

    def v_eval(z_now, t_now, cfg_flag, spk, style, text):
        t_tensor = torch.full((B,), float(t_now), device=device)
        return vf(z_now, t_tensor, maskK, cfg_flag=cfg_flag, spk_e=spk, style_e=style, text_cond=text)

    def v_mix(z_now, t_now):
        is_un = (spk_e.abs().sum(dim=1) < 1e-8)
        if style_e is not None:
            is_un = is_un & (style_e.abs().sum(dim=1) < 1e-8)
        if text_cond is not None:
            is_un = is_un & (text_cond.abs().sum(dim=(1, 2)) < 1e-8)
        if cfg_scale == 1.0:
            cfg = torch.where(is_un, cfg_un, cfg_cond)
            spk = torch.where(is_un[:, None], spk_zero, spk_e)
            style = None
            if style_e is not None:
                style = torch.where(is_un[:, None], style_zero, style_e)
            text = text_cond
            if text_cond is not None:
                text = torch.where(is_un[:, None, None], text_zero, text_cond)
            return v_eval(z_now, t_now, cfg, spk, style, text)

        v_c = v_eval(z_now, t_now, cfg_cond, spk_e, style_e, text_cond)
        v_u = v_eval(z_now, t_now, cfg_un,   spk_zero, style_zero, text_zero)
        return v_u + cfg_scale * (v_c - v_u)

    for i in range(steps):
        t = t0 + i * dt
        t_next = t + dt
        k1 = v_mix(z, t)
        z_pred = z + dt * k1
        k2 = v_mix(z_pred, t_next)
        z = z + dt * 0.5 * (k1 + k2)
    return z

def euler_integrate_grad(vf, z0, maskK, steps=10, direction=+1, cfg_flag_value=1, spk_e=None, style_e=None, text_cond=None):
    z = z0
    dt = direction * (1.0 / steps)
    t0 = 0.0 if direction == +1 else 1.0
    B = z.shape[0]
    device = z.device

    if spk_e is None:
        spk_e = torch.zeros(B, vf.E_spk, device=device, dtype=z.dtype)
    if getattr(vf, "style_dim", 0) > 0 and style_e is None:
        style_e = torch.zeros(B, vf.style_dim, device=device, dtype=z.dtype)
    cfg_flag = torch.full((B,), int(cfg_flag_value), dtype=torch.long, device=device)

    for i in range(steps):
        t = t0 + i * dt
        t_tensor = torch.full((B,), float(t), device=device)
        v = vf(z, t_tensor, maskK, cfg_flag=cfg_flag, spk_e=spk_e, style_e=style_e, text_cond=text_cond)
        z = z + dt * v
    return z

def euler_integrate(vf, z0, maskK, steps=10, direction=+1, cfg_flag_value=1, spk_e=None, style_e=None, text_cond=None):
    z = z0
    dt = direction * (1.0 / steps)
    t0 = 0.0 if direction == +1 else 1.0
    B = z.shape[0]
    device = z.device

    if spk_e is None:
        spk_e = torch.zeros(B, vf.E_spk, device=device, dtype=z.dtype)
    if getattr(vf, "style_dim", 0) > 0 and style_e is None:
        style_e = torch.zeros(B, vf.style_dim, device=device, dtype=z.dtype)
    cfg_flag = torch.full((B,), int(cfg_flag_value), dtype=torch.long, device=device)

    for i in range(steps):
        t = t0 + i * dt
        t_tensor = torch.full((B,), float(t), device=device)
        v = vf(z, t_tensor, maskK, cfg_flag=cfg_flag, spk_e=spk_e, style_e=style_e, text_cond=text_cond)
        z = z + dt * v
    return z
