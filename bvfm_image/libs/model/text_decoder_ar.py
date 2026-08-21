"""Autoregressive text decoder for FlowTok image-to-text.

A small causal transformer LM that cross-attends to the text latent Z_T
[B, 77, 16]. Unlike the non-autoregressive TextDecoder, output tokens are
conditioned on previously generated tokens, so the language-model prior
resolves the per-position multimodality that produces "a a a" degeneration:
the latent only has to supply the content, not the exact token at each slot.
"""

import torch
import torch.nn as nn

SOT_ID = 49406
EOT_ID = 49407


class ARBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask):
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(h)
        h = self.norm2(x)
        h, _ = self.cross_attn(h, memory, memory, need_weights=False)
        x = x + self.dropout(h)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ARTextDecoder(nn.Module):
    def __init__(self, latent_dim=16, d_model=768, depth=6, num_heads=8,
                 d_ff=3072, vocab_size=49408, seq_len=77, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.mem_proj = nn.Linear(latent_dim, d_model)
        self.mem_pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([
            ARBlock(d_model, num_heads, d_ff, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.mem_pos_emb, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _logits(self, x):
        # weight-tied output head
        return x @ self.tok_emb.weight.t()

    def forward(self, z, input_ids):
        """Teacher forcing.

        z: [B, seq_len, latent_dim] text latent (memory)
        input_ids: [B, T] decoder input tokens (caption shifted right)
        returns logits [B, T, vocab_size]
        """
        T = input_ids.shape[1]
        memory = self.mem_proj(z) + self.mem_pos_emb
        x = self.tok_emb(input_ids) + self.pos_emb[:, :T]
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        for block in self.blocks:
            x = block(x, memory, causal)
        return self._logits(self.norm(x))

    @torch.no_grad()
    def generate_beam(self, z, beam=3, max_len=None, length_penalty=1.0):
        """Beam search for a single sample. z: [1, L, D] -> ids [1, T]."""
        assert z.shape[0] == 1, "generate_beam handles one sample at a time"
        max_len = max_len or self.seq_len
        device = z.device
        z_k = z.expand(beam, -1, -1)
        ids = torch.full((1, 1), SOT_ID, dtype=torch.long, device=device)
        # first step: seed the beams from the top-k first tokens
        logp = torch.log_softmax(self.forward(z, ids)[:, -1].float(), dim=-1)
        scores, nxt = logp[0].topk(beam)
        ids = torch.cat([ids.expand(beam, -1),
                         nxt.unsqueeze(-1)], dim=-1)
        done = nxt == EOT_ID
        for _ in range(max_len - 2):
            if bool(done.all()):
                break
            logp = torch.log_softmax(self.forward(z_k, ids)[:, -1].float(), dim=-1)
            logp[done] = float("-inf")
            logp[done, 0] = 0.0  # finished beams extend with pad at no cost
            cand = scores.unsqueeze(-1) + logp  # [K, V]
            flat_scores, flat_idx = cand.reshape(-1).topk(beam)
            beam_idx = flat_idx // logp.shape[-1]
            tok_idx = flat_idx % logp.shape[-1]
            ids = torch.cat([ids[beam_idx], tok_idx.unsqueeze(-1)], dim=-1)
            scores = flat_scores
            done = done[beam_idx] | (tok_idx == EOT_ID)
        # length-normalized selection
        lengths = (ids != 0).sum(-1).float().clamp(min=1)
        best = (scores / lengths.pow(length_penalty)).argmax()
        return ids[best:best + 1]

    @torch.no_grad()
    def generate(self, z, max_len=None, temperature=0.0):
        """Greedy (temperature=0) batch decoding from SOT until EOT."""
        max_len = max_len or self.seq_len
        B = z.shape[0]
        device = z.device
        ids = torch.full((B, 1), SOT_ID, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = self.forward(z, ids)[:, -1]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.zeros_like(nxt), nxt)
            ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=-1)
            done = done | (nxt == EOT_ID)
            if bool(done.all()):
                break
        return ids
