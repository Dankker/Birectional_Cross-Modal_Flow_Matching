import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp

import ml_collections
import torch.utils.checkpoint
import open_clip

from .trans_autoencoder import FlowEncoder, Adaptor


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    CrossFlow: update it for CFG with indicator
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels):
        embeddings = self.embedding_table(labels.int())
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        return torch.utils.checkpoint.checkpoint(self._forward, x, c)
        # return self._forward(x, c)
    
    def _forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ConditionalResidualAdapter(nn.Module):
    """Zero-initializable residual branch with a continuous condition."""

    def __init__(self, hidden_size, rank, out_size=None):
        super().__init__()
        out_size = hidden_size if out_size is None else out_size
        self.norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.down_x = nn.Linear(hidden_size, rank, bias=False)
        self.down_c = nn.Linear(hidden_size, rank, bias=True)
        self.up = nn.Linear(rank, out_size, bias=False)

    def forward(self, x, c):
        hidden = self.down_x(self.norm(x))
        hidden = hidden + self.down_c(c).unsqueeze(1)
        return self.up(F.silu(hidden))


class I2TResidualAdapter(ConditionalResidualAdapter):
    """Backward-compatible name for the earlier task-1-only baseline."""


class BVFMVariationProjector(nn.Module):
    """Project one utterance/pair-level variation sample into DiT space.

    This is deliberately not a direction or task embedding.  The caller must
    obtain ``z_v`` from the full-pair posterior during training, the text prior
    for forward T2I integration, or the image prior for reverse I2T integration.
    """

    def __init__(self, latent_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, variation_latent):
        return self.net(variation_latent)


class FlowTok(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        config,
        num_latent_tokens=77,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=2, # for cfg indicator
    ):
        super().__init__()
        self.in_channels = config.channels
        self.out_channels = self.in_channels
        self.num_heads = num_heads
        self.num_latent_tokens = num_latent_tokens

        self.x_embedder = nn.Linear(self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # Optional asymmetric task latent for a single shared bidirectional
        # vector field. T2I is the zero sentinel, so enabling this branch and
        # loading a released T2I checkpoint leaves the original T2I function
        # exactly unchanged. I2T adds one learned z_v vector to the existing
        # time/CFG conditioning. The task is known at inference time, so this
        # has no q/p train-inference mismatch.
        self.use_task_condition = bool(
            getattr(config, "use_task_condition", False))
        if self.use_task_condition:
            self.i2t_task_embedding = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("i2t_task_embedding", None)

        self.use_task_adapters = bool(
            getattr(config, "use_task_adapters", False))
        if self.use_task_adapters and not self.use_task_condition:
            raise ValueError(
                "use_task_adapters requires use_task_condition=True")
        self.i2t_adapter_scale = float(
            getattr(config, "i2t_adapter_scale", 1.0))
        if self.use_task_adapters:
            adapter_rank = int(getattr(config, "i2t_adapter_rank", 128))
            if adapter_rank <= 0:
                raise ValueError("i2t_adapter_rank must be positive")
            self.i2t_adapters = nn.ModuleList([
                I2TResidualAdapter(hidden_size, adapter_rank)
                for _ in range(depth)
            ])
            self.i2t_output_adapter = I2TResidualAdapter(
                hidden_size, adapter_rank, self.out_channels)
        else:
            self.i2t_adapters = nn.ModuleList()
            self.i2t_output_adapter = None

        # Paper-faithful BVFM conditioning.  The same conditional residual
        # vector field is evaluated for both integration directions; there is
        # no task id and no direction embedding.  Zero-initialized residual
        # outputs make a released FlowTok checkpoint an exact initialization.
        self.use_bvfm_condition = bool(
            getattr(config, "use_bvfm_condition", False))
        if self.use_bvfm_condition and (
                self.use_task_condition or self.use_task_adapters):
            raise ValueError(
                "BVFM variation conditioning cannot be combined with "
                "task/direction conditioning")
        self.bvfm_adapter_scale = float(
            getattr(config, "bvfm_adapter_scale", 1.0))
        if self.use_bvfm_condition:
            self.bvfm_latent_dim = int(
                getattr(config, "bvfm_latent_dim", 128))
            bvfm_rank = int(getattr(config, "bvfm_adapter_rank", 128))
            if self.bvfm_latent_dim <= 0 or bvfm_rank <= 0:
                raise ValueError(
                    "bvfm_latent_dim and bvfm_adapter_rank must be positive")
            self.bvfm_variation_projector = BVFMVariationProjector(
                self.bvfm_latent_dim, hidden_size)
            self.bvfm_adapters = nn.ModuleList([
                ConditionalResidualAdapter(hidden_size, bvfm_rank)
                for _ in range(depth)
            ])
            self.bvfm_output_adapter = ConditionalResidualAdapter(
                hidden_size, bvfm_rank, self.out_channels)
        else:
            self.bvfm_latent_dim = 0
            self.bvfm_variation_projector = None
            self.bvfm_adapters = nn.ModuleList()
            self.bvfm_output_adapter = None
        
        self.use_t2t_temperature = config.use_t2t_temperature
        if self.use_t2t_temperature:
            self.t2t_temperature = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        else:
            self.t2t_temperature = None

        self.pos_embed = nn.Parameter(torch.zeros(1, num_latent_tokens, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

        self.context_encoder = FlowEncoder(d_model=config.clip_dim, N=config.textVAE.num_blocks,
                                            head_num=config.textVAE.num_attention_heads, d_ff=config.textVAE.hidden_dim, 
                                            latten_size=config.channels * 2, dropout=config.textVAE.dropout_prob, last_norm=False)
        
        self.context_projector = nn.Sequential(
            nn.Linear(config.clip_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, config.channels),
        )
            
        if config.textVAE.clip_loss_weight > 0.0:
            self.open_clip, _, self.open_clip_preprocess = open_clip.create_model_and_transforms('ViT-L-16-SigLIP-256', pretrained=None)
            self.open_clip_output = Adaptor(input_dim=1024, tar_dim=num_latent_tokens*config.channels)

            del self.open_clip.text
            del self.open_clip.logit_bias


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.num_latent_tokens))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Task-1 adapters must be exact zero residuals at initialization.
        for adapter in self.i2t_adapters:
            nn.init.constant_(adapter.up.weight, 0)
        if self.i2t_output_adapter is not None:
            nn.init.constant_(self.i2t_output_adapter.up.weight, 0)

        # BVFM starts at the released unconditional vector field.  These are
        # shared residuals, however: once trained they are used by both 0->1
        # and 1->0 integrations with the corresponding sampled z_v.
        for adapter in self.bvfm_adapters:
            nn.init.constant_(adapter.up.weight, 0)
        if self.bvfm_output_adapter is not None:
            nn.init.constant_(self.bvfm_output_adapter.up.weight, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _forward(
            self, x, t, null_indicator, task_id=None,
            variation_latent=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(null_indicator)    # (N, D)
        c = t + y                                # (N, D)
        task_mask = None
        if self.use_task_condition:
            if task_id is None:
                task_id = torch.zeros(
                    x.shape[0], device=x.device, dtype=x.dtype)
            task_id = task_id.to(device=x.device).reshape(-1)
            if task_id.shape[0] != x.shape[0]:
                raise ValueError(
                    f"task_id batch {task_id.shape[0]} != input batch {x.shape[0]}")
            if bool(((task_id != 0) & (task_id != 1)).any()):
                raise ValueError("task_id values must be 0 (T2I) or 1 (I2T)")
            task_mask = task_id.to(dtype=x.dtype).reshape(-1, 1)
            c = c + task_mask * self.i2t_task_embedding.to(
                dtype=x.dtype).unsqueeze(0)
        elif task_id is not None and bool((task_id != 0).any()):
            raise ValueError(
                "I2T task requested but task conditioning is disabled")

        bvfm_condition = None
        if variation_latent is not None:
            if not self.use_bvfm_condition:
                raise ValueError(
                    "variation_latent was provided but BVFM is disabled")
            variation_latent = variation_latent.to(
                device=x.device, dtype=x.dtype)
            if variation_latent.ndim != 2:
                raise ValueError(
                    "variation_latent must have shape [batch, latent_dim]")
            if variation_latent.shape != (
                    x.shape[0], self.bvfm_latent_dim):
                raise ValueError(
                    "variation_latent shape "
                    f"{tuple(variation_latent.shape)} != "
                    f"({x.shape[0]}, {self.bvfm_latent_dim})")
            bvfm_condition = c + self.bvfm_variation_projector(
                variation_latent)

        adapters_active = (
            self.use_task_adapters
            and task_mask is not None
            and bool((task_mask != 0).any()))
        if bvfm_condition is not None:
            for block, adapter in zip(self.blocks, self.bvfm_adapters):
                x = block(x, c)
                x = x + self.bvfm_adapter_scale * adapter(
                    x, bvfm_condition)
            output = self.final_layer(x, c)
            output = output + (
                self.bvfm_adapter_scale
                * self.bvfm_output_adapter(x, bvfm_condition))
        elif adapters_active:
            residual_mask = task_mask.unsqueeze(-1)
            for block, adapter in zip(self.blocks, self.i2t_adapters):
                x = block(x, c)
                x = x + residual_mask * self.i2t_adapter_scale * adapter(x, c)
            output = self.final_layer(x, c)
            output = output + (
                residual_mask * self.i2t_adapter_scale
                * self.i2t_output_adapter(x, c))
        else:
            # This is deliberately the released operation sequence.  Task 0
            # never evaluates or adds an adapter residual.
            for block in self.blocks:
                x = block(x, c)
            output = self.final_layer(x, c)
        return [output]
    
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def _text_encoder(self, condition_context):
        # [B, 77, 768] -> [B, 77, 16]
        output = self.context_encoder(condition_context)
        mu, log_var = torch.chunk(output, 2, dim=-1)        
        z = self._reparameterize(mu, log_var)
        return [z, mu, log_var]

    def _text_projector(self, condition_context):
        z = self.context_projector(condition_context)

        return z, self.t2t_temperature
    
    def _img_clip(self, image_input):
        image_latent = self.open_clip.encode_image(image_input)
        image_latent = self.open_clip_output(image_latent)

        return image_latent, self.open_clip.logit_scale
    
    def forward(self, x, t = None, text_encoder=False,
                text_projector=False, image_clip=False,
                null_indicator=None, task_id=None,
                variation_latent=None):
        if text_encoder:
            return self._text_encoder(condition_context = x)
        elif text_projector:
            return self._text_projector(condition_context = x)
        elif image_clip:
            return self._img_clip(image_input = x) 
        else:
            return self._forward(
                x=x, t=t, null_indicator=null_indicator, task_id=task_id,
                variation_latent=variation_latent)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def FlowTok_H(config, **kwargs):
    return FlowTok(config=config, depth=36, hidden_size=1280, num_heads=20, **kwargs)

def FlowTok_XL(config, **kwargs):
    return FlowTok(config=config, depth=28, hidden_size=1152, num_heads=16, **kwargs)

def FlowTok_L(config, **kwargs):
    return FlowTok(config=config, depth=24, hidden_size=1024, num_heads=16, **kwargs)

def FlowTok_B(config, **kwargs):
    return FlowTok(config=config, depth=12, hidden_size=768, num_heads=12, **kwargs)

def FlowTok_S(config, **kwargs):
    return FlowTok(config=config, depth=12, hidden_size=384, num_heads=6, **kwargs)
