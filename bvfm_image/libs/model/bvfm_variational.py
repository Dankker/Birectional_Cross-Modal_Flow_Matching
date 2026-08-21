"""Variational heads for bidirectional Flow Matching (BVFM).

The vector field itself lives in ``flowtok_t2i.py``.  These training and
inference heads implement Eq. (6)--(8) of the BVFM paper for paired image/text
latents:

  q(z_v | z_text, z_image), p_text(z_v | z_text), p_image(z_v | z_image).

All three distributions are diagonal Gaussians and produce one global z_v per
pair.  Neither the heads nor z_v contain an integration-direction label.
"""

import torch
import torch.nn as nn


def mean_std_pool(tokens):
    """Pool fixed-length latent tokens without discarding their dispersion."""
    tokens = tokens.float()
    mean = tokens.mean(dim=1)
    variance = (tokens - mean.unsqueeze(1)).pow(2).mean(dim=1)
    std = torch.sqrt(variance.clamp_min(1e-6))
    return torch.cat([mean, std], dim=-1)


class DiagonalGaussianHead(nn.Module):
    def __init__(
            self, input_dim, latent_dim=128, hidden_dim=256,
            dropout=0.1, logvar_bias=-2.0):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(hidden_dim, self.latent_dim)

        # All three distributions start aligned.  The KL is therefore finite
        # and zero at initialization, while the sampled latent is non-degenerate.
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, float(logvar_bias))

    def forward(self, features):
        hidden = self.backbone(features)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-8.0, 4.0)
        return mu, logvar


class BVFMVariationalHeads(nn.Module):
    """Full-pair posterior plus the two one-sided inference priors."""

    def __init__(
            self, token_dim=16, latent_dim=128, hidden_dim=256,
            dropout=0.1, logvar_bias=-2.0):
        super().__init__()
        endpoint_dim = 2 * int(token_dim)
        common = dict(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            logvar_bias=logvar_bias)
        self.pair_posterior = DiagonalGaussianHead(
            2 * endpoint_dim, **common)
        self.text_prior = DiagonalGaussianHead(endpoint_dim, **common)
        self.image_prior = DiagonalGaussianHead(endpoint_dim, **common)
        self.latent_dim = int(latent_dim)

    def posterior(self, text_latent, image_latent):
        text_stats = mean_std_pool(text_latent)
        image_stats = mean_std_pool(image_latent)
        return self.pair_posterior(
            torch.cat([text_stats, image_stats], dim=-1))

    def prior_from_text(self, text_latent):
        return self.text_prior(mean_std_pool(text_latent))

    def prior_from_image(self, image_latent):
        return self.image_prior(mean_std_pool(image_latent))

    @staticmethod
    def sample(mu, logvar, temperature=1.0):
        if float(temperature) <= 0.0:
            return mu
        return mu + float(temperature) * torch.exp(
            0.5 * logvar) * torch.randn_like(mu)

    @staticmethod
    def kl(q_mu, q_logvar, p_mu, p_logvar):
        """Dimension-normalized D_KL(q || p) for diagonal Gaussians.

        The FlowTok velocity reconstruction uses ``F.mse_loss`` and is thus
        averaged over all token/channel dimensions.  Averaging the diagonal KL
        over z_v dimensions keeps Eq. (17)'s terms on matching discretization
        scales instead of multiplying the KL strength by ``latent_dim``.
        """
        q_var = q_logvar.exp()
        p_var = p_logvar.exp()
        per_dim = 0.5 * (
            p_logvar - q_logvar
            + (q_var + (q_mu - p_mu).pow(2)) / p_var
            - 1.0)
        return per_dim.mean(dim=-1).mean()

    def all_distributions(self, text_latent, image_latent):
        q_mu, q_logvar = self.posterior(text_latent, image_latent)
        text_mu, text_logvar = self.prior_from_text(text_latent)
        image_mu, image_logvar = self.prior_from_image(image_latent)
        return {
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "text_mu": text_mu,
            "text_logvar": text_logvar,
            "image_mu": image_mu,
            "image_logvar": image_logvar,
        }
