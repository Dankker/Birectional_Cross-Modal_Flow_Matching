"""This file contains the model definition of TA-TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch.cuda.amp import autocast

from libs.model.blocks import FlowTiTokEncoder, FlowTiTokDecoder


class DiagonalGaussianDistribution(object):
    @autocast(enabled=False)
    def __init__(self, parameters, deterministic=False):
        """Initializes a Gaussian distribution instance given the parameters.

        Args:
            parameters (torch.Tensor): The parameters for the Gaussian distribution. It is expected
                to be in shape [B, 2 * C, *], where B is batch size, and C is the embedding dimension.
                First C channels are used for mean and last C are used for logvar in the Gaussian distribution.
            deterministic (bool): Whether to use deterministic sampling. When it is true, the sampling results
                is purely based on mean (i.e., std = 0).
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    @autocast(enabled=False)
    def sample(self):
        x = self.mean.float() + self.std.float() * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    @autocast(enabled=False)
    def mode(self):
        return self.mean

    @autocast(enabled=False)
    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean.float(), 2)
                                    + self.var.float() - 1.0 - self.logvar.float(),
                                    dim=[1, 2])


class FlowTiTok(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder = FlowTiTokEncoder(config)
        self.decoder = FlowTiTokDecoder(config)

        self.num_latent_tokens = config.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        self.quantize = DiagonalGaussianDistribution

        self.return_quantized = config.vq_model.get("return_quantized", True)
        self.use_pretrained = config.vq_model.get("use_pretrained", True)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def extract_latents(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        if not self.return_quantized:
            z = z.squeeze(2).permute(0,2,1)
            if self.use_pretrained:
                z = z.view(z.shape[0], z.shape[1], self.encoder.token_size // 2, -1)
                z = z.mean(-1)
            return z
        else:
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            z_quantized = z_quantized.squeeze(2).permute(0,2,1)
            return z_quantized

    def encode(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        posteriors = self.quantize(z)
        z_quantized = posteriors.sample()
        result_dict = posteriors
        return z_quantized, result_dict

    def decode(self, z_quantized, text_guidance):
        decoded = self.decoder(z_quantized, text_guidance)
        return decoded
    
    def decode_tokens(self, tokens, text_guidance):
        z_quantized = tokens
        decoded = self.decode(z_quantized, text_guidance)
        return decoded
    
    def forward(self, x, text_guidance):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized, text_guidance)
        return decoded, result_dict
