import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=timesteps.device).float() / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ConditionalFeatureDenoiser(nn.Module):
    def __init__(self, feature_dim, condition_dim, hidden_dim, diffusion_steps, time_dim=32, dropout=0.1):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim + condition_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, noisy_features, condition, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_dim).to(noisy_features.device)
        x = torch.cat([noisy_features, condition, time_emb], dim=-1)
        return self.net(x)


class GraphConditionedDiffusionDenoiser(nn.Module):
    def __init__(self, feature_dim, condition_dim, hidden_dim, diffusion_steps):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.text_denoiser = ConditionalFeatureDenoiser(
            feature_dim, condition_dim, hidden_dim, diffusion_steps
        )
        self.image_denoiser = ConditionalFeatureDenoiser(
            feature_dim, condition_dim, hidden_dim, diffusion_steps
        )
        betas = torch.linspace(1e-4, 2e-2, diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(torch.cumprod(alphas, dim=0)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - torch.cumprod(alphas, dim=0)),
        )

    def q_sample(self, clean_features, timesteps):
        noise = torch.randn_like(clean_features)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(-1)
        return sqrt_alpha * clean_features + sqrt_one_minus * noise

    def denoise_batch(self, text_features, image_features, condition, timesteps=None):
        if timesteps is None:
            timesteps = torch.zeros(
                text_features.shape[0], dtype=torch.long, device=text_features.device
            )
            text_noisy = text_features
            image_noisy = image_features
        else:
            text_noisy = self.q_sample(text_features, timesteps)
            image_noisy = self.q_sample(image_features, timesteps)

        text_clean = self.text_denoiser(text_noisy, condition, timesteps)
        image_clean = self.image_denoiser(image_noisy, condition, timesteps)
        return text_clean, image_clean

    def warmup_loss(self, text_features, image_features, condition, text_graph, image_graph, beta_graph):
        timesteps = torch.randint(
            0, self.diffusion_steps, (text_features.shape[0],), device=text_features.device
        )
        text_clean, image_clean = self.denoise_batch(text_features, image_features, condition, timesteps)
        diff_loss = F.mse_loss(text_clean, text_features) + F.mse_loss(image_clean, image_features)
        graph_loss = F.mse_loss(text_clean, text_graph) + F.mse_loss(image_clean, image_graph)
        return diff_loss + beta_graph * graph_loss, diff_loss.detach(), graph_loss.detach()
