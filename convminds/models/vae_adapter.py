from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)

class VaeBrainAdapter(nn.Module):
    """
    A VAE-based adapter that maps brain manifolds (reduced fMRI TRs) to 
    LLM latent representations.
    
    Architecture:
    Encoder: [4 x 1000] + PE -> Flatten(4000) -> 2048 -> 1024 ->mu/logvar(768)
    Decoder: 768 -> 1024 -> 2048 -> 4000
    """
    def __init__(
        self, 
        input_dim: int = 1000, 
        n_frames: int = 4, 
        latent_dim: int = 768,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_frames = n_frames
        self.flattened_dim = input_dim * n_frames
        self.latent_dim = latent_dim
        
        # Positional Encoding (initialized small)
        self.positional_params = nn.Parameter(torch.randn(n_frames, input_dim) * 0.01)
        
        # Encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.flattened_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, self.flattened_dim)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_frames, input_dim)
        """
        # Add Positional Encoding
        x = x + self.positional_params.unsqueeze(0)
        # Flatten
        x = x.view(x.shape[0], -1) # (batch, 4000)
        
        h = self.encoder_mlp(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_mlp(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "x_hat": x_hat,
            "x_orig": x.view(x.shape[0], -1)
        }


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Computes KL divergence loss."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


def info_nce_loss(z: torch.Tensor, h_text: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Computes Symmetric InfoNCE contrastive loss between brain (z) and text (h_text).
    """
    z = F.normalize(z, p=2, dim=-1)
    h_text = F.normalize(h_text, p=2, dim=-1)
    
    # Cosine similarities
    logits = torch.matmul(z, h_text.t()) / temperature
    
    # Labels (diagonal)
    labels = torch.arange(z.shape[0], device=z.device)
    
    # Cross entropy in both directions
    loss_br_to_tx = F.cross_entropy(logits, labels)
    loss_tx_to_br = F.cross_entropy(logits.t(), labels)
    
    return (loss_br_to_tx + loss_tx_to_br) / 2