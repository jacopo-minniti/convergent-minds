from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE contrastive loss between two modalities (e.g. brain and text).
    
    L = (L(z->h) + L(h->z)) / 2
    """
    def __init__(self, temperature: float = 0.07, learnable_temp: bool = True):
        super().__init__()
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        else:
            self.register_buffer("logit_scale", torch.log(torch.tensor(1 / temperature)))

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Brain latent representation (batch, dim)
            h: Text latent representation (batch, dim)
        """
        z = F.normalize(z, p=2, dim=-1)
        h = F.normalize(h, p=2, dim=-1)
        
        # Scaling factor (1/tau)
        logit_scale = self.logit_scale.exp()
        
        # Cosine similarities
        logits_per_z = torch.matmul(z, h.t()) * logit_scale
        logits_per_h = logits_per_z.t()
        
        # Labels are the diagonals
        labels = torch.arange(z.shape[0], device=z.device)
        
        loss_z = F.cross_entropy(logits_per_z, labels)
        loss_h = F.cross_entropy(logits_per_h, labels)
        
        return (loss_z + loss_h) / 2


class TripartiteVAELoss(nn.Module):
    """
    Combined loss for VAE with Contrastive Alignment:
    L = rec_weight * MSE + kl_weight * KL + align_weight * InfoNCE
    """
    def __init__(
        self, 
        rec_weight: float = 1.0, 
        kl_weight: float = 0.005, 
        align_weight: float = 0.5,
        temperature: float = 0.07
    ):
        super().__init__()
        self.rec_weight = rec_weight
        self.kl_weight = kl_weight
        self.align_weight = align_weight
        self.info_nce = InfoNCELoss(temperature=temperature)

    def forward(
        self, 
        recon_x: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor, 
        z: torch.Tensor, 
        h_text: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # 1. Reconstruction Loss (MSE per sample)
        # We use squared error summed over features and averaged over batch
        # This matches the scale of the KL divergence (which is also per sample)
        rec_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # 2. KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        # 3. Alignment Loss (InfoNCE)
        align_loss = self.info_nce(z, h_text)
        
        total_loss = (
            self.rec_weight * rec_loss + 
            self.kl_weight * kl_loss + 
            self.align_weight * align_loss
        )
        
        return {
            "loss": total_loss,
            "rec_loss": rec_loss,
            "kl_loss": kl_loss,
            "align_loss": align_loss
        }

def info_nce_loss(z: torch.Tensor, h: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Standalone InfoNCE contrastive loss."""
    # Symmetrize so that we align both directions (brain -> text and text -> brain)
    z = F.normalize(z, p=2, dim=-1)
    h = F.normalize(h, p=2, dim=-1)
    logits = torch.matmul(z, h.t()) / temperature
    labels = torch.arange(z.shape[0], device=z.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

def vae_reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 0.005) -> dict[str, torch.Tensor]:
    """Basic VAE reconstruction and KL loss."""
    rec_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    total_loss = rec_loss + kl_weight * kl_loss
    return {"loss": total_loss, "rec_loss": rec_loss, "kl_loss": kl_loss}
