import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple Residual block for deep feature alignment."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        return x + self.net(x)

class BrainEncoder(nn.Module):
    """
    Base Encoder for Brain-to-Latent mapping.
    Maps N frames of brain PCA components to a hidden representation.
    """
    def __init__(self, input_dim=1000, n_frames=4, hidden_dim=1000):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * n_frames, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, hidden_dim)
        )

    def forward(self, x):
        # x: (Batch, N, 1000)
        x = x.view(x.size(0), -1)
        return self.encoder(x)

class VaeBrainEncoder(nn.Module):
    """
    VAE-based Encoder that produces Mu and LogVar for reconstruction tasks.
    """
    def __init__(self, input_dim=1000, n_frames=4, latent_dim=1000):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim * n_frames, 2048),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.base(x)
        return self.fc_mu(h), self.fc_logvar(h)

class BrainDecoder(nn.Module):
    """
    Decodes the latent representation back to the brain PCA space.
    Used for the VAE reconstruction task.
    """
    def __init__(self, hidden_dim=1000, output_dim=1000, n_frames=4):
        super().__init__()
        self.n_frames = n_frames
        self.output_dim = output_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, output_dim * n_frames)
        )

    def forward(self, z):
        recon = self.decoder(z)
        return recon.view(z.size(0), self.n_frames, self.output_dim)

class BrainLanguageAdapter(nn.Module):
    """
    The main adapter that maps brain activity to LLM latent space.
    Can be configured to use a VaeEncoder or a standard MLP stem.
    """
    def __init__(self, input_dim=1000, n_frames=4, hidden_dim=1000, llm_dim=768, use_vae=False):
        super().__init__()
        self.use_vae = use_vae
        
        if use_vae:
            self.stem = VaeBrainEncoder(input_dim, n_frames, hidden_dim)
            self.decoder = BrainDecoder(hidden_dim, input_dim, n_frames)
        else:
            self.stem = BrainEncoder(input_dim, n_frames, hidden_dim)
            self.decoder = None
            
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, llm_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.use_vae:
            mu, logvar = self.stem(x)
            z = self.reparameterize(mu, logvar) if self.training else mu
            x_hat = self.decoder(z)
            h_llm = self.head(z)
            return {
                "h_llm": h_llm,
                "x_hat": x_hat,
                "mu": mu,
                "logvar": logvar,
                "z": z,
                "x_orig": x
            }
        else:
            z = self.stem(x)
            h_llm = self.head(z)
            return {"h_llm": h_llm, "z": z}
