import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class ModelDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, z):
        return self.net(z)

class MultiEncoderSharedAutoencoder(nn.Module):
    def __init__(self, dims, latent_dim):
        super().__init__()
        self.encoders = nn.ModuleList([ModelEncoder(d, latent_dim) for d in dims])
        self.decoders = nn.ModuleList([ModelDecoder(latent_dim, d) for d in dims])

    def forward(self, embeddings):
        # embeddings: list of tensors, one per model [E_m shape: B x d_m]
        latents = [enc(e) for enc, e in zip(self.encoders, embeddings)]
        recons = [dec(z) for dec, z in zip(self.decoders, latents)]
        return latents, recons
