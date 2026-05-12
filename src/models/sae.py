import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalSAE(nn.Module):
    def __init__(self, d_model, n_features=256):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def loss(self, x, l1_lambda=1e-3):
        recon, z = self.forward(x)
        mse = F.mse_loss(recon, x)
        l1 = z.abs().mean()
        return mse + l1_lambda * l1, {"mse": mse.item(), "l1": l1.item()}

