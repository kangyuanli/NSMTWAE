# models.py ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTWAE(nn.Module):
    """Multi‑Task Wasserstein Auto‑Encoder (与原实现一致)."""

    def __init__(self, in_features: int = 30, latent_size: int = 8):
        super().__init__()
        # ---------------- Encoder ----------------
        self.encoder_layer = nn.Sequential(
            nn.Linear(in_features, 90),
            nn.LayerNorm(90), nn.LeakyReLU(),
            nn.Linear(90, 48),
            nn.LayerNorm(48), nn.LeakyReLU(),
            nn.Linear(48, 30),
            nn.LayerNorm(30), nn.LeakyReLU(),
            nn.Linear(30, latent_size),
        )
        # ---------------- Decoder ----------------
        self.decoder_layer = nn.Sequential(
            nn.Linear(latent_size, 30),
            nn.LayerNorm(30), nn.LeakyReLU(),
            nn.Linear(30, 48),
            nn.LayerNorm(48), nn.LeakyReLU(),
            nn.Linear(48, 90),
            nn.LayerNorm(90), nn.LeakyReLU(),
            nn.Linear(90, in_features),
        )
        # ---------------- Predictors ------------
        def _make_head():  # 3×完全相同的前馈网络
            return nn.Sequential(
                nn.Linear(latent_size, 90), nn.LayerNorm(90), nn.LeakyReLU(),
                nn.Linear(90, 90), nn.LayerNorm(90), nn.LeakyReLU(),
                nn.Linear(90, 90), nn.LayerNorm(90), nn.LeakyReLU(),
                nn.Linear(90, 1),
            )

        self.head_Bs = _make_head()
        self.head_Hc = _make_head()
        self.head_Dc = _make_head()

    # ------------------------------------------------------------------ #
    def encoder(self, x): return self.encoder_layer(x)

    def decoder(self, z): return F.softmax(self.decoder_layer(z), dim=1)

    def forward(self, x):
        z = self.encoder(x)
        return {
            "recon": self.decoder(z),
            "z": z,
            "Bs": self.head_Bs(z),
            "Hc": self.head_Hc(z),
            "Dc": self.head_Dc(z),
        }

