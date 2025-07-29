import torch
import torch.nn as nn

class MTWAE(nn.Module):
    def __init__(self, in_features: int = 30, latent_size: int = 8):
        super().__init__()
        # ============== Encoder / Decoder ==============
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 90), nn.LayerNorm(90), nn.LeakyReLU(),
            nn.Linear(90, 48),           nn.LayerNorm(48), nn.LeakyReLU(),
            nn.Linear(48, 30),           nn.LayerNorm(30), nn.LeakyReLU(),
            nn.Linear(30, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 30),  nn.LayerNorm(30), nn.LeakyReLU(),
            nn.Linear(30, 48),           nn.LayerNorm(48), nn.LeakyReLU(),
            nn.Linear(48, 90),           nn.LayerNorm(90), nn.LeakyReLU(),
            nn.Linear(90, in_features),
        )
        # ============== Property heads ==============
        def _head():
            return nn.Sequential(
                nn.Linear(latent_size, 90), nn.LayerNorm(90), nn.LeakyReLU(),
                nn.Linear(90, 90),           nn.LayerNorm(90), nn.LeakyReLU(),
                nn.Linear(90, 90),           nn.LayerNorm(90), nn.LeakyReLU(),
                nn.Linear(90, 1),
            )
        self.head_Bs = _head()
        self.head_Hc = _head()
        self.head_Dc = _head()

    # ---------- public API ----------
    def encode(self, x):          return self.encoder(x)
    def decode(self, z):          return torch.softmax(self.decoder(z), dim=1)
    def predict_Bs(self, z):      return self.head_Bs(z)
    def predict_Hc(self, z):      return self.head_Hc(z)
    def predict_Dc(self, z):      return self.head_Dc(z)

    def forward(self, x):
        z = self.encode(x)
        return ( self.decode(z),
                 z,
                 self.predict_Bs(z),
                 self.predict_Hc(z),
                 self.predict_Dc(z) )
