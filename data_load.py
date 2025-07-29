# data_load.py -----------------------------------------------------------
import pickle
from pathlib import Path
import torch
from sklearn.preprocessing import StandardScaler
from models import MTWAE


def load_scalers(dir_path: str = "scalers"):
    """读取事先持久化的三个 StandardScaler."""
    scalers = {}
    for target in ["Bs", "Hc", "Dc"]:
        with open(Path(dir_path) / f"{target}_scaler.pkl", "rb") as fp:
            scalers[target] = pickle.load(fp)            # type: StandardScaler
    return scalers


def load_model(
    ckpt_path: str = "MTWAE_latent8.pth",
    device: str | torch.device = "cpu",
):
    model = MTWAE(in_features=30, latent_size=8)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

