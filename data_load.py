# data_load.py -----------------------------------------------------------
import pickle
from pathlib import Path
import torch
from sklearn.preprocessing import StandardScaler
from models import MTWAE


def load_scalers(dir_path: str = "scalers"):
    
    scalers = {}
    for target in ["Bs", "Hc", "Dc"]:
        with open(Path(dir_path) / f"{target}_scaler.pkl", "rb") as fp:
            scalers[target] = pickle.load(fp)            # type: StandardScaler
    return scalers


# data_load.py ----------------------------------------------------------
def load_model(ckpt_path="MTWAE_latent8.pth", device="cpu"):
    import torch, re
    model = MTWAE(in_features=30, latent_size=8)

    raw_state = torch.load(ckpt_path, map_location=device)

    # 
    key_map = {
        r"^Predicted_Bs_layer\.": "head_Bs.",
        r"^Predicted_Hc_layer\.": "head_Hc.",
        r"^Predicted_Dc_layer\.": "head_Dc.",
    }

    renamed_state = {}
    for k, v in raw_state.items():
        new_key = k
        for pattern, repl in key_map.items():
            new_key = re.sub(pattern, repl, new_key)
        renamed_state[new_key] = v

    
    model.load_state_dict(renamed_state, strict=True)
    return model.to(device).eval()



