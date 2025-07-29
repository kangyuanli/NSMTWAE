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


def load_model(ckpt_path="checkpoints/MTWAE_latent8.pth", device="cpu"):
    from pathlib import Path, PurePosixPath
    import torch, traceback, sys
    model = MTWAE(in_features=30, latent_size=8)
    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)     # ⚠️ 出错点
    except Exception as e:
        # 把真正的报错打印到前端
        import streamlit as st
        st.error(f"❌  权重加载失败: {e}")
        st.caption("完整 traceback ↓")
        st.code(traceback.format_exc())
        sys.exit(0)
    model.to(device).eval()
    return model


