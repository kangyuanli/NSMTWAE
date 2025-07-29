"""
把 geatpy‑NSGA‑III 优化过程封装成函数，便于 Streamlit 调用。
"""

import numpy as np, torch, multiprocessing, geatpy as ea, random, os
from pathlib import Path
from models import MTWAE
from data_utils import load_or_build_scalers

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "Joint_MTWAE_Model_latent_8_sigma_8_epoch_800.pth"

# -------------------- 固定随机性 --------------------
def _set_seed(seed=11):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------- geatpy 问题定义 --------------------
class NSMTGMProblem(ea.Problem):
    def __init__(self, model, scalers, sigma=8.0):
        self.model = model.eval()
        self.scalers = scalers  # (Bs,Hc,Dc) StandardScaler
        super().__init__(name='NSMTGM', M=3, Dim=8,
                         maxormins=[-1, 1, -1],   # Bs↑, lnHc↓, Dc↑
                         varTypes=[0]*8,
                         lb=[-sigma]*8, ub=[ sigma]*8,
                         lbin=[1]*8,    ubin=[1]*8)

    def evalVars(self, Z_np):
        """Z_np 形状 (N,8) → 返回 shape (N,3) 的目标矩阵"""
        Z = torch.from_numpy(Z_np).float()
        with torch.no_grad():
            Bs = self.model.predict_Bs(Z).numpy()
            Hc = self.model.predict_Hc(Z).numpy()
            Dc = self.model.predict_Dc(Z).numpy()
        Bs = self.scalers[0].inverse_transform(Bs)
        Hc = self.scalers[1].inverse_transform(Hc)
        Dc = self.scalers[2].inverse_transform(Dc)
        return np.hstack([Bs, Hc, Dc])

# -------------------- 核心接口 --------------------
def run_nsga3(NIND=200, MAXGEN=500, seed=11):
    _set_seed(seed)
    multiprocessing.set_start_method("spawn", force=True)

    # ---- 模型 & scaler ----
    scalers = load_or_build_scalers()
    model   = MTWAE()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    problem   = NSMTGMProblem(model, scalers)
    population = ea.Population(Encoding='RI', NIND=NIND)
    algorithm  = ea.moea_NSGA3_templet(problem, population,
                                       MAXGEN=MAXGEN, logTras=10, seed=seed)

    res = ea.optimize(algorithm, verbose=False, outputMsg=False,
                      saveFlag=False, drawing=0)
    return res  # dict 结构：{'ObjV','Vars',...}
