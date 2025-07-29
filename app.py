import streamlit as st, pandas as pd, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from optimizer import run_nsga3
from models import MTWAE

ELEMENTS = [
    'Fe','B','Si','P','C','Co','Nb','Ni','Mo','Zr','Ga','Al',
    'Dy','Cu','Cr','Y','Nd','Hf','Ti','Tb','Ho','Ta','Er','Sn',
    'W','Tm','Gd','Sm','V','Pr'
]

# --------------------------- 组分格式化 ---------------------------
def comp2str(vec: np.ndarray, thr=1e-3):
    txt = ""
    for e, frac in zip(ELEMENTS, vec):
        if frac > thr:
            txt += f"{e}{frac*100:.2f}"
    return txt or "—"

# --------------------------- 主 UI ---------------------------
st.set_page_config(page_title="NSMTGM Optimizer", layout="wide")
st.title("NSMTGM ▶ Multi‑objective Generative Optimization")

with st.sidebar:
    st.header("Evolution parameters")
    NIND   = st.number_input("Population size", 10, 1000, 200, 10)
    MAXGEN = st.number_input("Generations",      10, 2000, 500, 10)
    seed   = st.number_input("Random seed",       1,  999, 11, 1)
    run_btn = st.button("🚀  Start optimization")

if run_btn:
    with st.spinner("Running NSGA‑III …"):
        res = run_nsga3(NIND=NIND, MAXGEN=MAXGEN, seed=seed)
    st.success("Optimization finished!")

    obj   = res['ObjV'];  vars_ = res['Vars']
    rank  = (res['CV'])[:,0]   # geatpy 内部返回

    # ---------- Pareto 分层 ----------
    NDSet = (rank == 0).astype(int)  # 第一层 0；其余 >0
    front1 = obj[NDSet==1]; front2 = obj[NDSet==0]

    # ---------- 3 视图 ----------
    def scatter(x,y, xlabel,ylabel):
        fig, ax = plt.subplots()
        ax.scatter(x, y, c="tab:red", label="1st Pareto", edgecolor="k")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend()
        st.pyplot(fig)

    scatter(front1[:,0], front1[:,1], "Bs (T)", "ln(Hc)")
    scatter(front1[:,0], front1[:,2], "Bs (T)", "Dc (mm)")
    scatter(front1[:,2], front1[:,1], "Dc (mm)", "ln(Hc)")

    # ---------- DataFrame 展示 ----------
    z_front1 = vars_[NDSet==1]
    # 再次调用模型一次拿 composition（避免重复导入 scalers）
    model = MTWAE(); model.load_state_dict(
        torch.load(Path(__file__).parent/"Joint_MTWAE_Model_latent_8_sigma_8_epoch_800.pth",
                   map_location="cpu"))
    comp = model.decode(torch.tensor(z_front1).float()).detach().numpy()
    comp[comp < 0.001] = 0  # 去噪
    df = pd.DataFrame({
        "Composition": [comp2str(c) for c in comp],
        "Bs (T)": front1[:,0],
        "ln(Hc)": front1[:,1],
        "Dc (mm)": front1[:,2],
    })
    st.subheader("1st Pareto front (filtered Bs>1.5, lnHc<1.5, Dc>1)")
    st.dataframe(df.query("`Bs (T)`>1.5 & `ln(Hc)`<1.5 & `Dc (mm)`>1"))
    st.subheader("Full 1st Pareto front")
    st.dataframe(df)
