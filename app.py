# app.py -----------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optimizer import run_nsga3
from data_load import load_scalers, load_model



st.set_page_config(page_title="NSMTGM – Alloy Optimizer", layout="wide")
st.title("NSMTGM – Multi‑objective Design of Fe‑based Metallic Glasses")


st.sidebar.header("Evolutionary Parameters")
#VALID_POPS = [3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136]
#NIND = st.sidebar.selectbox("Population size (NSGA‑III valid set)", VALID_POPS, index=VALID_POPS.index(28))
NIND = st.sidebar.number_input("Desired population size", 10, 2000, 200, 1)
MAXGEN = st.sidebar.number_input("Generations", 10, 1000, 100, 10)
SEED = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)
run_btn = st.sidebar.button("Run Optimization 🚀")


ELEMENTS = [
    'Fe','B','Si','P','C','Co','Nb','Ni','Mo','Zr','Ga','Al','Dy','Cu','Cr',
    'Y','Nd','Hf','Ti','Tb','Ho','Ta','Er','Sn','W','Tm','Gd','Sm','V','Pr'
]

def row_to_formula(row: pd.Series, decimals: int = 6, eps: float = 1e-8) -> str:
    vals = row[ELEMENTS].to_numpy(dtype=float)
    s = vals.sum()
    if s > 0:
        vals = vals / s

    comps = []
    for elem, frac in zip(ELEMENTS, vals):
        if frac > eps:
            comps.append(f"{elem}{frac*100:.{decimals}f}")
    return "".join(comps)

def draw_ellipse(ax, x, y, color="lightcoral"):
    
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_std,  y_std  = np.std(x),  np.std(y)
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((x_mean, y_mean), 4.4*x_std, 4.4*y_std,
                         edgecolor=color, facecolor=color, alpha=0.18, lw=0))



if run_btn:
    with st.spinner("Loading model & scalers …"):
        model   = load_model("MTWAE_latent8.pth", device="cpu")
        scalers = load_scalers("scalers")

    st.success("Loaded. Starting NSGA‑III …")
    df_raw = run_nsga3(model, scalers, pop_size=NIND, n_gen=MAXGEN, seed=SEED)

   
    df_display = pd.DataFrame({
        "idx": range(1, len(df_raw)+1),
        "Composition": df_raw.apply(row_to_formula, axis=1),
        "Bs (T)":        df_raw["Bs (T)"].round(4),
        "ln(Hc) (A/m)":  df_raw["ln(Hc) (A/m)"].round(4),
        "Hc (A/m)":      df_raw["Hc (A/m)"].round(2),
        "Dc (mm)":       df_raw["Dc (mm)"].round(3),
    })

    st.subheader("Pareto‑optimal candidate alloys")
    st.dataframe(df_display, use_container_width=True)

    
    df_export = df_raw.copy()
    df_export["Composition_exact"] = df_raw.apply(
        lambda row: row_to_formula(row, decimals=6, eps=1e-8), axis=1
    )
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "pareto_alloys.csv", mime="text/csv")

    
    Bs  = df_raw["Bs (T)"].values
    lnHc = df_raw["ln(Hc) (A/m)"].values
    Dc  = df_raw["Dc (mm)"].values

    
    fig1, ax1 = plt.subplots(figsize=(6,5))
    draw_ellipse(ax1, Bs, lnHc)
    ax1.scatter(Bs, lnHc, c="red", edgecolor="k", s=80, label="Pareto Front")
    ax1.axvline(1.5, ls="--", c="k"); ax1.axhline(1.5, ls="--", c="k")
    ax1.set_xlabel("Bs (T)"); ax1.set_ylabel("ln(Hc) (A/m)"); ax1.invert_yaxis()
    ax1.legend(); ax1.grid(alpha=.3)
    st.pyplot(fig1)

    
    fig2, ax2 = plt.subplots(figsize=(6,5))
    draw_ellipse(ax2, Bs, Dc)
    ax2.scatter(Bs, Dc, c="red", edgecolor="k", s=80)
    ax2.axvline(1.5, ls="--", c="k"); ax2.axhline(1.0, ls="--", c="k")
    ax2.set_xlabel("Bs (T)"); ax2.set_ylabel("Dc (mm)")
    ax2.grid(alpha=.3)
    st.pyplot(fig2)

    
    fig3, ax3 = plt.subplots(figsize=(6,5))
    draw_ellipse(ax3, Dc, lnHc)
    ax3.scatter(Dc, lnHc, c="red", edgecolor="k", s=80)
    ax3.axvline(1.0, ls="--", c="k"); ax3.axhline(1.5, ls="--", c="k")
    ax3.set_xlabel("Dc (mm)"); ax3.set_ylabel("ln(Hc) (A/m)"); ax3.invert_yaxis()
    ax3.grid(alpha=.3)
    st.pyplot(fig3)

    st.toast("Done ✔️  You can adjust parameters and re‑run!")
else:
    st.info(" **Run Optimization**")

