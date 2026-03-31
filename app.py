# app.py -----------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from optimizer import run_nsga3
from data_load import load_scalers, load_model


st.set_page_config(page_title="NSMTGM – Alloy Optimizer", layout="wide")
st.title("NSMTGM – Multi-objective Design of Fe-based Metallic Glasses")

st.sidebar.header("Evolutionary Parameters")
NIND = st.sidebar.number_input("Desired population size", 10, 2000, 200, 1)
MAXGEN = st.sidebar.number_input("Generations", 10, 1000, 100, 10)
SEED = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)

st.sidebar.header("Display / Post-processing")
CUTOFF_AT_PERCENT = st.sidebar.number_input(
    "Cutoff threshold (at.%)", min_value=0.00, max_value=1.00, value=0.05, step=0.01, format="%.2f"
)
DISPLAY_DECIMALS = st.sidebar.number_input(
    "Displayed composition decimals", min_value=1, max_value=6, value=2, step=1
)
run_btn = st.sidebar.button("Run Optimization 🚀")

ELEMENTS = [
    'Fe','B','Si','P','C','Co','Nb','Ni','Mo','Zr','Ga','Al','Dy','Cu','Cr',
    'Y','Nd','Hf','Ti','Tb','Ho','Ta','Er','Sn','W','Tm','Gd','Sm','V','Pr'
]


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Row-wise normalization to sum to 1."""
    arr = np.asarray(arr, dtype=np.float64).copy()
    row_sums = arr.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    arr = arr / safe_sums
    return arr


def clean_compositions(df_comp: pd.DataFrame, cutoff_at_percent: float = 0.05) -> pd.DataFrame:
    """
    1) Zero out elements below cutoff_at_percent
    2) Renormalize each composition to sum to 1
    """
    cutoff_frac = cutoff_at_percent / 100.0
    arr = df_comp[ELEMENTS].to_numpy(dtype=np.float64).copy()

    arr[arr < cutoff_frac] = 0.0
    arr = normalize_rows(arr)

    return pd.DataFrame(arr, columns=ELEMENTS, index=df_comp.index)


def row_to_formula(row: pd.Series, decimals: int = 2, eps: float = 1e-12) -> str:
    """Convert one composition row to formula string."""
    vals = row[ELEMENTS].to_numpy(dtype=float)
    s = vals.sum()
    if s > 0:
        vals = vals / s

    comps = []
    for elem, frac in zip(ELEMENTS, vals):
        if frac > eps:
            comps.append(f"{elem}{frac*100:.{decimals}f}")
    return "".join(comps)


@torch.no_grad()
def predict_from_composition_df(model, scalers, df_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate Bs / ln(Hc) / Hc / Dc from composition vectors:
    composition -> encoder -> property heads
    """
    device = next(model.parameters()).device
    X = torch.tensor(df_comp[ELEMENTS].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

    z = model.encoder(X)
    pre_Bs = model.head_Bs(z).cpu().numpy()
    pre_Hc = model.head_Hc(z).cpu().numpy()
    pre_Dc = model.head_Dc(z).cpu().numpy()

    Bs = scalers["Bs"].inverse_transform(pre_Bs).reshape(-1)
    lnHc = scalers["Hc"].inverse_transform(pre_Hc).reshape(-1)
    Dc = scalers["Dc"].inverse_transform(pre_Dc).reshape(-1)

    return pd.DataFrame({
        "Bs (T)": Bs,
        "ln(Hc) (A/m)": lnHc,
        "Hc (A/m)": np.exp(lnHc),
        "Dc (mm)": Dc,
    }, index=df_comp.index)


def draw_ellipse(ax, x, y, color="lightcoral"):
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_std, y_std = np.std(x), np.std(y)

    # Avoid degenerate ellipse when only one point or nearly zero variance
    x_std = max(x_std, 1e-6)
    y_std = max(y_std, 1e-6)

    from matplotlib.patches import Ellipse
    ax.add_patch(
        Ellipse(
            (x_mean, y_mean),
            4.4 * x_std,
            4.4 * y_std,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
            lw=0,
        )
    )


if run_btn:
    with st.spinner("Loading model & scalers …"):
        model = load_model("MTWAE_latent8.pth", device="cpu")
        scalers = load_scalers("scalers")

    st.success("Loaded. Starting NSGA-III …")

    # df_raw contains:
    #   1) exact internal properties
    #   2) exact decoded compositions
    df_raw = run_nsga3(model, scalers, pop_size=NIND, n_gen=MAXGEN, seed=SEED)

    # Split exact properties and exact compositions
    df_exact_props = df_raw[["Bs (T)", "ln(Hc) (A/m)", "Hc (A/m)", "Dc (mm)"]].copy()
    df_exact_comp = df_raw[ELEMENTS].copy()

    # Clean displayed compositions: cutoff + renormalize
    df_clean_comp = clean_compositions(df_exact_comp, cutoff_at_percent=CUTOFF_AT_PERCENT)

    # Recalculate properties from cleaned compositions
    df_clean_props = predict_from_composition_df(model, scalers, df_clean_comp)

    # Build display dataframe (clean composition + recalculated properties)
    df_display = pd.DataFrame({
        "idx": range(1, len(df_clean_comp) + 1),
        "Composition": df_clean_comp.apply(
            lambda row: row_to_formula(row, decimals=int(DISPLAY_DECIMALS)), axis=1
        ),
        "Bs (T)": df_clean_props["Bs (T)"].round(4),
        "ln(Hc) (A/m)": df_clean_props["ln(Hc) (A/m)"].round(4),
        "Hc (A/m)": df_clean_props["Hc (A/m)"].round(2),
        "Dc (mm)": df_clean_props["Dc (mm)"].round(3),
    })

    st.subheader("Pareto-optimal candidate alloys")
    st.caption(
        f"Displayed compositions are post-processed by cutoff = {CUTOFF_AT_PERCENT:.2f} at.% "
        f"+ renormalization, and the shown Bs/Hc/Dc are recalculated from the cleaned compositions."
    )
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Export both exact and cleaned versions
    df_export = pd.DataFrame({
        "idx": range(1, len(df_clean_comp) + 1),
        "Composition_display": df_clean_comp.apply(
            lambda row: row_to_formula(row, decimals=int(DISPLAY_DECIMALS)), axis=1
        ),
        "Composition_exact": df_exact_comp.apply(
            lambda row: row_to_formula(row, decimals=6), axis=1
        ),
        "Bs_display (T)": df_clean_props["Bs (T)"].round(6),
        "ln(Hc)_display (A/m)": df_clean_props["ln(Hc) (A/m)"].round(6),
        "Hc_display (A/m)": df_clean_props["Hc (A/m)"].round(6),
        "Dc_display (mm)": df_clean_props["Dc (mm)"].round(6),
        "Bs_exact (T)": df_exact_props["Bs (T)"].round(6),
        "ln(Hc)_exact (A/m)": df_exact_props["ln(Hc) (A/m)"].round(6),
        "Hc_exact (A/m)": df_exact_props["Hc (A/m)"].round(6),
        "Dc_exact (mm)": df_exact_props["Dc (mm)"].round(6),
    })

    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "pareto_alloys_cleaned_and_exact.csv", mime="text/csv")

    # Plot using cleaned properties, since these correspond to displayed compositions
    Bs = df_clean_props["Bs (T)"].values
    lnHc = df_clean_props["ln(Hc) (A/m)"].values
    Dc = df_clean_props["Dc (mm)"].values

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    draw_ellipse(ax1, Bs, lnHc)
    ax1.scatter(Bs, lnHc, c="red", edgecolor="k", s=80, label="Pareto Front")
    ax1.axvline(1.5, ls="--", c="k")
    ax1.axhline(1.5, ls="--", c="k")
    ax1.set_xlabel("Bs (T)")
    ax1.set_ylabel("ln(Hc) (A/m)")
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    draw_ellipse(ax2, Bs, Dc)
    ax2.scatter(Bs, Dc, c="red", edgecolor="k", s=80)
    ax2.axvline(1.5, ls="--", c="k")
    ax2.axhline(1.0, ls="--", c="k")
    ax2.set_xlabel("Bs (T)")
    ax2.set_ylabel("Dc (mm)")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    draw_ellipse(ax3, Dc, lnHc)
    ax3.scatter(Dc, lnHc, c="red", edgecolor="k", s=80)
    ax3.axvline(1.0, ls="--", c="k")
    ax3.axhline(1.5, ls="--", c="k")
    ax3.set_xlabel("Dc (mm)")
    ax3.set_ylabel("ln(Hc) (A/m)")
    ax3.invert_yaxis()
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)

    st.toast("Done ✔️  You can adjust parameters and re-run!")
else:
    st.info("Click **Run Optimization** to start.")

