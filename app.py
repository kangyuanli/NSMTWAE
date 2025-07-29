# app.py ---------------------------------------------------------------
import streamlit as st
from optimizer import run_nsga3
from data_load import load_scalers, load_model

st.set_page_config(page_title="NSMTGM – Alloy Optimizer", layout="wide")

st.title("NSMTGM – Multi‑objective Design of Fe‑based Metallic Glasses")

# ------------------------- 侧边栏参数 ---------------------------------- #
st.sidebar.header("Evolutionary Parameters")
NIND = st.sidebar.number_input("Population size", 10, 1000, 200, 10)
MAXGEN = st.sidebar.number_input("Generations", 10, 1000, 100, 10)
SEED = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)

run_btn = st.sidebar.button("Run Optimization 🚀")

# ------------------------- 主逻辑 ------------------------------------- #
if run_btn:
    with st.spinner("Loading model & scalers …"):
        device = "cpu"  # GPU 可改为 "cuda"（若部署机器支持）
        model = load_model("checkpoints/MTWAE_latent8.pth", device=device)
        scalers = load_scalers("scalers")

    st.success("Loaded. Starting NSGA‑III …")
    df_result = run_nsga3(
        model=model,
        scalers=scalers,
        pop_size=NIND,
        n_gen=MAXGEN,
        seed=SEED,
    )

    st.subheader("Pareto‑optimal candidate alloys")
    st.dataframe(df_result.style.format(precision=4), use_container_width=True)

    # 提供下载
    csv = df_result.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download CSV",
        csv,
        file_name="pareto_alloys.csv",
        mime="text/csv",
    )

    st.toast("Done ✔️  You can adjust parameters and re‑run!")
else:
    st.info("在左侧栏设置参数，点击 **Run Optimization** 开始计算。")
