# optimizer.py -----------------------------------------------------------
import numpy as np
import torch
import pandas as pd
# --- Non‑dominated sorting (版本兼容) -------------------------
try:
    # 0.6.1 独有
    from pymoo.util.non_dominated_sorting import NonDominatedSorting
except ModuleNotFoundError:
    # 0.6.0 及 ≥0.7.x
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions, get_termination
from pymoo.optimize import minimize
#from pymoo.util.non_dominated_sorting import NonDominatedSorting

# ------------------------ 自定义多目标问题 ------------------------------ #
class MTDesignProblem(Problem):
    """
    目标：
        1) 最大化 Bs      → 转为最小化  -Bs
        2) 最小化 ln(Hc)
        3) 最大化 Dc      → 转为最小化  -Dc
    """
    def __init__(self, model, scalers, latent_size=8, sigma=8.0):
        super().__init__(
            n_var=latent_size,
            n_obj=3,
            n_constr=0,
            xl=np.full(latent_size, -sigma),   # 下界
            xu=np.full(latent_size,  sigma),   # 上界
            type_var=np.float32,
        )
        self._model = model
        self._scalers = scalers

    # 核心评估函数 -------------------------------------------------------- #
    def _evaluate(self, X, out, *_):
        z = torch.from_numpy(X).float()
        with torch.no_grad():
            Bs = self._model.head_Bs(z).cpu().numpy()
            lnHc = self._model.head_Hc(z).cpu().numpy()
            Dc = self._model.head_Dc(z).cpu().numpy()

        # 逆标准化
        Bs = self._scalers["Bs"].inverse_transform(Bs)
        lnHc = self._scalers["Hc"].inverse_transform(lnHc)
        Dc = self._scalers["Dc"].inverse_transform(Dc)

        out["F"] = np.column_stack([-Bs, lnHc, -Dc])        # nsga‑③ => 全部最小化
        # 额外保存原始值，后处理时直接使用
        out["Bs_raw"] = Bs
        out["lnHc_raw"] = lnHc
        out["Dc_raw"] = Dc


# ------------------------- 对外优化接口 --------------------------------- #
def run_nsga3(
    model,
    scalers,
    pop_size: int = 200,
    n_gen: int = 100,
    seed: int = 42,
    sigma: float = 8.0,
    periodic_table: list[str] | None = None,
):
    """执行 NSGA‑III 优化并返回 *第一条* 非支配前沿及其解码结果。"""
    problem = MTDesignProblem(model, scalers, latent_size=8, sigma=sigma)

    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_points=pop_size)
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
        save_history=False,
    )

    # ---------------------- 非支配筛选 ---------------------------------- #
    nd_idx = NonDominatedSorting().do(res.F, only_non_dominated_front=True)
    Z_nd = res.X[nd_idx]

    # 解码到成分空间
    with torch.no_grad():
        comps = (
            model.decoder(torch.from_numpy(Z_nd).float())
            .cpu()
            .numpy()
        )

    # 重新预测以取到标度后的物性
    with torch.no_grad():
        z_nd_tensor = torch.from_numpy(Z_nd).float()
        Bs = scalers["Bs"].inverse_transform(model.head_Bs(z_nd_tensor).numpy()).ravel()
        lnHc = scalers["Hc"].inverse_transform(model.head_Hc(z_nd_tensor).numpy()).ravel()
        Dc = scalers["Dc"].inverse_transform(model.head_Dc(z_nd_tensor).numpy()).ravel()

    # 组装 DataFrame
    elements = periodic_table or [
        'Fe', 'B', 'Si', 'P', 'C', 'Co', 'Nb', 'Ni', 'Mo', 'Zr',
        'Ga', 'Al', 'Dy', 'Cu', 'Cr', 'Y', 'Nd', 'Hf', 'Ti', 'Tb',
        'Ho', 'Ta', 'Er', 'Sn', 'W', 'Tm', 'Gd', 'Sm', 'V', 'Pr'
    ]
    df_comp = pd.DataFrame(comps, columns=elements)
    df_props = pd.DataFrame({
        "Bs (T)": Bs,
        "ln(Hc) (A/m)": lnHc,
        "Hc (A/m)": np.exp(lnHc),
        "Dc (mm)": Dc,
    })
    return pd.concat([df_props, df_comp], axis=1)

