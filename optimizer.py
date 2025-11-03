# optimizer.py -----------------------------------------------------------
import numpy as np
import torch
import pandas as pd
# --- Non‑dominated sorting -------------------------
try:
    
    from pymoo.util.non_dominated_sorting import NonDominatedSorting
except ModuleNotFoundError:
    
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions, get_termination
from pymoo.optimize import minimize
#from pymoo.util.non_dominated_sorting import NonDominatedSorting


class MTDesignProblem(Problem):

    def __init__(self, model, scalers, latent_size=8, sigma=8.0):
        super().__init__(
            n_var=latent_size,
            n_obj=3,
            n_constr=0,
            xl=np.full(latent_size, -sigma),   
            xu=np.full(latent_size,  sigma),   
            type_var=np.float32,
        )
        self._model = model
        self._scalers = scalers

    
    def _evaluate(self, X, out, *args, **kwargs):
        import streamlit as st, traceback, numpy as np, torch
    
        try:   
            device = next(self._model.parameters()).device
            z = torch.from_numpy(X).float().to(device)
    
            with torch.no_grad():
                Bs   = self._model.head_Bs(z).cpu().numpy()     # (n,1)
                lnHc = self._model.head_Hc(z).cpu().numpy()     # (n,1)
                Dc   = self._model.head_Dc(z).cpu().numpy()     # (n,1)
    
           
            Bs   = self._scalers["Bs"].inverse_transform(Bs)
            lnHc = self._scalers["Hc"].inverse_transform(lnHc)
            Dc   = self._scalers["Dc"].inverse_transform(Dc)
    
            
            out["F"] = np.hstack([-Bs, lnHc, -Dc]).astype(np.float64)
    
        except Exception as e:   
            st.error(f"🚨 _evaluate: {e}")
            st.code(traceback.format_exc())
            raise   

def _nearest_valid_population(n_desired: int) -> int:
    
    import math
    valid = []
    for H in range(1, 100):
        n = math.comb(H + 3 - 1, 3 - 1)
        valid.append(n)
        if n >= 2500: break
    
    for v in valid:
        if v >= n_desired:
            return v
    return valid[-1]



def run_nsga3(
    model,
    scalers,
    pop_size: int = 200,
    n_gen: int = 100,
    seed: int = 42,
    sigma: float = 8.0,
    periodic_table: list[str] | None = None,
):
    
    problem = MTDesignProblem(model, scalers, latent_size=8, sigma=sigma)

    pop_size_valid = _nearest_valid_population(pop_size)
    ref_dirs = get_reference_directions("das-dennis", 3, n_points=pop_size_valid)
    algorithm = NSGA3(pop_size=ref_dirs.shape[0], ref_dirs=ref_dirs)

    #ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_points=pop_size)
    #algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
        save_history=False,
    )

    
    nd_idx = NonDominatedSorting().do(res.F, only_non_dominated_front=True)
    Z_nd = res.X[nd_idx]

   
    with torch.no_grad():
        comps = (
            model.decoder(torch.from_numpy(Z_nd).float())
            .cpu()
            .numpy()
        )

   
    with torch.no_grad():
        z_nd_tensor = torch.from_numpy(Z_nd).float()
        Bs = scalers["Bs"].inverse_transform(model.head_Bs(z_nd_tensor).numpy()).ravel()
        lnHc = scalers["Hc"].inverse_transform(model.head_Hc(z_nd_tensor).numpy()).ravel()
        Dc = scalers["Dc"].inverse_transform(model.head_Dc(z_nd_tensor).numpy()).ravel()

    
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


