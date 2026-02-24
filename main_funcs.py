import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle
import os

def load_array_data(base_dirs: dict[str, Path]):
    """
    Load all EDM data as arrays.

    Returns:
        data: dict
            data[edm_state][run_id] = {
                "X": X,  # lambda_frac grid
                "Y": Y,  # theta grid (mrad)
                "Z": Z,  # probability density
            }
    """
    data = {}
    for edm_state, dir_path in base_dirs.items():
        state_data = {}
        if edm_state == "With" or edm_state == "Without":
            filename_type = "G2_*.pkl"
        else:
            filename_type = "G2NEW_*.pkl"
        for file in sorted(dir_path.glob(filename_type)):
            run_id = int(file.stem.split("_")[1])
            with open(file, "rb") as f:
                X, Y, Z = pickle.load(f)
            state_data[run_id] = {
                "X": X,
                "Y": Y * 1000,
                "Z": Z,
            }
        data[edm_state] = state_data
    return data

def calculate_metrics_array(Z, Y, N, w=None):
    probs = N * Z * 1e-3

    if w is None:
        n = np.sum(probs)
        if n == 0:
            return np.nan, np.nan, np.nan
        mean = np.sum(probs * Y) / n
        var = np.sum(probs * (Y - mean) ** 2) / (n-1)
        sig = np.sqrt(max(var, 0.0))
        sem = sig / np.sqrt(n)
        return mean, sig, sem

    # Weighted version
    w = np.abs(w)
    n = np.sum(probs * w)
    if n == 0:
        return np.nan, np.nan, np.nan

    neff = (np.sum(probs * w) ** 2) / np.sum(probs * w ** 2)
    neff = max(neff, 1.0)

    mean = np.sum(w * probs * Y) / n
    var = np.sum(probs * w * (Y - mean) ** 2) / (n-1)
    sig = np.sqrt(max(var, 0.0))
    sem = sig / np.sqrt(neff)
    return mean, sig, sem

# --- FOM calculator ---
def compute_fom_for_lambda(data_all, n_val, lambda_curr, weight_func=None, plotting=False, normalize_weights=True):
    X = data_all["With"][0]["X"][0, :]
    i_lambda = np.argmin(np.abs(X - lambda_curr))

    run_means = []
    run_sems = []

    for run_id, grids in data_all["With"].items():
        Z = grids["Z"][:, i_lambda]
        Y_vals = grids["Y"][:, 0]

        # --- Apply weights properly ---
        if weight_func is not None:
            full_w = weight_func(grids["Y"], grids["X"])

            # Ensure shape alignment
            if full_w.shape != grids["Z"].shape:
                print(f"Shape mismatch for run {run_id}: w={full_w.shape}, Z={grids['Z'].shape}")
                full_w = np.resize(full_w, grids["Z"].shape)

            w = np.abs(full_w[:, i_lambda])

            if normalize_weights:
                w /= np.nanmax(full_w)

        else:
            w = None
        mean, sig, sem = calculate_metrics_array(Z, Y_vals, n_val, w=w)
        run_means.append(mean)
        run_sems.append(sem)

    means_with = np.array(run_means)
    sems_with = np.array(run_sems)
    valid = np.isfinite(means_with) & np.isfinite(sems_with) & (sems_with > 0)

    FOM = np.sum(means_with[valid] ** 2 / (2 * sems_with[valid] ** 2)) if np.any(valid) else np.nan
    return {"lambda_frac": lambda_curr, "FOM": FOM}


def compute_fom_over_lambda(
    data_all, n_val, lambda_min=0.2, lambda_max=0.95, lam_steps=None,
    weight_func=None, plotting=False, normalize_weights=True
):
    X = data_all["With"][0]["X"][0, :]
    if lam_steps is None:
        min_indx = np.argmin(np.abs(X - lambda_min))
        max_indx = np.argmin(np.abs(X - lambda_max))
        sliced_lam = X[min_indx:max_indx]
    else:
        sliced_lam = np.linspace(lambda_min, lambda_max, lam_steps)

    FOM = [
        compute_fom_for_lambda(
            data_all, n_val, lam_curr,
            weight_func=weight_func, plotting=plotting,
            normalize_weights=normalize_weights
        )["FOM"]
        for lam_curr in sliced_lam
    ]

    FOM = np.array(FOM, dtype=float)
    FOM = FOM[np.isfinite(FOM)]
    return FOM
