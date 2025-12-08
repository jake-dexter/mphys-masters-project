from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import nevergrad as ng
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import pickle
import optuna
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
        for file in sorted(dir_path.glob("G2_*.pkl")):
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
        var = np.sum(probs * (Y - mean) ** 2) / n
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
    var = np.sum(probs * w * (Y - mean) ** 2) / n
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

            # 🔧 Ensure shape alignment
            if full_w.shape != grids["Z"].shape:
                print(f"⚠️ Shape mismatch for run {run_id}: w={full_w.shape}, Z={grids['Z'].shape}")
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

def expand_weight_grid(W_coarse, full_shape=(500, 500)):
    """
    Upsample a coarse 2D weight grid (e.g. 50x50) to the full 500x500 simulation grid.
    Uses bilinear interpolation over normalised coordinates (0–1).
    
    Parameters
    ----------
    W_coarse : 2D np.ndarray
        The coarse weight matrix to expand.
    full_shape : tuple of int, optional
        The desired output shape (default is (500, 500)).
        
    Returns
    -------
    np.ndarray
        Interpolated full-resolution weight grid.
    """
    nx, ny = W_coarse.shape
    lambda_idx = np.linspace(0,1,nx)
    theta_idx = np.linspace(0,1,ny)

    # --- Step 2: create interpolator over coarse grid ---
    interp_func = RegularGridInterpolator(
        (theta_idx, lambda_idx),
        W_coarse,
        bounds_error=False,
        fill_value=None  # extrapolate smoothly at edges
    )

    # --- Step 3: build the full-resolution coordinate mesh (500x500) ---
    ny_full, nx_full = full_shape
    theta_full = np.linspace(0, 1, ny_full)
    lambda_full = np.linspace(0, 1, nx_full)

    # Meshgrid in "ij" order (so Y = rows, X = columns)
    theta_mesh, lambda_mesh = np.meshgrid(theta_full, lambda_full, indexing="ij")

    # Flatten mesh for interpolation input
    points = np.stack([theta_mesh.ravel(), lambda_mesh.ravel()], axis=-1)

    # --- Step 4: evaluate interpolator and reshape to 2D ---
    W_full = interp_func(points).reshape(full_shape)
    return W_full

def generate_mask(data_all,gamma=29.30343,lam_curr = 0.6):
    # --- Precompute theta_max(λ) and physical mask once ---
    gamma = 29.30343

    # Reference full coordinate grids
    Y_full_ref = data_all["With"][0]["Y"]      # shape (500,500)
    X_full_ref = data_all["With"][0]["X"]

    # Compute θ_max(λ)
    theta_max_full = np.arcsin(
        np.sqrt(X_full_ref * (1 - X_full_ref)) / (gamma * X_full_ref)
    )

    # Convert θ_max from radians → mrad to match Y grid
    theta_max_full_mrad = theta_max_full * 1e3

    # Build mask using mrad units
    physical_mask = np.abs(Y_full_ref) <= theta_max_full_mrad

    lam_idx = np.argmin(np.abs(X_full_ref - lam_curr))
    lambda_mask = np.zeros_like(X_full_ref)
    lambda_mask[:, lam_idx] = 1

    physical_mask = np.where(lambda_mask, physical_mask, 0.0)
    return physical_mask

def plot_weight(Y, X, W, name="Weight", save_path=None, lam_plot=-1):
    """
    Plot 1D (vs theta) and 2D contour of a weight array.

    Parameters
    ----------
    Y, X : 2D arrays
        Theta (Y) and lambda (X) coordinate grids.
    W : 2D array
        Weight array to visualise.
    name : str
        Label or title for the weight.
    save_path : str or Path
        Optional — if provided, saves instead of showing inline.
    lam_plot : float, optional
        Specific lambda value to plot a 1D theta slice for.
        If -1 (default), uses the central lambda index.
    """

    # --- Determine lambda index to slice along ---
    lam_values = X[0, :]  # 1D lambda array from grid
    theta_vals = Y[:, 0]  # 1D theta array

    if lam_plot < 0:
        mid_idx = W.shape[1] // 2
        lam_selected = lam_values[mid_idx]
    else:
        mid_idx = np.argmin(np.abs(lam_values - lam_plot))
        lam_selected = lam_values[mid_idx]

    weight_slice = W[:, mid_idx]

    # --- Create figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- 1D theta slice plot ---
    axes[0].plot(theta_vals, weight_slice, color="blue", lw=1.5)
    axes[0].set_xlabel(r"$\theta_L$ [mrad]")
    axes[0].set_ylabel("Weight")
    axes[0].set_title(f"λ = {lam_selected:.3f}")

    # --- 2D heatmap of full weighting ---
    c = axes[1].pcolormesh(X, Y, W, shading="auto", cmap="Blues")
    axes[1].set_xlabel(r"Fractional Lab Frame Energy ($\lambda$)")
    axes[1].set_ylabel(r"Longitudinal Angle ($\theta_L$) [mrad]")
    axes[1].set_title(name)
    fig.colorbar(c, ax=axes[1], label="Weighting (w)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def objective(weight_vector, grid_size=500, N_target=10000):
    """
    CMA-ES objective for symmetric θ and physical cutoff.
    - Optimises half of the θ range (θ ≥ 0)
    - Mirrors weights to enforce symmetry
    - Applies precomputed θ_max(λ) mask
    - Computes FOM and returns -FOM
    """

    # --- Step 1: interpret the half-grid ---
    # The vector length corresponds to half the θ dimension
    ny_half = grid_size // 2
    nx = grid_size
    W_half = weight_vector.reshape((ny_half, nx))
    W_half = np.clip(W_half, 0, None)

    # --- Step 2: mirror to create full θ symmetry ---
    # Stack flipped copy (θ<0) over θ>0
    W_coarse = np.vstack([np.flipud(W_half), W_half])

    # --- Step 3: upscale to full 500x500 grid ---
    W_full = expand_weight_grid(W_coarse)

    # --- Step 4: smooth weights for continuity ---
    W_full = gaussian_filter(W_full, sigma=2)

    # --- Step 5: apply precomputed θ_max(λ) mask (using mrad units) ---
    W_full = np.where(physical_mask, W_full, 0)

    # --- Step 6: compute FOM ---
    def weight_func(Y, X, run_id=None):
        return W_full

    metrics = compute_fom_over_lambda(
        data_all,
        N_target,
        weight_func=weight_func,
        lambda_min=0.2,
        lambda_max=0.95
    )

    FOM = np.sum(metrics)
    return -FOM

base_dirs = {
    "Without": Path("data/Without EDM"),
    "With": Path("data/With EDM"),
}

edm_state="With"
run_id=7

data_all = load_array_data(base_dirs)
grids = data_all[edm_state][run_id]
Y, X, Z = grids["Y"], grids["X"], grids["Z"]

physical_mask = generate_mask(data_all)

# CHANGE THESE PARAMETERS TO CHANGE THE PROGRAM
num_workers = 2
budget = 500
lam_curr = 0.2
checkpoint_file = f"Outputs/Checkpoints/cma_checkpoint_lam={lam_curr}_budget={budget}.pkl"

# --- CMA-ES configuration ---
grid_size = 500
ny_half = grid_size // 2
num_params = ny_half * grid_size

fom_history = []   # store FOM values
iter_history = []  # store iteration numbers

# --- Create or resume optimiser ---
RESUME_EXISTING = True
if os.path.exists(checkpoint_file) and RESUME_EXISTING:
    print(f"Resuming optimiser from checkpoint: {checkpoint_file}")
    with open(checkpoint_file, "rb") as f:
        optimizer = pickle.load(f)
else:
    parametrization = ng.p.Array(shape=(num_params,)).set_bounds(0, 10)
    optimizer = ng.optimizers.CMA(
        parametrization=parametrization,
        budget=budget,
        num_workers=num_workers,
    )

# --- Run optimisation loop with progress tracking ---
for i in range(budget):
    x = optimizer.ask()
    value = objective(x.value)
    optimizer.tell(x, value)

    fom_history.append(-value)
    iter_history.append(i + 1)

    # --- Save lightweight checkpoint (overwrite each time) ---
    if (i + 1) % 20 == 0 or i == budget - 1:
        best_vec = optimizer.provide_recommendation().value
        best_fom = -objective(best_vec)

        checkpoint_data = {
            "iteration": i + 1,
            "best_vector": best_vec,
            "best_FOM": best_fom,
            "fom_history": fom_history,
            "iter_history": iter_history,
        }

        np.savez_compressed(
            "Outputs/Checkpoints/light_checkpoint_latest.npz",  # single rolling file
            **checkpoint_data
        )

        print(f"Checkpoint updated at iteration {i+1}/{budget} | FOM = {best_fom:.3f}")


# --- Final recommendation ---
recommendation = optimizer.provide_recommendation()
best_vector = recommendation.value
best_FOM = -objective(best_vector)
print(f"\n✅ Optimisation finished.\nBest FOM = {best_FOM:.4f}")

# reshape the optimised half-θ grid
best_half = best_vector.reshape((ny_half, grid_size))
best_half = np.clip(best_half, 0, None)

# mirror it vertically to enforce θ symmetry
best_coarse = np.vstack([np.flipud(best_half), best_half])

best_full = expand_weight_grid(best_coarse)
best_full = gaussian_filter(best_full, sigma=2)
best_full = np.where(physical_mask, best_full, 0)

plot_weight(Y, X, best_full, name="Weight", save_path=None, lam_plot = lam_curr)