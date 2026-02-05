from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from pathlib import Path
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


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

folder = "outputs/Checkpoints"
save_folder = "outputs/Weight_Maps/"

base_dirs = {
    "Without": Path("data/Without_EDM"),
    "With": Path("data/With_EDM"),
}

data_all = load_array_data(base_dirs)

for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if os.path.isfile(filepath):
        print(f"Processing file: {filename}")
        budget = int(filename.split("budget=")[1].split(".")[0])
        lam = float(filename.split("lam=")[1].split("_")[0])
        with open(filepath, "rb") as f:
            optimizer = pickle.load(f)

        physical_mask = generate_mask(data_all,lam_curr=lam)
        
        recommendation = optimizer.provide_recommendation()
        best_vector = recommendation.value

        best_half = best_vector.reshape((250, 500))
        best_half = np.clip(best_half, 0, None)

        best_coarse = np.vstack([np.flipud(best_half), best_half])

        best_full = expand_weight_grid(best_coarse)
        best_full = gaussian_filter(best_full, sigma=2)
        best_full = np.where(physical_mask, best_full, 0)

        outname = save_folder + f"optimal_weight_lam={lam}_budget={budget}.npz"
        np.savez(outname, weight=best_full, lam=lam, budget=budget)