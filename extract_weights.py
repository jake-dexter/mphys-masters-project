from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


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

folder = "partial weights/"

base_dirs = {
    "Without": Path("data/Without EDM"),
    "With": Path("data/With EDM"),
}

data_all = load_array_data(base_dirs)

run_id = next(iter(data_all["With"]))
X = data_all["With"][run_id]["X"]   # (500,500)
Y = data_all["With"][run_id]["Y"]   # (500,500) in mrad

lam_axis = X[0, :]                  # shape (500,) lambda value for each column
theta_axis = Y[:, 0]                # shape (500,) theta value for each row (mrad)

lam_targets = []
w_slices = []   # each element shape (500,)

for filename in os.listdir(folder):
    if not filename.endswith(".npz"):
        continue

    npz = np.load(os.path.join(folder, filename), allow_pickle=True)
    lam_k = float(npz["lam"])
    W_k = npz["weight"]             # should be (500,500) or something you can map to it

    # find which column in the full grid corresponds to this lambda
    j = int(np.argmin(np.abs(lam_axis - lam_k)))

    # take the theta-profile at that lambda
    w_k = W_k[:, j]

    lam_targets.append(lam_k)
    w_slices.append(w_k)

lam_targets = np.array(lam_targets)
w_slices = np.stack(w_slices, axis=0)   # shape (n_lam=13, 500)

order = np.argsort(lam_targets)
lam_targets = lam_targets[order]        # (13,)
w_slices = w_slices[order, :]           # (13, 500)

W_full = np.empty((len(theta_axis), len(lam_axis)), dtype=float)  # (500,500)

for i in range(len(theta_axis)):
    # w_slices[:, i] are the values at fixed theta_i across the 13 lambdas
    W_full[i, :] = np.interp(lam_axis, lam_targets, w_slices[:, i])

np.savez_compressed("full_weight_map_interpolated.npz",
                    weight=W_full, lam_axis=lam_axis, theta_axis=theta_axis)

