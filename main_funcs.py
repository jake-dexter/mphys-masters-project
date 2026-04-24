from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from tqdm import tqdm
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
def compute_fom_for_lambda(
    data_all, n_val, lambda_curr,
    weight_func=None, plotting=False, normalize_weights=True,
    acceptance_func=lambda x, y, z: z
):
    X = data_all["With"][0]["X"][0, :]
    i_lambda = np.argmin(np.abs(X - lambda_curr))

    run_means = []
    run_sems = []

    for run_id, grids in data_all["With"].items():
        Z = grids["Z"][:, i_lambda]
        Y_vals = grids["Y"][:, 0]

        # --- Apply detector acceptance to the event distribution ---
        Z = acceptance_func(Y_vals, lambda_curr, Z[:, None])[:, 0]

        # --- Apply weights properly ---
        if weight_func is not None:
            full_w = weight_func(grids["Y"], grids["X"])

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
    weight_func=None, plotting=False, normalize_weights=True,
    acceptance_func=lambda x, y, z: z
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
            normalize_weights=normalize_weights,
            acceptance_func=acceptance_func
        )["FOM"]
        for lam_curr in sliced_lam
    ]

    FOM = np.array(FOM, dtype=float)
    FOM = FOM[np.isfinite(FOM)]
    return FOM

def weight_per_lam(lam, data_all, w0=np.ones((500,)), N=1e5, acceptance_func = lambda x, y, z: z):
    theta = data_all["With New"][7]["Y"][:, 0]
    lambdas = data_all["With New"][7]["X"][0, :]
    i_lambda = np.argmin(np.abs(lambdas - lam))

    num_sum = np.zeros_like(theta, dtype=float)
    den_sum = np.zeros_like(theta, dtype=float)

    for run_id, grids in data_all["With New"].items():
        Zcol = grids["Z"][:, i_lambda]
        p = N * Zcol * 1e-3
        p = acceptance_func(theta, lam, p[:, None])[:, 0]

        A = np.sum(p * w0)
        B = np.sum(p * w0 * theta)
        C = np.sum(p * w0 * theta**2)
        D = np.sum(p * w0**2)

        y  = B / A
        s2 = C / A - (B / A)**2
        sigma2 = s2 * D / A**2

        Delta = theta - y
        a = Delta - (y * Delta**2)/(2 * s2) + 1.5 * y

        b = (y * A) / D

        weight = (y * p) / sigma2
        num_sum += weight * a
        den_sum += weight * b
        
    eps = 1e-30
    mask = den_sum > eps

    w_opt = np.zeros_like(theta, dtype=float)
    w_opt[mask] = num_sum[mask] / den_sum[mask]

    w_opt = np.maximum(0.0, w_opt)
    return w_opt

def plot_heatmap(X, Y, Z, w=None, title="", xlabel="", ylabel="",
                 cbar_label="", save_path=None):
    """
    Generic 2D heatmap plotter with optional weighting.
    Normalises using p = Z * dθ * dλ so that Σ(p_i * w_i_norm) = Σ(w_i).
    """
    # Compute bin widths
    dtheta = np.abs(Y[1, 0] - Y[0, 0])
    dlambda = np.abs(X[0, 1] - X[0, 0])
    p = Z * dtheta * dlambda  # actual probability per bin

    if w is not None:
        # Compute normalisation constant
        numerator = np.sum(p)
        denominator = np.sum(p * w)
        N = numerator / denominator if denominator > 0 else 1.0
    Z_plot = Z * (N*w if w is not None else 1.0)

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(X, Y, Z_plot, shading='auto', cmap='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=cbar_label)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def save_probability_maps(data_all, output_base=Path("outputs"), save=True, plotting=False):
    if not save:
        print("Skipping save — files already generated.")
        return

    for edm_state, runs in tqdm(data_all.items(), desc="EDM States"):
        folder_name = f"{edm_state} EDM"
        output_dir = output_base / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for run_id, grids in runs.items():
            X, Y, Z = grids["X"], grids["Y"], grids["Z"]
            fname = f"FRACTIONAL G2_{run_id:02}_{edm_state.lower()}EDM.png"
            save_path = output_dir / fname

            plot_heatmap(
                X, Y, Z,
                title=f"{folder_name} - $T_{run_id}$",
                xlabel=r"Fractional energy ($\lambda$)",
                ylabel=r"Longitudinal angle ($\theta_L$) [mrad]",
                cbar_label="Probability density",
                save_path=save_path
            )

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

def plot_weight(Y, X, W, name="Weight", save_path=None, lam_plot=-1):
    """
    Plot 1D (vs theta) and 2D heatmap of a weight array.

    If save_path is provided, saves TWO separate figures:
      - <save_path>_slice.png   : 1D theta slice at chosen lambda
      - <save_path>_map.png     : 2D weight map

    If save_path is None, displays a single combined (1x2) figure inline.

    Parameters
    ----------
    Y, X : 2D arrays
        Theta (Y) and lambda (X) coordinate grids.
    W : 2D array
        Weight array to visualise.
    name : str
        Label/title for the weight (used on the 2D plot).
    save_path : str or Path, optional
        Base path (without extension preferred). Suffixes are added automatically.
    lam_plot : float, optional
        Specific lambda value to plot a 1D theta slice for.
        If -1 (default), uses the central lambda index.
    """
    # --- Determine lambda index to slice along ---
    lam_values = X[0, :]   # 1D lambda array from grid
    theta_vals = Y[:, 0]   # 1D theta array

    if lam_plot < 0:
        mid_idx = W.shape[1] // 2
        lam_selected = lam_values[mid_idx]
    else:
        mid_idx = int(np.argmin(np.abs(lam_values - lam_plot)))
        lam_selected = lam_values[mid_idx]

    weight_slice = W[:, mid_idx]

    cmap = "RdYlBu_r"
    norm = mpl.colors.TwoSlopeNorm(vmin=W.min(),vcenter=1.5,vmax=W.max())

    # --- Inline display (single combined figure) ---
    if save_path is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        # 1D slice
        axes[0].plot(theta_vals, weight_slice, lw=1.5)
        axes[0].set_xlabel(r"$\theta_L$ [mrad]")
        axes[0].set_ylabel("Weight")
        axes[0].set_title(fr"$\lambda$ = {lam_selected:.3f}")

        # 2D map
        c = axes[1].pcolormesh(X, Y, W, shading="auto", cmap=cmap)#, norm=norm)
        axes[1].set_xlabel(r"Fractional Lab Frame Energy ($\lambda$)")
        axes[1].set_ylabel(r"Longitudinal Angle ($\theta_L$) [mrad]")
        axes[1].set_title(name)
        fig.colorbar(c, ax=axes[1], label="Weighting (w)")

        plt.tight_layout()
        plt.show()
        return

    # --- Saving (two separate figures) ---
    base = Path(save_path)

    # If user passed a filename with an extension, strip it so we can append suffixes cleanly
    if base.suffix:
        base = base.with_suffix("")

    # 1) Save 1D slice figure
    mm = 1 / 25.4

    fig1, ax1 = plt.subplots(
    figsize=(120 * mm, 100 * mm)
    )
    ax1.plot(theta_vals, weight_slice)#, rasterized=True)
    ax1.set_xlabel(r"Longitudinal Angle $\theta_L$ [mrad]")
    ax1.set_ylabel(fr"Weight $w(\theta_\mathrm{{L}},\lambda$={lam_selected:.3f})")
    # ax1.set_title(fr"Weight w($\theta$,$\lambda$={lam_selected:.3f})")
    # ax1.set_facecolor("#def4f4")
    # fig1.set_facecolor("#def4f4")
    ax1.set_xlim([-70,70])
    fig1.tight_layout()
    fig1.savefig(base.parent / f"{base.name}_slice.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # 2) Save 2D map figure
    mm = 1 / 25.4

    fig2, ax2 = plt.subplots(
        figsize=(120 * mm, 100 * mm)
    )
    c = ax2.pcolormesh(X, Y, W, shading="auto",cmap=cmap, norm=norm)#, rasterized=True)
    ax2.set_xlabel(r"Fractional Lab Frame Energy $\lambda$")
    ax2.set_ylabel(r"Longitudinal Angle $\theta_\mathrm{L}$ [mrad]")
    # ax2.set_title("Weighting phase space")
    # ax2.set_facecolor("#def4f4")
    # fig2.set_facecolor("#def4f4")
    ax2.set_ylim([-70,70])
    fig2.colorbar(c, ax=ax2, label=r"Weighting $w(\theta_\mathrm{L},\lambda)$")
    fig2.tight_layout()
    fig2.savefig(base.parent / f"{base.name}_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

def objective(data_all, physical_mask, weight_vector, acceptance_func=lambda x, y, z: z, grid_size=500, N_target=10000):
    """
    CMA-ES objective for symmetric θ and physical cutoff.
    - Optimises half of the θ range (θ ≥ 0)
    - Mirrors weights to enforce symmetry
    - Applies precomputed θ_max(λ) mask
    - Computes FOM and returns -FOM
    """
    # --- Step 1: interpret the half-grid ---
    ny_half = grid_size // 2
    nx = grid_size
    W_half = weight_vector.reshape((ny_half, nx))
    W_half = np.clip(W_half, 0, None)

    # --- Step 2: mirror to create full θ symmetry ---
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
        lambda_max=0.95,
        acceptance_func=acceptance_func
    )

    FOM = np.sum(metrics)
    return -FOM