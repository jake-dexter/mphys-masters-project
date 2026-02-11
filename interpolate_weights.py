import numpy as np
import pickle
import os
from pathlib import Path

# ------------------------------------------------------------
# 1) LOAD SIMULATION DATA (defines theta_axis and lam_axis)
# ------------------------------------------------------------

def load_array_data(base_dirs):
    data = {}
    for edm_state, dir_path in base_dirs.items():
        state_data = {}
        for file in sorted(Path(dir_path).glob("G2_*.pkl")):
            run_id = int(file.stem.split("_")[1])
            with open(file, "rb") as f:
                X, Y, Z = pickle.load(f)
            state_data[run_id] = {
                "X": X,
                "Y": Y * 1000.0,   # rad → mrad
                "Z": Z,
            }
        data[edm_state] = state_data
    return data


base_dirs = {
    "Without": Path("data/Without EDM"),
    "With": Path("data/With EDM"),
}

data_all = load_array_data(base_dirs)

# use any run (all runs share the same grids)
run_id = next(iter(data_all["With"]))
X = data_all["With"][run_id]["X"]   # (500,500)
Y = data_all["With"][run_id]["Y"]   # (500,500) mrad

lam_axis = X[0, :]      # (500,)
theta_axis = Y[:, 0]    # (500,)

# ------------------------------------------------------------
# 2) LOAD ALL PARTIAL WEIGHT FILES
# ------------------------------------------------------------

folder = "partial weights2/"

lam_targets = []
w_slices = []

for filename in os.listdir(folder):
    if not filename.endswith(".npz"):
        continue

    npz = np.load(os.path.join(folder, filename), allow_pickle=True)

    # lambda value
    if "lam" in npz.files:
        lam_k = float(npz["lam"])
    else:
        raise RuntimeError(f"No lam stored in {filename}")

    W_k = npz["weight"]  # expected (500,500)

    # extract the active lambda column
    col_strength = np.sum(np.abs(W_k), axis=0)
    j = int(np.argmax(col_strength))
    w_k = W_k[:, j]      # (500,)

    lam_targets.append(lam_k)
    w_slices.append(w_k)

lam_targets = np.array(lam_targets)
w_slices = np.stack(w_slices, axis=0)   # (n_lam, 500)

# ------------------------------------------------------------
# 3) SORT BY LAMBDA
# ------------------------------------------------------------

order = np.argsort(lam_targets)
lam_targets = lam_targets[order]
w_slices = w_slices[order]

# ------------------------------------------------------------
# 4) INTERPOLATE ACROSS LAMBDA FOR EACH THETA
# ------------------------------------------------------------

W_full = np.empty((len(theta_axis), len(lam_axis)), dtype=float)

for i in range(len(theta_axis)):
    W_full[i, :] = np.interp(
        lam_axis,
        lam_targets,
        w_slices[:, i],
        left=0.0,
        right=0.0,
    )

# ------------------------------------------------------------
# 5) SAVE FULL WEIGHT MAP
# ------------------------------------------------------------

np.savez_compressed(
    "random.npz",
    weight=W_full,
    lam_axis=lam_axis,
    theta_axis=theta_axis,
)