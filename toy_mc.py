import numpy as np
import matplotlib.pyplot as plt

outFolder = 'Plots/'

# model pieces
def modA(x):
    return np.cos(np.pi * x) + 1

def modB(x):
    return np.sin(np.pi * x)

def model(x, delta, s):
    return s * (modA(x) * (1.0 + delta * modB(x)))

xmin = -1.0
xmax = +1.0
K = 100

delta_true = 0.1
s_arr = np.array([125,250,500,1000,2000])
saved_err = []

# x grid
xdata = np.linspace(xmin, xmax, K)

# perfect model and uncertainties
for s in s_arr:
    yperf = model(xdata, delta_true, s)

    # avoid zeros because modA(±1)=0 -> yperf can be 0 at endpoints
    eps = 1e-12
    yperf_safe = np.maximum(yperf, eps)

    yperr = np.sqrt(yperf_safe)

    # generate fake data
    np.random.seed()
    data = np.random.normal(yperf_safe, yperr)

    # trial delta values
    trialdelta = np.linspace(-1, 1, 10000)

    # vectorised mu(d, x): (D, K)
    mu = model(xdata[None, :], trialdelta[:, None], s)

    # log-likelihood (Gaussian NLL up to constant): (D,)
    ll = 0.5 * np.sum(
        (data[None, :] - mu)**2 / yperf_safe[None, :]
        + np.log(2 * np.pi * yperf_safe[None, :]),
        axis=1
    )

    # ---------- FIT RESULT ----------
    idx = np.argmin(ll)
    mindelta = trialdelta[idx]
    minll = ll[idx]

    dll = ll - minll

    left = trialdelta[(dll >= 1) & (trialdelta < mindelta)].max()
    right = trialdelta[(dll >= 1) & (trialdelta > mindelta)].min()

    errdelta = 0.5 * ((mindelta - left) + (right - mindelta))
    saved_err.append(errdelta)

for i in range(len(saved_err)-1):
    print(f"err s={s_arr[i]}/err s={s_arr[i+1]} : {(saved_err[i]/saved_err[i+1])**2:.3g}")