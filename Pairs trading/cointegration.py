import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def compute_mean_reverting_spread(
    df, cols=("mid_geo", "mid_cxw"), dt=1.0, verbose=True
):
    S = df[list(cols)].dropna()

    dS = S.diff().dropna()
    var_res = VAR(dS).fit(1)

    A = var_res.params.iloc[0].values
    B = var_res.coefs[0]

    I = np.eye(B.shape[0])
    kappa = (I - B) / dt

    theta = np.linalg.solve(kappa, A * dt)

    eigvals, eigvecs = np.linalg.eig(kappa)

    idx = np.argmax(np.real(eigvals))
    v = np.real(eigvecs[:, idx])

    alpha = v / (np.abs(v).max() + 1e-12)

    spread = S.values @ alpha
    spread = pd.Series(spread, index=S.index, name="spread_best")

    if verbose:
        print("VAR(1) on ΔS_t results")
        print("A (intercept):", A)
        print("B (lag-1 matrix):\n", B)
        print("\nDerived kappa = (I - B)/dt:\n", kappa)
        print("Derived theta = kappa^{-1} A dt:", theta)

        print("\nEigenvalues of kappa:", eigvals)
        print("Chosen eigenvalue (fastest MR):", eigvals[idx])

        print("\nBest linear combination (up to scale/sign):")
        print(f"  spread = {alpha[0]: .6f} * {cols[0]}  + {alpha[1]: .6f} * {cols[1]}")

        if np.abs(alpha[1]) > 1e-12:
            c = alpha[0] / alpha[1]
            print("\nSame combo normalized to second asset coefficient = 1:")
            print(f"  spread = {c: .6f} * {cols[0]}  + 1.000000 * {cols[1]}")

    return alpha, spread, kappa, theta, eigvals
