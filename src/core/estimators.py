"""Time-series estimators with HAC inference, written from first principles.

Everything here takes plain numpy arrays so the same code runs on the data and
inside the Monte Carlo. Unit tests in tests/ check each estimator against known
parameters and, where one exists, a reference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats


def nw_lags(n: int) -> int:
    """Newey-West (1994) plug-in bandwidth, floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def hac_meat(g: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-kernel long-run sum of the score matrix ``g`` (n x k)."""
    omega = g.T @ g
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = g[lag:].T @ g[:-lag]
        omega += w * (gamma + gamma.T)
    return omega


@dataclass
class Fit:
    """Estimates, covariance, and diagnostics from one estimator."""

    names: list[str]
    params: np.ndarray
    cov: np.ndarray
    nobs: int
    resid: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def get(self, name: str) -> tuple[float, float]:
        i = self.names.index(name)
        return float(self.params[i]), float(self.se[i])

    def ci(self, name: str, level: float = 0.95) -> tuple[float, float]:
        b, s = self.get(name)
        z = stats.norm.ppf(0.5 + level / 2)
        return b - z * s, b + z * s


def ols(y: np.ndarray, X: np.ndarray, names: list[str], lags: int | None = None,
        small_sample: bool = True) -> Fit:
    """OLS with Newey-West HAC covariance (lags=0 gives White)."""
    n, k = X.shape
    lags = nw_lags(n) if lags is None else lags
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    u = y - X @ b
    V = xtx_inv @ hac_meat(X * u[:, None], lags) @ xtx_inv
    if small_sample:
        V *= n / (n - k)
    r2 = 1 - (u @ u) / np.sum((y - y.mean()) ** 2)
    return Fit(names, b, V, n, u, {"r2": float(r2), "lags": lags, "estimator": "OLS"})


def long_run(fit: Fit, coef: str, lag_coef: str) -> tuple[float, float]:
    """Long-run response b / (1 - rho) and its delta-method standard error."""
    i, j = fit.names.index(coef), fit.names.index(lag_coef)
    b, rho = fit.params[i], fit.params[j]
    grad = np.zeros(len(fit.params))
    grad[i] = 1.0 / (1.0 - rho)
    grad[j] = b / (1.0 - rho) ** 2
    return float(b / (1.0 - rho)), float(np.sqrt(grad @ fit.cov @ grad))


def gmm_linear(y: np.ndarray, X: np.ndarray, Z: np.ndarray, names: list[str],
               lags: int | None = None) -> Fit:
    """Two-step efficient linear GMM with a HAC weighting matrix and Hansen's J."""
    n, k = X.shape
    m = Z.shape[1]
    lags = nw_lags(n) if lags is None else lags
    zx, zy = Z.T @ X, Z.T @ y
    W = np.linalg.inv(Z.T @ Z)
    b1 = np.linalg.solve(zx.T @ W @ zx, zx.T @ W @ zy)
    u1 = y - X @ b1
    g1 = Z * u1[:, None]
    S = hac_meat(g1 - g1.mean(0), lags) / n
    W2 = np.linalg.inv(S)
    b = np.linalg.solve(zx.T @ W2 @ zx, zx.T @ W2 @ zy)
    u = y - X @ b
    gbar = Z.T @ u / n
    G = zx / n
    V = np.linalg.inv(G.T @ W2 @ G) / n
    J = float(n * gbar @ W2 @ gbar)
    dof = m - k
    return Fit(names, b, V, n, u, {"J": J, "J_df": dof, "J_p": float(stats.chi2.sf(J, dof)) if dof > 0 else np.nan,
                                    "lags": lags, "estimator": "two-step GMM"})


def _partial_out(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    return A - W @ np.linalg.lstsq(W, A, rcond=None)[0]


def iv_single(y: np.ndarray, x: np.ndarray, W: np.ndarray, Z: np.ndarray, names: list[str],
              lags: int | None = None) -> Fit:
    """2SLS for one endogenous regressor ``x`` with exogenous controls ``W`` and instruments ``Z``.

    Reports the HAC covariance, the Montiel Olea-Pflueger (2013) effective F, and the
    HAC-robust first-stage Wald F. ``names`` labels [x, *W columns].
    """
    n = len(y)
    lags = nw_lags(n) if lags is None else lags
    X = np.column_stack([x, W])
    Zfull = np.column_stack([Z, W])
    P = Zfull @ np.linalg.solve(Zfull.T @ Zfull, Zfull.T)
    Xh = P @ X
    A = np.linalg.inv(Xh.T @ X)
    b = A @ Xh.T @ y
    u = y - X @ b
    V = A @ hac_meat(Xh * u[:, None], lags) @ A.T * n / (n - X.shape[1])

    xt, Zt = _partial_out(x[:, None], W)[:, 0], _partial_out(Z, W)
    ztz = Zt.T @ Zt
    pi = np.linalg.solve(ztz, Zt.T @ xt)
    v = xt - Zt @ pi
    Vpi = np.linalg.solve(ztz, np.linalg.solve(ztz, hac_meat(Zt * v[:, None], lags)).T)
    f_eff = float(pi @ ztz @ pi / np.trace(ztz @ Vpi))
    f_wald = float(pi @ np.linalg.solve(Vpi, pi) / Z.shape[1])
    return Fit(names, b, V, n, u, {"F_eff": f_eff, "F_robust": f_wald, "first_stage": pi.tolist(),
                                   "lags": lags, "estimator": "2SLS"})


def anderson_rubin_set(y: np.ndarray, x: np.ndarray, W: np.ndarray, Z: np.ndarray, grid: np.ndarray,
                       lags: int | None = None, level: float = 0.95) -> dict:
    """HAC Anderson-Rubin confidence set for the coefficient on ``x`` by test inversion."""
    n, kz = Z.shape
    lags = nw_lags(n) if lags is None else lags
    R = np.column_stack([Z, W])
    accept = np.zeros(len(grid), dtype=bool)
    pvals = np.zeros(len(grid))
    for i, b0 in enumerate(grid):
        f = ols(y - b0 * x, R, [f"r{j}" for j in range(R.shape[1])], lags=lags)
        bz, Vz = f.params[:kz], f.cov[:kz, :kz]
        stat = float(bz @ np.linalg.solve(Vz, bz))
        pvals[i] = stats.chi2.sf(stat, kz)
        accept[i] = pvals[i] > 1 - level
    if not accept.any():
        return {"lo": np.nan, "hi": np.nan, "bounded": True, "empty": True, "connected": True}
    idx = np.flatnonzero(accept)
    connected = bool(np.all(np.diff(idx) == 1))
    return {"lo": float(grid[idx[0]]), "hi": float(grid[idx[-1]]),
            "bounded": bool(idx[0] > 0 and idx[-1] < len(grid) - 1), "empty": False,
            "connected": connected, "grid": grid, "pvals": pvals}


def num_hessian(f, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of a scalar function."""
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            e_i, e_j = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h * h)
            H[j, i] = H[i, j]
    return H


def tobit_left(y: np.ndarray, X: np.ndarray, names: list[str], censor: float) -> Fit:
    """Left-censored Tobit by maximum likelihood; observations with y <= censor are censored."""
    cens = y <= censor

    def nll(theta: np.ndarray) -> float:
        b, s = theta[:-1], np.exp(theta[-1])
        xb = X @ b
        ll_u = stats.norm.logpdf((y[~cens] - xb[~cens]) / s) - np.log(s)
        ll_c = stats.norm.logcdf((censor - xb[cens]) / s)
        return -(ll_u.sum() + ll_c.sum())

    b0 = np.linalg.lstsq(X[~cens], y[~cens], rcond=None)[0]
    s0 = np.log(np.std(y[~cens] - X[~cens] @ b0))
    res = optimize.minimize(nll, np.r_[b0, s0], method="BFGS")
    H = num_hessian(nll, res.x)
    V = np.linalg.inv(H)
    return Fit(names + ["log_sigma"], res.x, V, len(y), None,
               {"n_censored": int(cens.sum()), "converged": bool(res.success), "estimator": "Tobit MLE"})


def _sup_wald_null(k: int, trim: float, reps: int, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Simulated asymptotic null distribution of Andrews' (1993) sup-Wald statistic."""
    lam = np.arange(1, steps + 1) / steps
    keep = (lam >= trim) & (lam <= 1 - trim)
    out = np.empty(reps)
    for r in range(reps):
        B = np.cumsum(rng.standard_normal((steps, k)), axis=0) / np.sqrt(steps)
        bridge = B - lam[:, None] * B[-1]
        out[r] = np.max(np.sum(bridge[keep] ** 2, axis=1) / (lam[keep] * (1 - lam[keep])))
    return out


def qlr_break(y: np.ndarray, X: np.ndarray, trim: float = 0.15, lags: int | None = None,
              rng: np.random.Generator | None = None, reps: int = 4000) -> dict:
    """Quandt-Andrews sup-Wald test for a break in all coefficients at an unknown date."""
    n, k = X.shape
    rng = rng or np.random.default_rng(0)
    lo, hi = int(np.floor(trim * n)), int(np.ceil((1 - trim) * n))
    walds = np.full(n, np.nan)
    for tau in range(lo, hi):
        d = (np.arange(n) >= tau).astype(float)
        f = ols(y, np.column_stack([X, X * d[:, None]]), [f"b{j}" for j in range(2 * k)], lags=lags)
        bi, Vi = f.params[k:], f.cov[k:, k:]
        walds[tau] = bi @ np.linalg.solve(Vi, bi)
    tau_star = int(np.nanargmax(walds))
    null = _sup_wald_null(k, trim, reps, 1000, rng)
    stat = float(walds[tau_star])
    return {"sup_wald": stat, "break_index": tau_star, "p_value": float(np.mean(null >= stat)),
            "crit_5pct": float(np.quantile(null, 0.95)), "walds": walds}


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
