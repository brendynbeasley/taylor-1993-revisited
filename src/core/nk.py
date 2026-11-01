"""Three-equation New Keynesian model with a Taylor-type rule, solved by generalized Schur (QZ).

Variables are deviations from steady state in annualized percentage points:
    IS:    x_t  = E_t x_{t+1} - s (i_t - E_t pi_{t+1}) + d_t
    PC:    pi_t = beta E_t pi_{t+1} + kappa x_t + u_t
    Rule:  i_t  = rho_i i_{t-1} + (1 - rho_i)(phi_pi pibar_t + phi_y x_t) + v_t
with pibar_t the four-quarter average of pi and AR(1) shocks d, u, v.

State ordering. Predetermined k_t = [d_t, u_t, v_t, i_{t-1}, pi_{t-1}, pi_{t-2}, pi_{t-3}];
jump variables p_t = [x_t, pi_t]. The rule is substituted into the IS curve and into the law
of motion for i_{t-1}, which keeps A nonsingular. The system is A E_t z_{t+1} = B z_t, z = [k; p].
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import linalg

NK, NP = 7, 2
X, PI = 0, 1  # positions of jumps in p_t


@dataclass(frozen=True)
class Params:
    s: float = 0.25        # real-rate elasticity of the gap (annualized units)
    kappa: float = 0.10    # Phillips curve slope (annualized units)
    beta: float = 0.99
    phi_pi: float = 1.5
    phi_y: float = 0.5
    rho_i: float = 0.0
    rho_d: float = 0.8
    rho_u: float = 0.5
    rho_v: float = 0.5
    sd_d: float = 0.5
    sd_u: float = 0.5
    sd_v: float = 0.5
    four_quarter: bool = True  # rule responds to four-quarter average inflation

    def with_rule(self, phi_pi: float, phi_y: float, rho_i: float) -> "Params":
        return replace(self, phi_pi=phi_pi, phi_y=phi_y, rho_i=rho_i)


@dataclass
class Solution:
    F: np.ndarray       # [x_t, pi_t] = F k_t
    G: np.ndarray       # i_t = G k_t
    P: np.ndarray       # k_{t+1} = P k_t + Q eps_{t+1}
    Q: np.ndarray
    determinate: bool
    n_stable: int


def rule_row(p: Params) -> np.ndarray:
    """Coefficients c with i_t = c @ [k_t; p_t]."""
    c = np.zeros(NK + NP)
    w = 0.25 if p.four_quarter else 1.0
    c[2], c[3] = 1.0, p.rho_i
    c[NK + PI] = (1 - p.rho_i) * p.phi_pi * w
    if p.four_quarter:
        c[4:7] = (1 - p.rho_i) * p.phi_pi * w
    c[NK + X] = (1 - p.rho_i) * p.phi_y
    return c


def system(p: Params) -> tuple[np.ndarray, np.ndarray]:
    n = NK + NP
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    ix, ipi = NK + X, NK + PI
    c = rule_row(p)
    for r, rho in enumerate([p.rho_d, p.rho_u, p.rho_v]):
        A[r, r], B[r, r] = 1.0, rho
    A[3, 3], B[3] = 1.0, c                # next period's lagged rate is today's rate
    A[4, 4], B[4, ipi] = 1.0, 1.0         # pi_t becomes next period's pi_{t-1}
    A[5, 5], B[5, 4] = 1.0, 1.0
    A[6, 6], B[6, 5] = 1.0, 1.0
    A[7, ix], A[7, ipi] = 1.0, p.s        # IS with the rule substituted for i_t
    B[7] = p.s * c
    B[7, ix] += 1.0
    B[7, 0] -= 1.0
    A[8, ipi] = p.beta                    # Phillips curve
    B[8, ipi], B[8, ix], B[8, 1] = 1.0, -p.kappa, -1.0
    return A, B


def solve(p: Params) -> Solution:
    """Blanchard-Kahn (1980) solution of E_t z_{t+1} = M z_t with M = A^{-1} B.

    A is nonsingular by construction (det A = beta), so the system can be written in
    first-order form. Each unstable root lambda of M has a left eigenvector w with
    w' z_{t+1} = lambda w' z_t in expectation; boundedness requires w' z_t = 0 for all of
    them. With exactly NP unstable roots these NP restrictions pin down p_t = F k_t.
    (Klein's QZ method gives the same answer when A is singular; here the eigenvector
    route avoids QZ reordering failures at repeated zero roots from the lag states.)
    """
    A, B = system(p)
    M = np.linalg.solve(A, B)
    lam, Wl = np.linalg.eig(M.T)  # columns of Wl are left eigenvectors of M
    unstable = np.abs(lam) > 1.0 + 1e-10
    n_stable = int(np.sum(~unstable))
    Q = np.zeros((NK, 3))
    Q[0, 0], Q[1, 1], Q[2, 2] = p.sd_d, p.sd_u, p.sd_v
    if n_stable != NK:
        nan = np.full((NK, NK), np.nan)
        return Solution(np.full((NP, NK), np.nan), np.full(NK, np.nan), nan, Q, False, n_stable)
    Wu = Wl[:, unstable].T  # NP x (NK + NP)
    F = np.real(-np.linalg.solve(Wu[:, NK:], Wu[:, :NK]))
    P = M[:NK, :NK] + M[:NK, NK:] @ F
    c = rule_row(p)
    G = c[:NK] + c[NK:] @ F
    return Solution(F, G, P, Q, True, n_stable)


def observation_matrix(sol: Solution) -> np.ndarray:
    """Rows map k_t to [x_t, pi_t, i_t, pibar_t, i_{t-1}]."""
    H = np.zeros((5, NK))
    H[:2] = sol.F
    H[2] = sol.G
    H[3] = sol.F[PI] / 4.0
    H[3, 4:7] += 0.25
    H[4, 3] = 1.0
    return H


def state_cov(sol: Solution) -> np.ndarray:
    return linalg.solve_discrete_lyapunov(sol.P, sol.Q @ sol.Q.T)


def autocov(sol: Solution, lags: tuple[int, ...] = (0, 1, 2, 4)) -> dict[int, np.ndarray]:
    """Model autocovariances of y_t = [x, pi, i]: Gamma(j) = E[y_t y_{t-j}']."""
    S0 = state_cov(sol)
    H = observation_matrix(sol)[:3]
    return {j: H @ np.linalg.matrix_power(sol.P, j) @ S0 @ H.T for j in lags}


def ols_plim(sol: Solution, smoothing: bool) -> np.ndarray:
    """Analytic probability limit of OLS of i_t on [pibar_t, x_t] (and i_{t-1} if smoothing).

    Returns [phi_pi_hat, phi_y_hat] as long-run responses.
    """
    H = observation_matrix(sol)
    C = H @ state_cov(sol) @ H.T  # covariance of [x, pi, i, pibar, ilag]
    idx = [4, 3, 0] if smoothing else [3, 0]
    b = np.linalg.solve(C[np.ix_(idx, idx)], C[idx, 2])
    if smoothing:
        rho, bp, by = b
        return np.array([bp / (1 - rho), by / (1 - rho)])
    return b


def simulate(sol: Solution, T: int, rng: np.random.Generator, burn: int = 200) -> dict[str, np.ndarray]:
    """Simulate T periods; returns x, pi, i, pibar, ilag, shock levels, and innovations."""
    k = np.zeros(NK)
    n = T + burn
    eps = rng.standard_normal((n, 3))
    ks = np.empty((n, NK))
    for t in range(n):
        ks[t] = k
        k = sol.P @ k + sol.Q @ eps[t]
    ks, eps = ks[burn:], eps[burn:]
    xp = ks @ sol.F.T
    i = ks @ sol.G
    pibar = (xp[:, PI] + ks[:, 4] + ks[:, 5] + ks[:, 6]) / 4.0
    # eps[t] moves the state from t to t+1, so the innovation that hits period t is eps[t-1].
    innov = np.vstack([np.full((1, 3), np.nan), eps[:-1]])
    return {"x": xp[:, X], "pi": xp[:, PI], "i": i, "pibar": pibar, "ilag": ks[:, 3],
            "d": ks[:, 0], "u": ks[:, 1], "v": ks[:, 2], "eps_d": innov[:, 0], "eps_u": innov[:, 1],
            "eps_v": innov[:, 2]}


def irf(sol: Solution, shock: int, horizon: int = 16) -> np.ndarray:
    """Impulse responses of [x, pi, i] to a one-standard-deviation innovation in shock 0, 1, or 2."""
    k = sol.Q[:, shock].copy()
    out = np.empty((horizon, 3))
    for h in range(horizon):
        out[h] = np.r_[sol.F @ k, sol.G @ k]
        k = sol.P @ k
    return out


def frictionless_equilibrium(phi: float, rho_v: float) -> tuple[float, float]:
    """Cochrane (2011) benchmark: Fisher equation i_t = E_t pi_{t+1}, rule i_t = phi pi_t + v_t.

    For |phi| > 1 the unique bounded equilibrium is pi_t = v_t / (rho_v - phi), so i_t = rho_v pi_t.
    Returns (loading of pi on v, slope of i on pi).
    """
    return 1.0 / (rho_v - phi), rho_v
