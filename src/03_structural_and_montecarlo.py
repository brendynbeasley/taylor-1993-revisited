"""Structural New Keynesian estimation, the 2021 counterfactual, and the Monte Carlo study.

Writes:
  results/structural_estimates.csv, results/profile_phi_pi.csv, results/nk_irfs.csv,
  results/counterfactual_simulation.csv, results/monte_carlo_validation.csv,
  results/identification_surface.csv
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import SEED, estimators as est, history, nk, numbers  # noqa: E402
from core.io import PANEL, RESULTS  # noqa: E402

FREE = ["s", "kappa", "phi_pi", "phi_y", "rho_i", "rho_d", "rho_u", "rho_v", "sd_d", "sd_u", "sd_v"]
BOUNDS = [(0.01, 2.0), (0.005, 2.0), (0.0, 6.0), (0.0, 4.0), (0.0, 0.97), (0.0, 0.97), (0.0, 0.97),
          (0.0, 0.97), (0.05, 8.0), (0.05, 8.0), (0.05, 8.0)]
LAGS = (0, 1, 2, 4)
MC_REPS = 2000
TAYLOR_RULE = dict(phi_pi=1.5, phi_y=0.5, rho_i=0.0)
PI_STAR = 2.0
ZLB_FLOOR = 0.125  # midpoint of the 0 to 25 basis point target range


def moment_vector(G: dict[int, np.ndarray]) -> np.ndarray:
    iu = np.triu_indices(3)
    return np.concatenate([G[0][iu], G[1].ravel(), np.diag(G[2]), np.diag(G[4])])


def sample_autocov(Y: np.ndarray) -> dict[int, np.ndarray]:
    Yc = Y - Y.mean(0)
    n = len(Yc)
    return {j: Yc[j:].T @ Yc[: n - j] / n for j in LAGS}


def params_from(theta: np.ndarray, base: nk.Params) -> nk.Params:
    return replace(base, **dict(zip(FREE, map(float, theta))))


def model_moments(theta: np.ndarray, base: nk.Params) -> np.ndarray | None:
    sol = nk.solve(params_from(theta, base))
    if not sol.determinate:
        return None
    return moment_vector(nk.autocov(sol, LAGS))


def objective(theta: np.ndarray, base: nk.Params, m_data: np.ndarray, W: np.ndarray) -> float:
    m = model_moments(theta, base)
    if m is None or not np.all(np.isfinite(m)):
        return 1e8
    e = m_data - m
    return float(e @ W @ e)


def block_bootstrap_moments(Y: np.ndarray, rng: np.random.Generator, reps: int = 999, block: int = 8) -> np.ndarray:
    n = len(Y)
    nb = int(np.ceil(n / block))
    out = np.empty((reps, 21))
    for r in range(reps):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        out[r] = moment_vector(sample_autocov(Y[idx]))
    return out


def fit_model(m_data: np.ndarray, W: np.ndarray, base: nk.Params, rng: np.random.Generator, starts: int,
              fixed: dict[str, float] | None = None, x0: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Minimum-distance fit from several starts; ``fixed`` pins named parameters (for profiles)."""
    fixed = fixed or {}
    free_idx = [i for i, n in enumerate(FREE) if n not in fixed]
    bnds = [BOUNDS[i] for i in free_idx]

    def full(z: np.ndarray) -> np.ndarray:
        th = np.empty(len(FREE))
        th[free_idx] = z
        for name, val in fixed.items():
            th[FREE.index(name)] = val
        return th

    def f(z: np.ndarray) -> float:
        return objective(full(z), base, m_data, W)

    cands = []
    if x0 is not None:
        cands.append(np.asarray(x0)[free_idx])
    while len(cands) < starts:
        z = np.array([rng.uniform(lo + 0.1 * (hi - lo), hi - 0.4 * (hi - lo)) for lo, hi in bnds])
        if f(z) < 1e7:
            cands.append(z)
    lo_b, hi_b = [b[0] for b in bnds], [b[1] for b in bnds]
    best = (None, np.inf)
    for z0 in cands:
        r = optimize.minimize(f, z0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 600})
        if r.fun < best[1]:
            best = (r.x, r.fun)
    # Polish only the best basin: bounded Nelder-Mead, then one more gradient pass.
    r = optimize.minimize(f, best[0], method="Nelder-Mead", bounds=bnds,
                          options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-10})
    r = optimize.minimize(f, np.clip(r.x, lo_b, hi_b), method="L-BFGS-B", bounds=bnds, options={"maxiter": 600})
    z = np.clip(r.x, lo_b, hi_b)
    return full(z), f(z)


def numeric_jacobian(theta: np.ndarray, base: nk.Params, h: float = 1e-5) -> np.ndarray:
    m0 = model_moments(theta, base)
    J = np.empty((len(m0), len(theta)))
    for j in range(len(theta)):
        e = np.zeros(len(theta))
        e[j] = h
        J[:, j] = (model_moments(theta + e, base) - model_moments(theta - e, base)) / (2 * h)
    return J


def invert_shocks(sol: nk.Solution, obs: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Recover [d, u, v] from observed [x, pi, i] given the predetermined states k[3:7]."""
    H = np.vstack([sol.F, sol.G])
    return np.linalg.solve(H[:, :3], obs - H[:, 3:] @ known)


def run_counterfactual(df: pd.DataFrame, p_est: nk.Params, rule: dict, i_star: float, start: str) -> pd.DataFrame:
    """Back out shocks under the estimated model, then replay them under ``rule`` with no discretionary shocks.

    The zero lower bound is imposed with one-period policy surprises that are zero unless the rule
    would prescribe a rate below ZLB_FLOOR. The counterfactual regime has no persistent discretionary
    shocks (rho_v = 0), so agents expect each floor episode to end next quarter. With the estimated
    rho_v of 0.97 the floor would instead be expected to bind almost permanently, which produces a
    deflationary spiral that reflects the approximation rather than the economics.
    """
    sol = nk.solve(p_est)
    p_cf = replace(p_est, **rule, rho_v=0.0)
    sol_cf = nk.solve(p_cf)
    d = df[df["quarter"] >= pd.Period("2018Q1", "Q")].reset_index(drop=True)
    pi = d["infl_q_ann"].to_numpy() - PI_STAR
    x = d["gap_cbo"].to_numpy()
    i = d["ffr"].to_numpy() - i_star
    t0 = int(np.flatnonzero(d["quarter"] == pd.Period(start, "Q"))[0])
    shocks = np.full((len(d), 3), np.nan)
    for t in range(4, len(d)):
        known = np.array([i[t - 1], pi[t - 1], pi[t - 2], pi[t - 3]])
        shocks[t] = invert_shocks(sol, np.array([x[t], pi[t], i[t]]), known)
    cf = {"x": x.copy(), "pi": pi.copy(), "i": i.copy()}
    floor = np.zeros(len(d), dtype=bool)
    v = 0.0
    for t in range(t0, len(d)):
        v = p_cf.rho_v * v
        k = np.array([shocks[t, 0], shocks[t, 1], v, cf["i"][t - 1], cf["pi"][t - 1], cf["pi"][t - 2], cf["pi"][t - 3]])
        if sol_cf.G @ k + i_star < ZLB_FLOOR:
            v += (ZLB_FLOOR - i_star - sol_cf.G @ k) / sol_cf.G[2]
            k[2] = v
            floor[t] = True
        cf["x"][t], cf["pi"][t] = sol_cf.F @ k
        cf["i"][t] = sol_cf.G @ k
    out = pd.DataFrame({
        "quarter": d["quarter"].astype(str), "actual_ffr": d["ffr"], "actual_infl_q_ann": d["infl_q_ann"],
        "actual_gap": x, "cf_ffr": cf["i"] + i_star, "cf_infl_q_ann": cf["pi"] + PI_STAR, "cf_gap": cf["x"],
        "cf_at_zlb": floor, "shock_demand": shocks[:, 0], "shock_costpush": shocks[:, 1], "shock_policy": shocks[:, 2]})
    out["actual_infl_4q"] = out["actual_infl_q_ann"].rolling(4).mean()
    out["cf_infl_4q"] = out["cf_infl_q_ann"].rolling(4).mean()
    out["in_counterfactual"] = (d["quarter"] >= pd.Period(start, "Q")).to_numpy()
    return out


def mc_draw(sol: nk.Solution, p: nk.Params, T: int, rng: np.random.Generator, iv_noise: float) -> dict:
    s = nk.simulate(sol, T + 6, rng)
    keep = slice(6, None)
    y = {k: v[keep] for k, v in s.items()}
    n = T
    i, pibar, x, ilag = y["i"], y["pibar"], y["x"], y["ilag"]
    out = {}
    X = np.column_stack([np.ones(n), pibar, x])
    f = est.ols(i, X, ["c", "pi", "y"])
    out["ols_static"] = (f.get("pi"), None)
    Xs = np.column_stack([np.ones(n), ilag, pibar, x])
    fs = est.ols(i, Xs, ["c", "rho", "pi", "y"])
    out["ols_smoothing"] = (est.long_run(fs, "pi", "rho"), None)
    full_i, full_pb, full_x = s["i"], s["pibar"], s["x"]
    Z = np.column_stack([np.ones(n)] + [full_i[6 - k: len(full_i) - k] for k in range(1, 5)]
                        + [full_pb[6 - k: len(full_pb) - k] for k in range(1, 5)]
                        + [full_x[6 - k: len(full_x) - k] for k in range(1, 5)])
    g = est.gmm_linear(i, Xs, Z, ["c", "rho", "pi", "y"])
    out["gmm_lagged_instruments"] = (est.long_run(g, "pi", "rho"), g.extra["J_p"])
    if p.rho_i == 0.0:
        eu = s["eps_u"]
        zs = np.column_stack([eu[6 - k: len(eu) - k] for k in range(0, 4)])
        lhs = i - p.phi_y * x
        W1 = np.ones((n, 1))
        gi = est.iv_single(lhs, pibar, W1, zs, ["pi", "c"])
        out["iv_strong"] = (gi.get("pi"), gi.extra["F_eff"])
        zw = zs + iv_noise * rng.standard_normal(zs.shape)
        gw = est.iv_single(lhs, pibar, W1, zw, ["pi", "c"])
        f_ar = est.ols(lhs - p.phi_pi * pibar, np.column_stack([zw, W1]), [f"z{j}" for j in range(5)])
        bz, Vz = f_ar.params[:4], f_ar.cov[:4, :4]
        ar_accept = stats.chi2.sf(float(bz @ np.linalg.solve(Vz, bz)), 4) > 0.05
        out["iv_weak"] = (gw.get("pi"), (gw.extra["F_eff"], ar_accept))
    return out


def monte_carlo(p: nk.Params, label: str, T: int, reps: int, rng: np.random.Generator, iv_noise: float) -> list[dict]:
    sol = nk.solve(p)
    plim_static = nk.ols_plim(sol, smoothing=False)[0]
    plim_smooth = nk.ols_plim(sol, smoothing=True)[0]
    draws: dict[str, list] = {}
    for _ in range(reps):
        for k, v in mc_draw(sol, p, T, rng, iv_noise).items():
            draws.setdefault(k, []).append(v)
    rows = []
    z = stats.norm.ppf(0.975)
    for name, vals in draws.items():
        b = np.array([v[0][0] for v in vals])
        se = np.array([v[0][1] for v in vals])
        ok = np.isfinite(b) & np.isfinite(se)
        b, se = b[ok], se[ok]
        cover = np.mean(np.abs(b - p.phi_pi) <= z * se)
        row = {"dgp": label, "T": T, "reps": int(ok.sum()), "estimator": name, "true_phi_pi": p.phi_pi,
               "mean": b.mean(), "median": np.median(b), "bias": b.mean() - p.phi_pi,
               "median_bias": np.median(b) - p.phi_pi, "emp_se": b.std(ddof=1),
               "rmse": np.sqrt(np.mean((b - p.phi_pi) ** 2)), "coverage_95": cover,
               "analytic_plim": plim_static if name == "ols_static" else (plim_smooth if name == "ols_smoothing" else np.nan)}
        if name == "gmm_lagged_instruments":
            row["J_reject_5pct"] = float(np.mean(np.array([v[1] for v in vals])[ok] < 0.05))
        if name == "iv_strong":
            row["mean_F_eff"] = float(np.mean(np.array([v[1] for v in vals])[ok]))
        if name == "iv_weak":
            row["mean_F_eff"] = float(np.mean(np.array([v[1][0] for v in vals])[ok]))
            row["ar_coverage_95"] = float(np.mean(np.array([v[1][1] for v in vals])[ok]))
        rows.append(row)
    return rows


def main() -> None:
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    kn: dict = {}
    df = pd.read_parquet(PANEL)
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

    # ---------------- Cochrane (2011): the frictionless benchmark ----------------
    for phi, rho_v in [(1.5, 0.7), (3.0, 0.7), (1.5, 0.3)]:
        v = np.zeros(200_000)
        e = rng.standard_normal(len(v))
        for t in range(1, len(v)):
            v[t] = rho_v * v[t - 1] + e[t]
        load, slope = nk.frictionless_equilibrium(phi, rho_v)
        pi = load * v
        i = phi * pi + v
        b = np.polyfit(pi, i, 1)[0]
        kn[f"cochrane_phi{phi}_rho{rho_v}_ols"] = float(b)
    kn["cochrane_note"] = "OLS slope equals rho_v, not phi, in the frictionless model"
    numbers.update("settings", {
        "seed": SEED, "mc_reps": MC_REPS, "mc_reps_large_T": 500, "mc_large_T": 1000, "bootstrap_reps": 999,
        "bootstrap_block": 8, "md_moments": 21, "md_free_params": len(FREE), "md_starts": 10, "beta": nk.Params().beta,
        "qlr_trim": 0.15, "qlr_null_reps": 4000, "qlr_null_steps": 1000, "rolling_window": 40, "rolling_min_obs": 30,
        "cochrane_sim_T": 200_000, "profile_points": 19, "iv_noise_calibration_draws": 100, "dpi": 300,
        "eigen_tolerance": 1e-10, "ar_grid_lo": -10, "ar_grid_hi": 15, "ar_grid_points": 2501,
    })

    # ---------------- Minimum-distance estimation, 1987Q3-2007Q4 ----------------
    m = history.sample_mask(df, "greenspan_bernanke")
    d = df.loc[m, ["quarter", "gap_cbo", "infl_q_ann", "ffr"]].dropna()
    Y = d[["gap_cbo", "infl_q_ann", "ffr"]].to_numpy()
    m_data = moment_vector(sample_autocov(Y))
    boot = block_bootstrap_moments(Y, rng)
    Omega = np.cov(boot.T)
    W = np.diag(1.0 / np.diag(Omega))
    base = nk.Params()
    theta, val = fit_model(m_data, W, base, rng, starts=10)
    p_est = params_from(theta, base)
    sol_est = nk.solve(p_est)
    G = numeric_jacobian(theta, base)
    A = np.linalg.pinv(G.T @ W @ G)
    V = A @ G.T @ W @ Omega @ W @ G @ A
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    at_bound = [abs(theta[j] - BOUNDS[j][0]) < 1e-3 or abs(theta[j] - BOUNDS[j][1]) < 1e-3 for j in range(len(FREE))]
    m_fit = model_moments(theta, base)
    est_tab = pd.DataFrame({"parameter": FREE, "estimate": theta, "se": se, "at_bound": at_bound})
    est_tab = pd.concat([est_tab, pd.DataFrame([
        {"parameter": "beta (fixed)", "estimate": base.beta, "se": np.nan, "at_bound": False},
        {"parameter": "objective", "estimate": val, "se": np.nan, "at_bound": False},
        {"parameter": "N quarters", "estimate": len(d), "se": np.nan, "at_bound": False}])])
    est_tab.to_csv(RESULTS / "structural_estimates.csv", index=False)
    for name, b, s in zip(FREE, theta, se):
        kn[f"nk_{name}"], kn[f"nk_{name}_se"] = b, s
    kn["nk_N"] = len(d)
    kn["nk_objective"] = val
    kn["nk_moment_fit_corr"] = float(np.corrcoef(m_data, m_fit)[0, 1])
    kn["nk_plim_ols_static"] = float(nk.ols_plim(sol_est, smoothing=False)[0])
    kn["nk_plim_ols_smoothing"] = float(nk.ols_plim(sol_est, smoothing=True)[0])
    kn["nk_plim_ols_smoothing_phi_y"] = float(nk.ols_plim(sol_est, smoothing=True)[1])

    # Profile of the objective in phi_pi: how sharply do the moments pin it down?
    prof = []
    x0 = theta.copy()
    for phi in np.r_[np.linspace(0.6, 1.0, 5), np.linspace(1.1, 5.0, 14)]:
        # Two starts per point: the global estimate and the neighbouring profile solution.
        tries = [fit_model(m_data, W, base, rng, starts=1, fixed={"phi_pi": float(phi)}, x0=start)
                 for start in (theta, x0)]
        th, v_ = min(tries, key=lambda r: r[1])
        x0 = th
        prof.append({"phi_pi": phi, "objective": v_, "determinate": v_ < 1e7, "rho_v": th[FREE.index("rho_v")],
                     "sd_v": th[FREE.index("sd_v")], "phi_y": th[FREE.index("phi_y")]})
    prof = pd.DataFrame(prof)
    prof["delta_objective"] = prof["objective"] - val
    prof.to_csv(RESULTS / "profile_phi_pi.csv", index=False)
    det = prof[prof["determinate"]]
    kn["profile_phi_pi_range_within_1"] = [float(det.loc[det["delta_objective"] < 1.0, "phi_pi"].min()),
                                           float(det.loc[det["delta_objective"] < 1.0, "phi_pi"].max())]
    kn["profile_grid_lo"], kn["profile_grid_hi"] = float(prof["phi_pi"].min()), float(prof["phi_pi"].max())
    kn["profile_max_delta"] = float(det["delta_objective"].max())
    kn["profile_argmin"] = float(det.loc[det["objective"].idxmin(), "phi_pi"])

    # ---------------- Impulse responses: estimated rule versus Taylor's rule ----------------
    sol_tr = nk.solve(replace(p_est, **TAYLOR_RULE))
    irfs = []
    for lab, sol in [("estimated rule", sol_est), ("Taylor (1993) rule", sol_tr)]:
        for k, shock in enumerate(["demand", "cost-push", "policy"]):
            r = nk.irf(sol, k, 16)
            for h in range(16):
                irfs.append({"rule": lab, "shock": shock, "h": h, "gap": r[h, 0], "inflation": r[h, 1], "rate": r[h, 2]})
    pd.DataFrame(irfs).to_csv(RESULTS / "nk_irfs.csv", index=False)

    # ---------------- Counterfactual: the 2021-2023 inflation under Taylor's rule ----------------
    cfs = []
    for lab, rule, r_star in [("Taylor (1993) rule, r* = 2", TAYLOR_RULE, 2.0),
                              ("Taylor (1993) rule, r* = 0.5", TAYLOR_RULE, 0.5),
                              ("estimated 1987-2007 rule, no deviations", {}, 2.0)]:
        rule_full = rule if rule else dict(phi_pi=p_est.phi_pi, phi_y=p_est.phi_y, rho_i=p_est.rho_i)
        out = run_counterfactual(df, p_est, rule_full, r_star + PI_STAR, "2021Q1")
        out.insert(0, "scenario", lab)
        cfs.append(out)
    cf = pd.concat(cfs)
    cf.to_csv(RESULTS / "counterfactual_simulation.csv", index=False)
    for lab, key in [("Taylor (1993) rule, r* = 2", "cf_taylor"), ("Taylor (1993) rule, r* = 0.5", "cf_taylor_lowr"),
                     ("estimated 1987-2007 rule, no deviations", "cf_own")]:
        c = cf[(cf["scenario"] == lab) & cf["in_counterfactual"]]
        w23 = c[c["quarter"].str[:4].isin(["2021", "2022", "2023"])]
        kn[f"{key}_peak_infl_4q"] = c["cf_infl_4q"].max()
        kn[f"{key}_peak_infl_4q_quarter"] = c.loc[c["cf_infl_4q"].idxmax(), "quarter"]
        kn[f"{key}_mean_infl_2021_2023"] = w23["cf_infl_q_ann"].mean()
        kn[f"{key}_mean_gap_2021_2023"] = w23["cf_gap"].mean()
        kn[f"{key}_peak_ffr"] = c["cf_ffr"].max()
        kn[f"{key}_peak_ffr_quarter"] = c.loc[c["cf_ffr"].idxmax(), "quarter"]
        kn[f"{key}_ffr_2021Q4"] = float(c.loc[c["quarter"] == "2021Q4", "cf_ffr"].iloc[0])
        kn[f"{key}_cum_gap_diff_2021_2023"] = float((w23["cf_gap"] - w23["actual_gap"]).sum() / 4)
        kn[f"{key}_min_ffr"] = c["cf_ffr"].min()
        kn[f"{key}_zlb_quarters"] = int(c["cf_at_zlb"].sum())
        kn[f"{key}_trough_gap"] = c["cf_gap"].min()
        kn[f"{key}_trough_gap_quarter"] = c.loc[c["cf_gap"].idxmin(), "quarter"]
    a = cf[(cf["scenario"] == "Taylor (1993) rule, r* = 2") & cf["in_counterfactual"]]
    a23 = a[a["quarter"].str[:4].isin(["2021", "2022", "2023"])]
    kn["actual_peak_infl_4q"] = a["actual_infl_4q"].max()
    kn["actual_peak_infl_4q_quarter"] = a.loc[a["actual_infl_4q"].idxmax(), "quarter"]
    kn["actual_mean_infl_2021_2023"] = a23["actual_infl_q_ann"].mean()
    kn["actual_mean_gap_2021_2023"] = a23["actual_gap"].mean()
    kn["actual_ffr_2021Q4"] = float(a.loc[a["quarter"] == "2021Q4", "actual_ffr"].iloc[0])
    kn["cf_window_end"] = a["quarter"].iloc[-1]
    sac = (kn["actual_mean_infl_2021_2023"] - kn["cf_taylor_mean_infl_2021_2023"])
    kn["cf_taylor_output_cost_per_point"] = float(-kn["cf_taylor_cum_gap_diff_2021_2023"] / (sac * 3)) if sac else np.nan

    # ---------------- Identification surface: OLS plim against policy-shock persistence and size ----------------
    surf = []
    for rv in np.linspace(0.0, 0.95, 24):
        for sv in np.linspace(0.05, 3.0, 24):
            sol = nk.solve(replace(p_est, rho_v=float(rv), sd_v=float(sv)))
            if sol.determinate:
                surf.append({"rho_v": rv, "sd_v": sv, "plim_smoothing": nk.ols_plim(sol, smoothing=True)[0],
                             "plim_static": nk.ols_plim(sol, smoothing=False)[0]})
    pd.DataFrame(surf).to_csv(RESULTS / "identification_surface.csv", index=False)

    # ---------------- Monte Carlo ----------------
    p_static = replace(p_est, rho_i=0.0)
    # Weak-instrument noise chosen so the simulated first-stage strength matches the data (effective F near 2).
    econ = numbers.load().get("econometrics", {})
    target_f = econ.get("s16_F_eff", 2.0)
    noise_grid = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    f_by_noise = []
    for nz in noise_grid:
        fs = []
        for _ in range(100):
            r = mc_draw(nk.solve(p_static), p_static, kn["nk_N"], rng, nz)
            fs.append(r["iv_weak"][1][0])
        f_by_noise.append(np.mean(fs))
    iv_noise = float(np.interp(target_f, f_by_noise[::-1], noise_grid[::-1]))
    kn["mc_iv_noise"] = iv_noise
    kn["mc_target_F"] = target_f
    rows = []
    rows += monte_carlo(p_est, "estimated NK model (partial adjustment)", kn["nk_N"], MC_REPS, rng, iv_noise)
    rows += monte_carlo(p_est, "estimated NK model (partial adjustment)", 1000, 500, rng, iv_noise)
    rows += monte_carlo(p_static, "estimated NK model, static rule", kn["nk_N"], MC_REPS, rng, iv_noise)
    rows += monte_carlo(p_static, "estimated NK model, static rule", 1000, 500, rng, iv_noise)
    mc = pd.DataFrame(rows)
    mc.to_csv(RESULTS / "monte_carlo_validation.csv", index=False)
    for _, r in mc.iterrows():
        tag = f"mc_{'smooth' if 'partial' in r['dgp'] else 'static'}_T{r['T']}_{r['estimator']}"
        for col in ["mean", "bias", "rmse", "coverage_95", "analytic_plim", "J_reject_5pct", "mean_F_eff", "ar_coverage_95"]:
            if col in r and pd.notna(r[col]):
                kn[f"{tag}_{col}"] = r[col]

    kn["runtime_seconds"] = time.time() - t_start
    numbers.update("structural", kn)
    print(est_tab.round(3).to_string())
    print(prof.round(3).to_string())
    print(mc.round(3).to_string())
    for k in sorted(kn):
        if k.startswith(("cf_", "actual_", "cochrane", "nk_plim", "profile")):
            print(k, kn[k])


if __name__ == "__main__":
    main()
