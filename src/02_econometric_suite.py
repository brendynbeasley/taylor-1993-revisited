"""Stage 1 replication, Stage 2 hardening, and the critics' tests for the Taylor rule.

Reads the committed analytical panel and writes:
  results/replication_fit.csv, results/regression_horserace.csv,
  results/diagnostics_and_sensitivity.csv, results/rule_paths.csv,
  results/rolling_rule.csv, results/chair_deviations.csv, results/named_entities_table.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import SEED, estimators as est, history, numbers  # noqa: E402
from core.io import PANEL, RESULTS  # noqa: E402

TAYLOR = {"const": 1.0, "phi_pi": 1.5, "phi_y": 0.5}  # eq. (1): r = p + .5y + .5(p - 2) + 2
MOP_TAU = 0.10  # worst-case Nagar bias tolerance for the weak-instrument threshold

# Cited values from the original paper and the benchmark it is compared against. Nothing here is estimated.
BENCHMARKS = [
    ("Taylor (1993)", "p. 202, eq. (1)", "inflation coefficient (1 + 0.5)", 1.5, None),
    ("Taylor (1993)", "p. 202, eq. (1)", "output gap coefficient", 0.5, None),
    ("Taylor (1993)", "p. 202, eq. (1)", "inflation target, percent", 2.0, None),
    ("Taylor (1993)", "p. 202, eq. (1)", "equilibrium real rate, percent", 2.0, None),
    ("Taylor (1993)", "p. 202, eq. (1)", "implied intercept, 2 - 0.5 x 2", 1.0, None),
    ("Taylor (1993)", "p. 202, eq. (1)", "implied nominal rate at target, percent", 4.0, None),
    ("Taylor (1993)", "p. 202", "trend real GDP growth, percent per year", 2.2, None),
    ("Taylor (1993)", "p. 202 and Figure 1", "first year of the fit window", 1987, None),
    ("Taylor (1993)", "p. 202 and Figure 1", "last year of the fit window", 1992, None),
    ("Taylor (1993)", "p. 206, Figure 3", "inflation measure: GDP deflator, four quarters", None, None),
    ("Clarida, Gali, Gertler (1998 NBER WP 6442)", "Table 2, k=4, q=1, 1960:1-1979:2", "beta (inflation)", 0.80, 0.09),
    ("Clarida, Gali, Gertler (1998 NBER WP 6442)", "Table 2, k=4, q=1, 1960:1-1979:2", "gamma (output gap)", 0.52, 0.12),
    ("Clarida, Gali, Gertler (1998 NBER WP 6442)", "Table 2, k=4, q=1, 1960:1-1979:2", "rho (smoothing)", 0.76, 0.04),
    ("Clarida, Gali, Gertler (1998 NBER WP 6442)", "Table 2, k=4, q=1, 1960:1-1979:2", "J-test p-value", 0.84, None),
    ("Clarida, Gali, Gertler (1998 NBER WP 6442)", "Table 2, k=4, q=1, 1979:3-1996:4", "beta (inflation)", 1.96, 0.20),
    ("Clarida, Gali, Gertler (1998 NBER WP 6442)", "Table 2, k=4, q=1, 1979:3-1996:4", "gamma (output gap)", 0.07, 0.10),
    ("Clarida, Gali, Gertler (1998 NBER WP 6442)", "Table 2, k=4, q=1, 1979:3-1996:4", "rho (smoothing)", 0.66, 0.03),
    ("Clarida, Gali, Gertler (1998 NBER WP 6442)", "Table 2, k=4, q=1, 1979:3-1996:4", "J-test p-value", 0.89, None),
]


def taylor_rule(pi: pd.Series, y: pd.Series) -> pd.Series:
    return TAYLOR["const"] + TAYLOR["phi_pi"] * pi + TAYLOR["phi_y"] * y


def load() -> pd.DataFrame:
    df = pd.read_parquet(PANEL)
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df = df.sort_values("quarter").reset_index(drop=True)
    for c in ["infl_4q", "gap_cbo", "ffr"]:
        for k in range(1, 5):
            df[f"{c}_l{k}"] = df[c].shift(k)
    df["ffr_l1"] = df["ffr_lag1"]
    for k in range(0, 4):
        df[f"oil_s_l{k}"] = df["oil_surprise"].shift(k)
        df[f"oil_n_l{k}"] = df["oil_news_shock"].shift(k)
    for k in range(1, 5):
        df[f"oil_s_f{k}"] = df["oil_surprise"].shift(-k)
    return df


def take(df: pd.DataFrame, sample: str, cols: list[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(cols))
    sub = df.loc[history.sample_mask(df, sample), ["quarter", *cols]].dropna()
    return sub.reset_index(drop=True)


def row(spec: str, label: str, stage: str, sample: pd.DataFrame, sample_name: str, estimator: str, data: str,
        phi_pi: tuple[float, float] | None, phi_y: tuple[float, float] | None, rho: float | None = None,
        r2: float | None = None, f_eff: float | None = None, j_p: float | None = None, note: str = "") -> dict:
    b, s = phi_pi if phi_pi else (np.nan, np.nan)
    by, sy = phi_y if phi_y else (np.nan, np.nan)
    z = stats.norm.ppf(0.975)
    return {
        "spec": spec, "label": label, "stage": stage, "sample": sample_name,
        "period": f"{sample['quarter'].iloc[0]}-{sample['quarter'].iloc[-1]}" if len(sample) else "",
        "N": len(sample), "estimator": estimator, "data": data,
        "phi_pi": b, "se_pi": s, "ci_lo": b - z * s, "ci_hi": b + z * s,
        "t_pi_eq_1": (b - 1) / s if s and np.isfinite(s) else np.nan,
        "p_pi_le_1": float(stats.norm.sf((b - 1) / s)) if s and np.isfinite(s) else np.nan,
        "phi_y": by, "se_y": sy, "rho": rho if rho is not None else np.nan,
        "r2": r2 if r2 is not None else np.nan, "F_eff": f_eff if f_eff is not None else np.nan,
        "J_p": j_p if j_p is not None else np.nan, "note": note,
    }


def static_ols(d: pd.DataFrame, pi: str, y: str):
    X = np.column_stack([np.ones(len(d)), d[pi], d[y]])
    return est.ols(d["ffr"].to_numpy(), X, ["const", "pi", "y"])


def smooth_ols(d: pd.DataFrame, pi: str, y: str):
    X = np.column_stack([np.ones(len(d)), d["ffr_l1"], d[pi], d[y]])
    f = est.ols(d["ffr"].to_numpy(), X, ["const", "rho", "pi", "y"])
    return f, est.long_run(f, "pi", "rho"), est.long_run(f, "y", "rho")


def main() -> None:
    df = load()
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    diag: list[dict] = []
    kn: dict = {}

    # ---------------- Stage 1a: Taylor's Figure 1, three data vintages ----------------
    w = take(df, "taylor_1987_1992", ["ffr", "rt92_infl_4q", "rt92_gap", "infl_4q", "gap_taylor_trend", "gap_cbo"])
    fits = []
    for key, label, p, y in [("vintage_1992", "1992Q4 vintage, 2.2% trend gap", "rt92_infl_4q", "rt92_gap"),
                             ("revised_trend", "current vintage, 2.2% trend gap", "infl_4q", "gap_taylor_trend"),
                             ("revised_cbo", "current vintage, CBO output gap", "infl_4q", "gap_cbo")]:
        rule = taylor_rule(w[p], w[y])
        dev = w["ffr"] - rule
        imax = int(np.argmax(np.abs(dev)))
        fits.append({"data": key, "label": label, "N": len(w), "rmse": float(np.sqrt(np.mean(dev ** 2))),
                     "mean_dev": float(dev.mean()), "corr": float(np.corrcoef(w["ffr"], rule)[0, 1]),
                     "max_abs_dev": float(dev.iloc[imax]), "max_dev_quarter": str(w["quarter"].iloc[imax])})
        kn[f"fit_{key}_rmse"] = fits[-1]["rmse"]
        kn[f"fit_{key}_corr"] = fits[-1]["corr"]
        kn[f"fit_{key}_mean_dev"] = fits[-1]["mean_dev"]
    pd.DataFrame(fits).to_csv(RESULTS / "replication_fit.csv", index=False)
    kn["fit_rmse_ratio_cbo_to_vintage"] = fits[2]["rmse"] / fits[0]["rmse"]
    kn["taylor_window_N"] = len(w)
    q92 = w[w["quarter"] == pd.Period("1992Q3", "Q")].iloc[0]
    kn["gap_1992Q3_vintage"] = q92["rt92_gap"]
    kn["gap_1992Q3_cbo"] = q92["gap_cbo"]
    kn["gap_1992Q3_trend_revised"] = q92["gap_taylor_trend"]
    kn["infl_1992Q3_vintage"] = q92["rt92_infl_4q"]
    kn["infl_1992Q3_revised"] = q92["infl_4q"]
    kn["ffr_1992Q3"] = q92["ffr"]
    kn["rule_1992Q3_vintage"] = float(taylor_rule(q92["rt92_infl_4q"], q92["rt92_gap"]))
    kn["rule_1992Q3_cbo"] = float(taylor_rule(q92["infl_4q"], q92["gap_cbo"]))
    # Worked-example arithmetic for the README, so each displayed step has a computed source.
    kn["wx1992_infl_term"] = TAYLOR["phi_pi"] * q92["rt92_infl_4q"]
    kn["wx1992_gap_term"] = TAYLOR["phi_y"] * q92["rt92_gap"]
    kn["wx1992_gap_from_rate"] = q92["ffr"] - kn["rule_1992Q3_vintage"]
    q22 = df[df["quarter"] == pd.Period("2022Q2", "Q")].iloc[0]
    kn["wx2022_infl_term"] = TAYLOR["phi_pi"] * q22["infl_4q"]
    kn["wx2022_gap_term"] = TAYLOR["phi_y"] * q22["gap_cbo"]
    kn["wx2022_rule"] = float(taylor_rule(q22["infl_4q"], q22["gap_cbo"]))
    kn["wx2022_real_rate"] = q22["ffr"] - q22["infl_4q"]

    # ---------------- Stage 1b: unrestricted rule on Taylor's own window ----------------
    rows.append(row("S0", "Taylor (1993) rule as written", "Original", w, "taylor_1987_1992", "calibrated",
                    "Taylor (1993), p. 202, eq. (1)", (1.5, np.nan), (0.5, np.nan),
                    note="Coefficients chosen, not estimated"))
    f = static_ols(w, "rt92_infl_4q", "rt92_gap")
    rows.append(row("S1", "Unrestricted rule, Taylor's window, 1992 vintage", "Stage 1", w, "taylor_1987_1992",
                    "OLS, HAC", "1992Q4 vintage", f.get("pi"), f.get("y"), r2=f.extra["r2"]))
    kn["s1_phi_pi"], kn["s1_se_pi"] = f.get("pi")
    kn["s1_phi_y"], kn["s1_se_y"] = f.get("y")
    f = static_ols(w, "infl_4q", "gap_cbo")
    rows.append(row("S2", "Unrestricted rule, Taylor's window, revised data", "Stage 1", w, "taylor_1987_1992",
                    "OLS, HAC", "current vintage, CBO gap", f.get("pi"), f.get("y"), r2=f.extra["r2"]))
    kn["s2_phi_pi"], kn["s2_se_pi"] = f.get("pi")

    gb = take(df, "greenspan_bernanke", ["ffr", "ffr_l1", "infl_4q", "gap_cbo"])
    f = static_ols(gb, "infl_4q", "gap_cbo")
    rows.append(row("S3", "Static rule, Greenspan-Bernanke era", "Stage 1", gb, "greenspan_bernanke", "OLS, HAC",
                    "current vintage, CBO gap", f.get("pi"), f.get("y"), r2=f.extra["r2"]))
    kn["s3_phi_pi"], kn["s3_se_pi"] = f.get("pi")
    kn["s3_phi_y"], kn["s3_se_y"] = f.get("y")
    kn["s3_N"] = len(gb)

    # ---------------- Stage 2: smoothing, real time, forward-looking GMM, the ZLB ----------------
    f, lp, ly = smooth_ols(gb, "infl_4q", "gap_cbo")
    rows.append(row("S4", "Partial adjustment, Greenspan-Bernanke era", "Stage 2", gb, "greenspan_bernanke",
                    "OLS, HAC, delta method", "current vintage, CBO gap", lp, ly, rho=f.get("rho")[0], r2=f.extra["r2"]))
    kn["s4_phi_pi"], kn["s4_se_pi"] = lp
    kn["s4_phi_y"], kn["s4_se_y"] = ly
    kn["s4_rho"] = f.get("rho")[0]
    gb_fit = f

    rt = take(df, "realtime_greenbook", ["ffr", "ffr_l1", "infl_4q", "gap_cbo", "gb_infl_4q", "gb_gap", "gb_infl_f4"])
    f_rev, lp_rev, ly_rev = smooth_ols(rt, "infl_4q", "gap_cbo")
    rows.append(row("S5", "Partial adjustment, revised data, real-time quarters", "Stage 2", rt, "realtime_greenbook",
                    "OLS, HAC, delta method", "current vintage, CBO gap", lp_rev, ly_rev, rho=f_rev.get("rho")[0],
                    r2=f_rev.extra["r2"]))
    f_rt, lp_rt, ly_rt = smooth_ols(rt, "gb_infl_4q", "gb_gap")
    rows.append(row("S6", "Partial adjustment, Greenbook real-time data", "Stage 2", rt, "realtime_greenbook",
                    "OLS, HAC, delta method", "Greenbook real time", lp_rt, ly_rt, rho=f_rt.get("rho")[0],
                    r2=f_rt.extra["r2"]))
    f_fw, lp_fw, ly_fw = smooth_ols(rt, "gb_infl_f4", "gb_gap")
    rows.append(row("S7", "Forward-looking, Greenbook one-year forecast", "Stage 2", rt, "realtime_greenbook",
                    "OLS, HAC, delta method", "Greenbook real time", lp_fw, ly_fw, rho=f_fw.get("rho")[0],
                    r2=f_fw.extra["r2"]))
    kn.update({"s5_phi_pi": lp_rev[0], "s5_se_pi": lp_rev[1], "s5_phi_y": ly_rev[0], "s5_se_y": ly_rev[1],
               "s6_phi_pi": lp_rt[0], "s6_se_pi": lp_rt[1], "s6_phi_y": ly_rt[0], "s6_se_y": ly_rt[1],
               "s7_phi_pi": lp_fw[0], "s7_se_pi": lp_fw[1], "s7_phi_y": ly_fw[0], "s7_se_y": ly_fw[1],
               "realtime_N": len(rt), "realtime_period": f"{rt['quarter'].iloc[0]} to {rt['quarter'].iloc[-1]}"})

    zcols = ["ffr_l1", "ffr_l2", "ffr_l3", "ffr_l4", "infl_4q_l1", "infl_4q_l2", "infl_4q_l3", "infl_4q_l4",
             "gap_cbo_l1", "gap_cbo_l2", "gap_cbo_l3", "gap_cbo_l4"]
    for spec, sname, lab in [("S8", "volcker_greenspan", "1979Q3-2007Q4"),
                             ("S8b", "cgg_volcker_greenspan", "1979Q3-1996Q4 (CGG window)"),
                             ("S9", "pre_volcker", "1965Q1-1979Q2"),
                             ("S9b", "cgg_pre_volcker", "1960Q1-1979Q2 (CGG window)")]:
        d = take(df, sname, ["ffr", "ffr_l1", "infl_f4", "gap_cbo", *zcols])
        X = np.column_stack([np.ones(len(d)), d["ffr_l1"], d["infl_f4"], d["gap_cbo"]])
        Z = np.column_stack([np.ones(len(d)), d[zcols]])
        g = est.gmm_linear(d["ffr"].to_numpy(), X, Z, ["const", "rho", "pi", "y"])
        lp, ly = est.long_run(g, "pi", "rho"), est.long_run(g, "y", "rho")
        rows.append(row(spec, f"Forward-looking GMM, {lab}", "Stage 2", d, sname,
                        "two-step GMM, HAC", "current vintage, realized inflation t+1..t+4", lp, ly,
                        rho=g.get("rho")[0], j_p=g.extra["J_p"], note="instruments: four lags of rate, inflation, gap"))
        kn[f"{spec.lower()}_phi_pi"], kn[f"{spec.lower()}_se_pi"] = lp
        kn[f"{spec.lower()}_J_p"] = g.extra["J_p"]
        kn[f"{spec.lower()}_N"] = len(d)

    for spec, sname, lab in [("S10", "pre_volcker", "pre-Volcker"), ("S11", "volcker_greenspan", "Volcker-Greenspan")]:
        d = take(df, sname, ["ffr", "ffr_l1", "infl_4q", "gap_cbo"])
        f, lp, ly = smooth_ols(d, "infl_4q", "gap_cbo")
        rows.append(row(spec, f"Partial adjustment, {lab}", "Stage 2", d, sname, "OLS, HAC, delta method",
                        "current vintage, CBO gap", lp, ly, rho=f.get("rho")[0], r2=f.extra["r2"]))
        kn[f"{spec.lower()}_phi_pi"], kn[f"{spec.lower()}_se_pi"] = lp

    tb = df[(df["quarter"] >= pd.Period("1987Q3", "Q"))][["quarter", "ffr", "infl_4q", "gap_cbo", "zlb"]].dropna()
    X = np.column_stack([np.ones(len(tb)), tb["infl_4q"], tb["gap_cbo"]])
    t = est.tobit_left(tb["ffr"].to_numpy(), X, ["const", "pi", "y"], censor=0.25)
    rows.append(row("S12", "Static rule, ZLB as censoring, 1987 onward", "Stage 2", tb,
                    "1987Q3-latest incl. ZLB", "Tobit MLE", "current vintage, CBO gap", t.get("pi"), t.get("y"),
                    note=f"{t.extra['n_censored']} censored quarters"))
    kn["s12_phi_pi"], kn["s12_se_pi"] = t.get("pi")
    kn["s12_phi_y"], kn["s12_se_y"] = t.get("y")
    kn["s12_N"], kn["s12_censored"] = len(tb), t.extra["n_censored"]
    pz = df[history.sample_mask(df, "post_2008")][["quarter", "ffr", "infl_4q", "gap_cbo"]].dropna()
    X = np.column_stack([np.ones(len(pz)), pz["infl_4q"], pz["gap_cbo"]])
    t2 = est.tobit_left(pz["ffr"].to_numpy(), X, ["const", "pi", "y"], censor=0.25)
    rows.append(row("S13", "Static rule, ZLB as censoring, 2008 onward", "Stage 2", pz, "post_2008", "Tobit MLE",
                    "current vintage, CBO gap", t2.get("pi"), t2.get("y"), note=f"{t2.extra['n_censored']} censored quarters"))
    kn["s13_phi_pi"], kn["s13_se_pi"] = t2.get("pi")
    kn["s13_N"] = len(pz)

    # ---------------- Stage 3: the identification critic, an external instrument ----------------
    ivc = ["ffr", "infl_4q", "gap_cbo", "oil_s_l0", "oil_s_l1", "oil_s_l2", "oil_s_l3",
           "oil_n_l0", "oil_n_l1", "oil_n_l2", "oil_n_l3", "oil_s_f1", "oil_s_f2", "oil_s_f3", "oil_s_f4"]
    iv = take(df, "oil_iv", ivc)
    lhs = (iv["ffr"] - TAYLOR["phi_y"] * iv["gap_cbo"]).to_numpy()
    Wc = np.ones((len(iv), 1))
    f = est.ols(lhs, np.column_stack([np.ones(len(iv)), iv["infl_4q"]]), ["const", "pi"])
    rows.append(row("S14", "OLS on the instrument sample (gap response fixed at 0.5)", "Stage 3", iv, "oil_iv",
                    "OLS, HAC", "current vintage", f.get("pi"), (0.5, np.nan), r2=f.extra["r2"]))
    kn["s14_phi_pi"], kn["s14_se_pi"] = f.get("pi")
    grid = np.linspace(-10, 15, 2501)
    mop_crit = float(stats.ncx2.ppf(0.95, 1, 1 / MOP_TAU))  # simplified conservative MOP critical value
    kn["mop_crit"] = mop_crit
    for spec, prefix, lab in [("S15", "oil_s", "OPEC announcement surprises"), ("S16", "oil_n", "oil supply news shocks")]:
        Z = iv[[f"{prefix}_l{k}" for k in range(4)]].to_numpy()
        g = est.iv_single(lhs, iv["infl_4q"].to_numpy(), Wc, Z, ["pi", "const"])
        ar = est.anderson_rubin_set(lhs, iv["infl_4q"].to_numpy(), Wc, Z, grid)
        note = (f"AR 95% set [{ar['lo']:.2f}, {ar['hi']:.2f}]" + ("" if ar["bounded"] else ", unbounded within grid")
                + ("" if ar["connected"] else ", disjoint"))
        rows.append(row(spec, f"2SLS, deflator inflation on Kanzig {lab}", "Stage 3", iv, "oil_iv",
                        "2SLS, HAC", "current vintage", g.get("pi"), (0.5, np.nan), f_eff=g.extra["F_eff"], note=note))
        kn[f"{spec.lower()}_phi_pi"], kn[f"{spec.lower()}_se_pi"] = g.get("pi")
        kn[f"{spec.lower()}_F_eff"] = g.extra["F_eff"]
        kn[f"{spec.lower()}_ar_lo"], kn[f"{spec.lower()}_ar_hi"] = ar["lo"], ar["hi"]
        kn[f"{spec.lower()}_ar_bounded"] = ar["bounded"]
        diag.append({"test": f"first stage {lab}", "statistic": "Montiel Olea-Pflueger effective F",
                     "value": g.extra["F_eff"], "reference": f"{mop_crit:.2f} (5% test, 10% worst-case bias)",
                     "note": f"HAC robust first-stage F {g.extra['F_robust']:.2f}"})
    ivp = take(df, "oil_iv", ["ffr", "cpi_infl_4q", "gap_cbo", "oil_n_l0", "oil_n_l1", "oil_n_l2", "oil_n_l3"])
    lhs_c = (ivp["ffr"] - TAYLOR["phi_y"] * ivp["gap_cbo"]).to_numpy()
    Zc = ivp[[f"oil_n_l{k}" for k in range(4)]].to_numpy()
    gc = est.iv_single(lhs_c, ivp["cpi_infl_4q"].to_numpy(), np.ones((len(ivp), 1)), Zc, ["pi", "const"])
    arc = est.anderson_rubin_set(lhs_c, ivp["cpi_infl_4q"].to_numpy(), np.ones((len(ivp), 1)), Zc, grid)
    rows.append(row("S17", "2SLS, CPI inflation on Kanzig oil supply news shocks", "Stage 3", ivp, "oil_iv",
                    "2SLS, HAC", "current vintage, CPI", gc.get("pi"), (0.5, np.nan), f_eff=gc.extra["F_eff"],
                    note=f"AR 95% set [{arc['lo']:.2f}, {arc['hi']:.2f}]" + ("" if arc["bounded"] else ", unbounded within grid")))
    kn["s17_F_eff"] = gc.extra["F_eff"]
    kn["s17_ar_lo"], kn["s17_ar_hi"], kn["s17_ar_bounded"] = arc["lo"], arc["hi"], arc["bounded"]
    kn["s17_phi_pi"], kn["s17_se_pi"] = gc.get("pi")
    kn["iv_N"] = len(iv)
    kn["iv_period"] = f"{iv['quarter'].iloc[0]} to {iv['quarter'].iloc[-1]}"
    Zlead = iv[[f"oil_s_f{k}" for k in range(1, 5)]].to_numpy()
    gl = est.iv_single(lhs, iv["infl_4q"].to_numpy(), Wc, Zlead, ["pi", "const"])
    diag.append({"test": "placebo: future OPEC surprises as instruments", "statistic": "effective F",
                 "value": gl.extra["F_eff"], "reference": "should be near zero",
                 "note": "leads t+1..t+4 cannot move current inflation"})
    kn["placebo_lead_F_eff"] = gl.extra["F_eff"]

    # ---------------- Orphanides: real-time measurement and the attenuation formula ----------------
    rev = rt["gap_cbo"] - rt["gb_gap"]
    rev_pi = rt["infl_4q"] - rt["gb_infl_4q"]
    Xrt = np.column_stack([np.ones(len(rt)), rt["ffr_l1"], rt["gb_infl_4q"], rt["gb_gap"]])
    Xrv = np.column_stack([np.ones(len(rt)), rt["ffr_l1"], rt["infl_4q"], rt["gap_cbo"]])
    predicted = np.linalg.solve(Xrv.T @ Xrv, Xrv.T @ Xrt @ f_rt.params)
    pred_lr_pi = predicted[2] / (1 - predicted[1])
    pred_lr_y = predicted[3] / (1 - predicted[1])
    kn.update({
        "gap_revision_mean": float(rev.mean()), "gap_revision_sd": float(rev.std()),
        "gap_revision_corr_realtime": float(np.corrcoef(rev, rt["gb_gap"])[0, 1]),
        "gap_revision_corr_final": float(np.corrcoef(rev, rt["gap_cbo"])[0, 1]),
        "infl_revision_mean": float(rev_pi.mean()), "infl_revision_sd": float(rev_pi.std()),
        "gap_realtime_sd": float(rt["gb_gap"].std()), "gap_final_sd": float(rt["gap_cbo"].std()),
        "gap_rt_final_corr": float(np.corrcoef(rt["gb_gap"], rt["gap_cbo"])[0, 1]),
        "attenuation_predicted_phi_y": float(pred_lr_y), "attenuation_actual_phi_y": float(ly_rev[0]),
        "attenuation_predicted_phi_pi": float(pred_lr_pi), "attenuation_actual_phi_pi": float(lp_rev[0]),
    })
    diag.append({"test": "Greenbook gap revision (final CBO minus real time)", "statistic": "mean, SD",
                 "value": float(rev.mean()), "reference": "0 if unbiased", "note": f"SD {rev.std():.2f}"})
    diag.append({"test": "revision correlated with real-time gap (noise view predicts negative)",
                 "statistic": "correlation", "value": kn["gap_revision_corr_realtime"], "reference": "0 under pure news",
                 "note": f"corr with final gap {kn['gap_revision_corr_final']:.2f}"})
    diag.append({"test": "errors-in-variables formula: predicted long-run gap response on revised data",
                 "statistic": "phi_y", "value": float(pred_lr_y), "reference": f"actual {ly_rev[0]:.2f}",
                 "note": "plim under the hypothesis that the Fed set policy on Greenbook data"})

    # ---------------- Structural break, rolling windows, influence ----------------
    br = take(df, "full_nonzlb", ["ffr", "infl_4q", "gap_cbo"])
    br = br[br["quarter"] <= pd.Period("2007Q4", "Q")].reset_index(drop=True)
    X = np.column_stack([np.ones(len(br)), br["infl_4q"], br["gap_cbo"]])
    q = est.qlr_break(br["ffr"].to_numpy(), X, rng=rng)
    kn["qlr_sup_wald"], kn["qlr_p"], kn["qlr_crit5"] = q["sup_wald"], q["p_value"], q["crit_5pct"]
    kn["qlr_break_quarter"] = str(br["quarter"].iloc[q["break_index"]])
    diag.append({"test": "Quandt-Andrews sup-Wald break in all rule coefficients, 1965-2007", "statistic": "sup-Wald",
                 "value": q["sup_wald"], "reference": f"5% critical {q['crit_5pct']:.2f} (simulated)",
                 "note": f"break at {kn['qlr_break_quarter']}, p = {q['p_value']:.4f}"})

    nz = take(df, "full_nonzlb", ["ffr", "ffr_l1", "infl_4q", "gap_cbo"])
    roll = []
    for end in nz["quarter"]:
        start = end - 39
        d = nz[(nz["quarter"] >= start) & (nz["quarter"] <= end)]
        if len(d) < 30:
            continue
        f = static_ols(d, "infl_4q", "gap_cbo")
        b, s = f.get("pi")
        by, sy = f.get("y")
        roll.append({"end_quarter": str(end), "N": len(d), "phi_pi": b, "ci_lo": b - 1.96 * s, "ci_hi": b + 1.96 * s,
                     "phi_y": by, "chair_at_end": history.chair_for(end)})
    roll = pd.DataFrame(roll)
    roll.to_csv(RESULTS / "rolling_rule.csv", index=False)
    kn["rolling_min_phi_pi"] = roll["phi_pi"].min()
    kn["rolling_min_end"] = roll.loc[roll["phi_pi"].idxmin(), "end_quarter"]
    kn["rolling_max_phi_pi"] = roll["phi_pi"].max()
    kn["rolling_max_end"] = roll.loc[roll["phi_pi"].idxmax(), "end_quarter"]

    loo = []
    for chair in nz.assign(chair=nz["quarter"].apply(history.chair_for))["chair"].unique():
        d = nz[nz["quarter"].apply(history.chair_for) != chair]
        f, lp, ly = smooth_ols(d, "infl_4q", "gap_cbo")
        loo.append({"dropped": chair, "N": len(d), "phi_pi": lp[0], "se": lp[1]})
    loo = pd.DataFrame(loo)
    for _, r in loo.iterrows():
        diag.append({"test": f"leave-one-chair-out, drop {r['dropped']}", "statistic": "long-run phi_pi",
                     "value": r["phi_pi"], "reference": "full non-ZLB sample", "note": f"N {r['N']}, SE {r['se']:.2f}"})
    f_all, lp_all, _ = smooth_ols(nz, "infl_4q", "gap_cbo")
    kn["full_nonzlb_phi_pi"], kn["full_nonzlb_se_pi"] = lp_all
    kn["loo_chair_min"], kn["loo_chair_max"] = loo["phi_pi"].min(), loo["phi_pi"].max()
    kn["loo_chair_min_dropped"] = loo.loc[loo["phi_pi"].idxmin(), "dropped"]
    kn["loo_chair_max_dropped"] = loo.loc[loo["phi_pi"].idxmax(), "dropped"]

    jk = []
    for yr in sorted({qq.year for qq in gb["quarter"]}):
        d = gb[[qq.year != yr for qq in gb["quarter"]]]
        jk.append(smooth_ols(d, "infl_4q", "gap_cbo")[1][0])
    kn["jackknife_year_min"], kn["jackknife_year_max"] = float(np.min(jk)), float(np.max(jk))
    diag.append({"test": "leave-one-year-out jackknife, Greenspan-Bernanke partial adjustment",
                 "statistic": "range of long-run phi_pi", "value": float(np.max(jk) - np.min(jk)),
                 "reference": f"[{np.min(jk):.2f}, {np.max(jk):.2f}]", "note": f"{len(jk)} years"})

    pd.DataFrame(BENCHMARKS, columns=["source", "location", "quantity", "value", "se"]).to_csv(
        RESULTS / "original_benchmarks.csv", index=False)

    full = df[history.sample_mask(df, "full")]
    moments = []
    for col, label in [("ffr", "federal funds rate"), ("infl_4q", "deflator inflation, 4 quarters"),
                       ("infl_q_ann", "deflator inflation, quarterly annualized"), ("gap_cbo", "CBO output gap"),
                       ("cpi_infl_4q", "CPI inflation, 4 quarters"), ("unemp_gap", "unemployment gap"),
                       ("gb_infl_4q", "Greenbook inflation, real time"), ("gb_gap", "Greenbook gap, real time"),
                       ("oil_surprise", "OPEC surprise, quarterly sum"), ("oil_news_shock", "oil news shock, quarterly sum")]:
        x = full[col].dropna()
        q = x.quantile([0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]).to_numpy()
        moments.append({"variable": col, "label": label, "N": len(x), "mean": x.mean(), "sd": x.std(),
                        "skewness": stats.skew(x), "kurtosis": stats.kurtosis(x, fisher=False),
                        "p5": q[0], "p10": q[1], "p25": q[2], "p50": q[3], "p75": q[4], "p90": q[5], "p95": q[6]})
    pd.DataFrame(moments).to_csv(RESULTS / "summary_stats.csv", index=False)
    hr = pd.DataFrame(rows)
    testable = hr["p_pi_le_1"].notna()
    hr.loc[testable, "p_pi_le_1_holm"] = est.holm(hr.loc[testable, "p_pi_le_1"].to_numpy())
    hr.to_csv(RESULTS / "regression_horserace.csv", index=False)
    kn["holm_n_tests"] = int(testable.sum())
    kn["holm_n_reject_5pct"] = int((hr["p_pi_le_1_holm"] < 0.05).sum())

    # ---------------- Rule paths, chair-era deviations, named quarters ----------------
    paths = df[["quarter", "chair", "episode", "zlb", "ffr", "infl_4q", "gap_cbo", "gb_infl_4q", "gb_gap",
                "rt92_infl_4q", "rt92_gap"]].copy()
    paths = paths[paths["quarter"] >= pd.Period("1965Q1", "Q")]
    paths["rule_revised"] = taylor_rule(paths["infl_4q"], paths["gap_cbo"])
    paths["rule_realtime"] = taylor_rule(paths["gb_infl_4q"], paths["gb_gap"])
    paths["rule_vintage_1992"] = taylor_rule(paths["rt92_infl_4q"], paths["rt92_gap"])
    paths["dev_revised"] = paths["ffr"] - paths["rule_revised"]
    paths["dev_realtime"] = paths["ffr"] - paths["rule_realtime"]
    paths["neutral_term"] = TAYLOR["const"] + TAYLOR["phi_pi"] * 2.0
    paths["inflation_term"] = TAYLOR["phi_pi"] * (paths["infl_4q"] - 2.0)
    paths["gap_term"] = TAYLOR["phi_y"] * paths["gap_cbo"]
    out = paths.copy()
    out["quarter"] = out["quarter"].astype(str)
    out.to_csv(RESULTS / "rule_paths.csv", index=False)

    pv = paths.dropna(subset=["dev_revised"])
    chairs = []
    for chair in [c for c, _ in history.CHAIRS if c in set(pv["chair"])]:
        d = pv[pv["chair"] == chair]
        chairs.append({"chair": chair, "first": str(d["quarter"].iloc[0]), "last": str(d["quarter"].iloc[-1]),
                       "N": len(d), "mean_inflation": d["infl_4q"].mean(), "mean_gap": d["gap_cbo"].mean(),
                       "mean_ffr": d["ffr"].mean(), "mean_rule": d["rule_revised"].mean(),
                       "mean_deviation": d["dev_revised"].mean(), "rmse_deviation": np.sqrt((d["dev_revised"] ** 2).mean()),
                       "share_below_rule": (d["dev_revised"] < 0).mean(), "zlb_quarters": int(d["zlb"].sum())})
    chairs = pd.DataFrame(chairs)
    chairs.to_csv(RESULTS / "chair_deviations.csv", index=False)
    for _, r in chairs.iterrows():
        kn[f"chair_{r['chair'].lower()}_mean_dev"] = r["mean_deviation"]
        kn[f"chair_{r['chair'].lower()}_rmse"] = r["rmse_deviation"]
        kn[f"chair_{r['chair'].lower()}_mean_infl"] = r["mean_inflation"]
        kn[f"chair_{r['chair'].lower()}_share_below"] = r["share_below_rule"]

    for qs in ["1974Q4", "1980Q2", "1987Q4", "1992Q3", "2003Q3", "2004Q2", "2008Q4", "2021Q4", "2022Q2", "2022Q3",
               str(pv["quarter"].iloc[-1])]:
        r = pv[pv["quarter"] == pd.Period(qs, "Q")]
        if len(r):
            r = r.iloc[0]
            for col in ["ffr", "infl_4q", "gap_cbo", "rule_revised", "dev_revised", "gb_infl_4q", "gb_gap",
                        "rule_realtime", "dev_realtime", "inflation_term", "gap_term"]:
                if pd.notna(r[col]):
                    kn[f"q{qs}_{col}"] = r[col]
    kn["latest_quarter"] = str(pv["quarter"].iloc[-1])

    named = []
    nonzlb = pv[pv["zlb"] == 0]
    for kind, d in [("most below the rule (looser)", nonzlb.nsmallest(8, "dev_revised")),
                    ("most above the rule (tighter)", nonzlb.nlargest(8, "dev_revised"))]:
        for _, r in d.iterrows():
            named.append({"table": "Taylor (1993) rule deviation, revised data", "kind": kind, "quarter": str(r["quarter"]),
                          "chair": r["chair"], "episode": r["episode"], "ffr": r["ffr"], "rule": r["rule_revised"],
                          "deviation": r["dev_revised"], "neutral_term": r["neutral_term"],
                          "inflation_term": r["inflation_term"], "gap_term": r["gap_term"],
                          "inflation": r["infl_4q"], "gap": r["gap_cbo"]})
    contrib = pd.DataFrame({
        "quarter": gb["quarter"], "resid": gb_fit.resid,
        "const": gb_fit.params[0], "smoothing_term": gb_fit.params[1] * gb["ffr_l1"],
        "inflation_term": gb_fit.params[2] * gb["infl_4q"], "gap_term": gb_fit.params[3] * gb["gap_cbo"]})
    contrib["abs"] = contrib["resid"].abs()
    for _, r in contrib.nlargest(8, "abs").iterrows():
        qq = r["quarter"]
        named.append({"table": "largest residuals, estimated partial-adjustment rule 1987Q3-2007Q4",
                      "kind": "surprise cut" if r["resid"] < 0 else "surprise hike", "quarter": str(qq),
                      "chair": history.chair_for(qq), "episode": history.EPISODES.get(str(qq), ""),
                      "ffr": float(gb.loc[gb["quarter"] == qq, "ffr"].iloc[0]), "rule": np.nan, "deviation": r["resid"],
                      "neutral_term": r["const"], "inflation_term": r["inflation_term"], "gap_term": r["gap_term"],
                      "smoothing_term": r["smoothing_term"],
                      "inflation": float(gb.loc[gb["quarter"] == qq, "infl_4q"].iloc[0]),
                      "gap": float(gb.loc[gb["quarter"] == qq, "gap_cbo"].iloc[0])})
    pd.DataFrame(named).to_csv(RESULTS / "named_entities_table.csv", index=False)

    dec = nonzlb["dev_revised"].quantile([0.1, 0.5, 0.9])
    for p, v in dec.items():
        near = nonzlb.iloc[(nonzlb["dev_revised"] - v).abs().argsort().iloc[0]]
        kn[f"dev_p{int(p * 100)}"] = v
        kn[f"dev_p{int(p * 100)}_quarter"] = str(near["quarter"])
        kn[f"dev_p{int(p * 100)}_chair"] = near["chair"]
        kn[f"dev_p{int(p * 100)}_value"] = near["dev_revised"]

    pd.DataFrame(diag).to_csv(RESULTS / "diagnostics_and_sensitivity.csv", index=False)
    numbers.update("econometrics", kn)
    print(hr[["spec", "N", "phi_pi", "se_pi", "phi_y", "rho", "F_eff", "J_p", "note"]].round(3).to_string())
    print(pd.DataFrame(diag)[["test", "value", "reference", "note"]].to_string())
    print(chairs.round(2).to_string())


if __name__ == "__main__":
    main()
