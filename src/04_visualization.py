"""Publication figures (300 DPI) from the results CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import history, numbers, style  # noqa: E402
from core.io import RESULTS  # noqa: E402

DPI = 300


def qx(q: pd.Series) -> np.ndarray:
    """Quarter strings to decimal years at the quarter midpoint."""
    p = pd.PeriodIndex(q, freq="Q")
    return p.year + (p.quarter - 0.5) / 4.0


def shade_chairs(ax: plt.Axes, lo: float, hi: float, labels: bool = True, y: float | None = None) -> None:
    starts = [(n, pd.Timestamp(d)) for n, d in history.CHAIRS]
    for k, (name, start) in enumerate(starts):
        x0 = start.year + (start.day_of_year - 1) / 365.25
        x1 = (starts[k + 1][1].year + (starts[k + 1][1].day_of_year - 1) / 365.25) if k + 1 < len(starts) else hi
        a, b = max(x0, lo), min(x1, hi)
        if b <= a:
            continue
        if k % 2 == 0:
            ax.axvspan(a, b, color=style.SHADE, lw=0, zorder=0)
        if labels and b - a > 2.5:
            ax.text((a + b) / 2, y if y is not None else ax.get_ylim()[1], name, ha="center", va="top",
                    fontsize=7, color=style.MUTED)


def fig1(kn: dict) -> None:
    paths = pd.read_csv(RESULTS / "rule_paths.csv")
    paths["t"] = qx(paths["quarter"])
    e = kn["econometrics"]
    fig = plt.figure(figsize=(10, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.22, top=0.86, bottom=0.1,
                          left=0.06, right=0.98)

    ax = fig.add_subplot(gs[0, :])
    p = paths[paths["t"] >= 1965]
    ax.plot(p["t"], p["rule_revised"], color=style.ORANGE, lw=1.4, label="Taylor (1993) rule, revised data")
    ax.plot(p["t"], p["ffr"], color=style.BLUE, lw=1.8, label="Federal funds rate")
    ax.axhline(0, color=style.MUTED, lw=0.6)
    ax.set_ylim(-6, 22)
    ax.set_xlim(1965, p["t"].max() + 0.5)
    shade_chairs(ax, 1965, p["t"].max() + 0.5, y=21.5)
    for q, txt, dx, dy in [("1974Q4", f"1974Q4: rate {-e['q1974Q4_dev_revised']:.1f} points\nbelow the rule", -8.5, 2.3),
                           ("1981Q2", "Volcker, 1981", 3.5, -0.8),
                           ("2003Q3", "2003: funds rate at 1%", -4, -5.5),
                           ("2022Q2", f"2022Q2: {e['q2022Q2_ffr']:.2f}%\nvs rule {e['q2022Q2_rule_revised']:.1f}%", -9, 3)]:
        r = p[p["quarter"] == q]
        if len(r):
            y = float(r["rule_revised"].iloc[0]) if q in ("1974Q4", "2022Q2") else float(r["ffr"].iloc[0])
            style.label(ax, float(r["t"].iloc[0]), y, txt, dx, dy)
    ax.set_ylabel("percent")
    ax.set_title("A. Sixty years of the funds rate against Taylor's prescription")
    ax.legend(loc="lower left", ncol=2)

    ax = fig.add_subplot(gs[1, 0])
    w = paths[(paths["quarter"] >= "1987Q1") & (paths["quarter"] <= "1992Q3")]
    ax.plot(w["t"], w["ffr"], color=style.BLUE, lw=1.8, label="funds rate")
    ax.plot(w["t"], w["rule_vintage_1992"], color=style.ORANGE, lw=1.6, label="rule, data as of 1992")
    ax.plot(w["t"], w["rule_revised"], color=style.AQUA, lw=1.4, ls=(0, (4, 2)), label="rule, today's revised data")
    ax.legend(loc="upper right")
    ax.set_ylim(2.5, 11)
    ax.set_ylabel("percent")
    ax.set_title("B. Taylor's Figure 1, rebuilt from the 1992 vintage")
    ax.text(0.02, 0.04, f"RMSE {e['fit_vintage_1992_rmse']:.2f} pp on 1992 data\n"
                        f"RMSE {e['fit_revised_cbo_rmse']:.2f} pp on revised data (CBO gap)",
            transform=ax.transAxes, fontsize=7.5, color=style.INK2)

    ax = fig.add_subplot(gs[1, 1])
    s = paths.dropna(subset=["rule_revised"])
    s = s[(s["t"] >= 1965) & (s["zlb"] == 0)]
    groups = [("1965-1979", s["t"] < 1979.6, style.ORANGE), ("1979-2007", (s["t"] >= 1979.6) & (s["t"] < 2008), style.BLUE),
              ("2008 onward", s["t"] >= 2008, style.AQUA)]
    for lab, m, c in groups:
        ax.scatter(s.loc[m, "rule_revised"], s.loc[m, "ffr"], s=10, color=c, alpha=0.85, lw=0.3,
                   edgecolor="white", label=lab)
    lim = (-4, 20)
    ax.plot(lim, lim, color=style.MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(15.5, 13.2, "45°: rate equals rule", fontsize=7, color=style.MUTED, rotation=38)
    ax.set_xlim(lim)
    ax.set_ylim(-1, 20)
    ax.set_xlabel("Taylor (1993) prescription, percent")
    ax.set_ylabel("actual funds rate, percent")
    ax.set_title("C. Rate against prescription, by era")
    ax.legend(loc="upper left")
    style.headline(fig, "The Taylor rule fit its own era and almost no other",
                   "Quarterly federal funds rate and the rule r = 1 + 1.5π + 0.5y, with π the four-quarter GDP deflator "
                   f"inflation and y the output gap; N = {len(p)} quarters, 1965 to {kn['econometrics']['latest_quarter']}",
                   "Sources: FRED (FEDFUNDS, GDPDEF, GDPC1, GDPPOT); Philadelphia Fed Real-Time Data Set, 1992Q4 vintage. "
                   "Panel B uses Taylor's 2.2% trend for the 1992 vintage; revised series use the CBO gap. Shading marks Fed chair tenures. "
                   f"Panel C omits the {int(paths['zlb'].sum())} zero-lower-bound quarters.")
    fig.savefig(RESULTS / "fig1_primary_mechanism.png", dpi=DPI)
    plt.close(fig)


def fig2(kn: dict) -> None:
    hr = pd.read_csv(RESULTS / "regression_horserace.csv")
    stage_color = {"Original": style.INK, "Stage 1": style.BLUE, "Stage 2": style.BLUE, "Stage 3": style.ORANGE}
    fig, ax = plt.subplots(figsize=(10, 7.4))
    fig.subplots_adjust(left=0.42, right=0.97, top=0.86, bottom=0.12)
    lo, hi = -2.0, 6.5
    y = np.arange(len(hr))[::-1]
    for yy, (_, r) in zip(y, hr.iterrows()):
        c = stage_color[r["stage"]]
        b = r["phi_pi"]
        if np.isfinite(r["se_pi"]):
            a, z = r["ci_lo"], r["ci_hi"]
            ax.plot([max(a, lo), min(z, hi)], [yy, yy], color=c, lw=1.6, solid_capstyle="round")
            for edge, cond in [(lo, a < lo), (hi, z > hi)]:
                if cond:
                    ax.annotate("", xy=(edge, yy), xytext=(edge - 0.25 * np.sign(edge), yy),
                                arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
        marker = "D" if r["stage"] == "Original" else "o"
        if lo <= b <= hi:
            ax.plot(b, yy, marker, color=c, ms=6, mec="white", mew=0.8, zorder=3)
    def tick(r: pd.Series) -> str:
        extra = f", F {r['F_eff']:.1f}" if np.isfinite(r["F_eff"]) else ""
        return f"{r['spec']}  {r['label']}  (N {int(r['N'])}{extra})"
    ax.set_yticks(y)
    ax.set_yticklabels([tick(r) for _, r in hr.iterrows()], fontsize=7.5)
    ax.axvline(1.0, color=style.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(1.5, color=style.MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.text(0.95, len(hr) - 0.35, "Taylor principle, φπ = 1", ha="right", va="bottom", fontsize=7, color=style.INK2)
    ax.text(1.55, len(hr) - 0.35, "Taylor's 1.5", ha="left", va="bottom", fontsize=7, color=style.MUTED)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.8, len(hr) + 0.6)
    ax.set_xlabel("long-run response of the funds rate to inflation, φπ (95% CI, HAC)")
    ax.grid(axis="y", visible=False)
    legend = [Line2D([], [], color=style.INK, marker="D", ls="", label="original rule (calibrated)"),
              Line2D([], [], color=style.BLUE, marker="o", label="replication and hardening (Stages 1-2)"),
              Line2D([], [], color=style.ORANGE, marker="o", label="external-instrument critic (Stage 3)")]
    ax.legend(handles=legend, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    e = kn["econometrics"]
    style.headline(fig, "The estimated inflation response depends on the era, the data vintage, and the identification",
                   f"Estimated inflation coefficient φπ in {len(hr) - 1} specifications that vary sample, data vintage, "
                   "timing, and estimator, against Taylor's calibrated 1.5",
                   "Forward-looking GMM rows follow Clarida, Gali and Gertler (CGG). Arrows mark intervals that run past the axis. "
                   f"F is the Montiel Olea-Pflueger effective first-stage F; {e['mop_crit']:.1f} is the "
                   "threshold for a 5% test of 10% worst-case bias. IV rows fix the gap response at 0.5. Weak-IV-robust Anderson-Rubin "
                   "95% sets: " + "; ".join(
                       f"{s.upper()} " + (f"[{e[s + '_ar_lo']:.2f}, {e[s + '_ar_hi']:.2f}]" if e[s + "_ar_bounded"] else "unbounded")
                       for s in ("s15", "s16", "s17")) + ".")
    fig.savefig(RESULTS / "fig2_methodological_horserace.png", dpi=DPI)
    plt.close(fig)


def fig3(kn: dict) -> None:
    cf = pd.read_csv(RESULTS / "counterfactual_simulation.csv")
    cf["t"] = qx(cf["quarter"])
    cf = cf[cf["t"] >= 2019]
    base = cf[cf["scenario"] == "Taylor (1993) rule, r* = 2"]
    scen = [("Taylor (1993) rule, r* = 2", style.ORANGE, "-", "Taylor rule, r* = 2"),
            ("Taylor (1993) rule, r* = 0.5", style.ORANGE, (0, (3, 2)), "Taylor rule, r* = 0.5"),
            ("estimated 1987-2007 rule, no deviations", style.AQUA, "-", "Fed's own 1987-2007 rule")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.78, bottom=0.2, wspace=0.25)
    for ax, (act, cfc, ttl, unit) in zip(axes, [("actual_ffr", "cf_ffr", "A. Federal funds rate", "percent"),
                                                ("actual_infl_4q", "cf_infl_4q", "B. Inflation, GDP deflator, 4 quarters", "percent"),
                                                ("actual_gap", "cf_gap", "C. Output gap (CBO)", "percent of potential")]):
        ax.plot(base["t"], base[act], color=style.BLUE, lw=2.0, label="actual")
        for sname, c, ls, lab in scen:
            d = cf[(cf["scenario"] == sname) & cf["in_counterfactual"]]
            ax.plot(d["t"], d[cfc], color=c, ls=ls, lw=1.5, label=lab)
        ax.axvline(2021.0, color=style.MUTED, lw=0.6)
        ax.axhline(0, color=style.MUTED, lw=0.5)
        ax.set_title(ttl)
        ax.set_ylabel(unit)
    axes[0].text(2021.1, axes[0].get_ylim()[1] * 0.95, "counterfactual\nbegins 2021Q1", fontsize=7, color=style.MUTED, va="top")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=4, fontsize=7.5)
    s = kn["structural"]
    style.headline(fig, "Under Taylor's rule the 2021-2023 inflation peaks lower, at the price of a weaker economy",
                   f"New Keynesian model estimated on 1987Q3-2007Q4 (N = {s['nk_N']}); demand and cost-push shocks backed out of the data, "
                   "then replayed with no policy deviations",
                   "Model inflation is quarterly; panel B shows four-quarter averages. r* is the neutral real rate in the rule's intercept. "
                   f"Quarters at the zero lower bound: Taylor rule r* = 2, {s['cf_taylor_zlb_quarters']}; r* = 0.5, "
                   f"{s['cf_taylor_lowr_zlb_quarters']}; Fed's own rule, {s['cf_own_zlb_quarters']}.")
    fig.savefig(RESULTS / "fig3_structural_or_counterfactual.png", dpi=DPI)
    plt.close(fig)


def fig4(kn: dict) -> None:
    roll = pd.read_csv(RESULTS / "rolling_rule.csv")
    roll["t"] = qx(roll["end_quarter"])
    surf = pd.read_csv(RESULTS / "identification_surface.csv")
    prof = pd.read_csv(RESULTS / "profile_phi_pi.csv")
    s = kn["structural"]
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.2, 0.95], wspace=0.42, top=0.78, bottom=0.16, left=0.05, right=0.98)

    ax = fig.add_subplot(gs[0])
    gap = roll["t"].diff() > 1.0
    seg = gap.cumsum()
    for _, d in roll.groupby(seg):
        ax.fill_between(d["t"], d["ci_lo"], d["ci_hi"], color=style.BLUE, alpha=0.15, lw=0)
        ax.plot(d["t"], d["phi_pi"], color=style.BLUE, lw=1.6)
    ax.axhline(1.0, color=style.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.axhline(1.5, color=style.MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.set_ylim(-2, 5)
    ax.set_xlim(roll["t"].min() - 0.5, roll["t"].max() + 0.5)
    shade_chairs(ax, roll["t"].min() - 0.5, roll["t"].max() + 0.5, y=4.9)
    ax.set_ylabel("φπ, static rule (95% CI)")
    ax.set_title("A. Rolling 40-quarter inflation response")
    ax.text(roll["t"].min() + 0.5, 0.72, "φπ = 1", fontsize=7, color=style.INK2)

    ax = fig.add_subplot(gs[1])
    piv = surf.pivot(index="sd_v", columns="rho_v", values="plim_smoothing")
    cs = ax.contourf(piv.columns, piv.index, piv.values, levels=np.linspace(np.nanmin(piv.values), np.nanmax(piv.values), 12),
                     cmap="Blues_r")
    cl = ax.contour(piv.columns, piv.index, piv.values, levels=[s["nk_phi_pi"]], colors=[style.ORANGE], linewidths=1.6)
    ax.clabel(cl, fmt={s["nk_phi_pi"]: f"true {s['nk_phi_pi']:.2f}"}, fontsize=7)
    ax.plot(s["nk_rho_v"], s["nk_sd_v"], "o", color=style.ORANGE, mec="white", ms=7)
    style.label(ax, s["nk_rho_v"], s["nk_sd_v"], "estimate", -0.3, 0.45)
    cb = fig.colorbar(cs, ax=ax, fraction=0.05, pad=0.02, format="%.1f")
    cb.set_label("plim of OLS φπ", fontsize=7.5, color=style.INK2)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("persistence of policy shocks, ρv")
    ax.set_ylabel("size of policy shocks, σv")
    ax.set_title("B. What OLS recovers when φπ is fixed")
    ax.grid(False)

    ax = fig.add_subplot(gs[2])
    d = prof[prof["determinate"]]
    ax.plot(d["phi_pi"], d["delta_objective"], color=style.BLUE, marker="o", ms=4)
    ax.axvline(s["nk_phi_pi"], color=style.ORANGE, lw=1.0)
    ax.set_xlabel("φπ held fixed")
    ax.set_ylabel("rise in objective")
    ax.set_title("C. Profile in φπ")
    style.headline(fig, "The inflation response moved with the chair, and the data identify it only loosely",
                   "Heterogeneity over time (A) and identification diagnostics from the estimated New Keynesian model (B, C)",
                   "A: static rule by OLS with HAC intervals, windows of 40 quarters with at least 30 non-ZLB observations. "
                   "B: analytic plim of partial-adjustment OLS as policy-shock persistence and size vary, other parameters at estimates. "
                   "C: profile of the objective with φπ fixed and every other parameter re-estimated.")
    fig.savefig(RESULTS / "fig4_heterogeneity_and_sensitivity.png", dpi=DPI)
    plt.close(fig)


def fig5(kn: dict) -> None:
    paths = pd.read_csv(RESULTS / "rule_paths.csv")
    paths["t"] = qx(paths["quarter"])
    d = paths.dropna(subset=["gb_gap", "gb_infl_4q"])
    e = kn["econometrics"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.78, bottom=0.24, wspace=0.2)
    ax = axes[0]
    ax.plot(d["t"], d["gb_gap"], color=style.ORANGE, lw=1.6, label="Greenbook, in real time")
    ax.plot(d["t"], d["gap_cbo"], color=style.BLUE, lw=1.6, label="CBO, as revised today")
    ax.axhline(0, color=style.MUTED, lw=0.5)
    ax.set_title("A. The output gap the Fed saw versus the one we see now")
    ax.set_ylabel("percent of potential output")
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.plot(d["t"], d["dev_realtime"], color=style.ORANGE, lw=1.6, label="real-time data")
    ax.plot(d["t"], d["dev_revised"], color=style.BLUE, lw=1.6, label="revised data")
    ax.axhline(0, color=style.MUTED, lw=0.5)
    r = d[d["quarter"] == "2003Q3"]
    if len(r):
        style.label(ax, float(r["t"].iloc[0]), float(r["dev_revised"].iloc[0]),
                    f"2003Q3: {r['dev_revised'].iloc[0]:.1f} on revised data,\n{r['dev_realtime'].iloc[0]:.1f} in real time",
                    -8.3, -0.85)
    ax.set_title("B. Funds rate minus Taylor's prescription")
    ax.set_ylabel("percentage points")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.08), ncol=2)
    style.headline(fig, "Data revisions rewrite the verdict on individual episodes",
                   f"Federal Reserve staff estimates published with a five-year lag; {e['realtime_N']} quarters with both measures, "
                   f"mean gap revision {e['gap_revision_mean']:.2f} pp (SD {e['gap_revision_sd']:.2f})",
                   "Sources: Philadelphia Fed Tealbook (Greenbook) output gap and deflator forecasts, one vintage per quarter "
                   "closest to mid-quarter; FRED GDPC1, GDPPOT, GDPDEF.")
    fig.savefig(RESULTS / "fig5_realtime_versus_revised.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    style.apply()
    kn = numbers.load()
    for f in (fig1, fig2, fig3, fig4, fig5):
        f(kn)
    print("figures written")


if __name__ == "__main__":
    main()
