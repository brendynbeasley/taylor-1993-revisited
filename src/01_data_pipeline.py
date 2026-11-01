"""Download every source, verify it, and build the quarterly analytical panel.

Usage:  python src/01_data_pipeline.py [--refresh]

Sources (all public, no API keys):
  FRED                 policy rate, real GDP, deflator, CBO potential, core PCE, CPI, unemployment
  Philadelphia Fed     Real-Time Data Set, 1992Q4 vintage of real GDP and the deflator
  Philadelphia Fed     Greenbook/Tealbook real-time output gap and deflator forecasts
  Kanzig (2021)        OPEC announcement oil supply surprises and news shocks
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import history, numbers  # noqa: E402
from core.io import PANEL, RAW, RESULTS, fetch, load_manifest, record, save_manifest  # noqa: E402

FRED_IDS = ["FEDFUNDS", "GDPC1", "GDPDEF", "GDPPOT", "PCEPILFE", "CPIAUCSL", "UNRATE", "NROU"]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
PHIL = "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/real-time-data/data-files/xlsx/{}"
GB = "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/greenbook-data/{}"
KANZIG = "https://raw.githubusercontent.com/dkaenzig/oilsupplynews/master/oilSupplyNewsShocks_2025M12.xlsx"

TREND_GROWTH = 0.022  # Taylor (1993, p. 202): trend real GDP grows 2.2 percent per year
TREND_WINDOW = ("1984Q1", "1992Q3")  # Taylor (1993, p. 202): window over which the trend is drawn
VINTAGE = "92Q4"  # first vintage containing 1992Q3, the last quarter in Taylor's Figure 1


def to_quarter_mean(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.to_period("Q")).mean()


def read_fred(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.dropna()


def taylor_gap(log_y: pd.Series) -> pd.Series:
    """Percent gap from a 2.2 percent log-linear trend fitted over 1984Q1-1992Q3.

    Taylor fixes the slope; the level is not stated, so the intercept is set so the
    gap averages zero over his trend window (the least-squares intercept given the slope).
    """
    t = np.arange(len(log_y), dtype=float)
    slope = np.log1p(TREND_GROWTH) / 4.0
    win = (log_y.index >= pd.Period(TREND_WINDOW[0], "Q")) & (log_y.index <= pd.Period(TREND_WINDOW[1], "Q"))
    intercept = np.mean(log_y.values[win] - slope * t[win])
    trend = intercept + slope * t
    return pd.Series(100.0 * (np.exp(log_y.values - trend) - 1.0), index=log_y.index)


def rtdsm_vintage(path: Path, prefix: str, vintage: str) -> pd.Series:
    df = pd.read_excel(path)
    col = f"{prefix}{vintage}"
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = pd.PeriodIndex([d.replace(":", "") for d in df["DATE"].astype(str)], freq="Q")
    return s.dropna()


def mid_quarter_pick(dates: pd.Series) -> pd.Series:
    """For each quarter, the index of the vintage date closest to the quarter's midpoint."""
    q = dates.dt.to_period("Q")
    mid = q.dt.start_time + (q.dt.end_time - q.dt.start_time) / 2
    dist = (dates - mid).abs()
    frame = pd.DataFrame({"q": q, "dist": dist})
    return frame.groupby("q")["dist"].idxmin()


def greenbook_gap(prev_path: Path, new_path: Path) -> tuple[pd.DataFrame, float]:
    """Real-time Greenbook output gap for the current quarter, one vintage per quarter.

    The previous-format file (meetings 1987-2013) gives T-relative columns; the newer file
    (vintages 1996-2020) gives full histories. Overlap agreement is returned as a check.
    """
    raw = pd.read_excel(prev_path, sheet_name="Output Gap", header=None)
    body = raw.iloc[3:].copy()
    body = body[body[1].notna()]
    prev = pd.DataFrame({
        "date": pd.to_datetime(body[1]),
        "gb_gap": pd.to_numeric(body[11], errors="coerce"),
    }).dropna().reset_index(drop=True)
    pick = mid_quarter_pick(prev["date"])
    prev_q = pd.Series(prev.loc[pick.values, "gb_gap"].values, index=pick.index)

    new = pd.read_excel(new_path)
    new_index = pd.PeriodIndex([str(x).replace(":0", "Q").replace(":", "Q") for x in new.iloc[:, 0]], freq="Q")
    vint = [c for c in new.columns if str(c).startswith("GBgap_")]
    vdates = pd.Series(pd.to_datetime([c.split("_")[1] for c in vint], format="%y%m%d"), index=vint)
    pick_new = mid_quarter_pick(vdates.reset_index(drop=True))
    rows = {}
    for q, i in pick_new.items():
        col = vint[i]
        series = pd.Series(pd.to_numeric(new[col], errors="coerce").values, index=new_index)
        if q in series.index and pd.notna(series[q]):
            rows[q] = series[q]
    new_q = pd.Series(rows)

    overlap = prev_q.index.intersection(new_q.index)
    corr = float(np.corrcoef(prev_q[overlap], new_q[overlap])[0, 1])
    combined = pd.concat([prev_q, new_q[new_q.index > prev_q.index.max()]]).sort_index()
    return combined.rename("gb_gap").to_frame(), corr


def greenbook_inflation(zip_path: Path, out_dir: Path) -> pd.DataFrame:
    """Real-time Greenbook deflator inflation: four quarters ending now and one year ahead."""
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    frames = []
    for f in sorted(out_dir.glob("gPGDP_*.xlsx")):
        df = pd.read_excel(f)
        years = df.iloc[:, 0].astype(float)
        idx = pd.PeriodIndex([f"{int(y)}Q{int(round((y - int(y)) * 10))}" for y in years], freq="Q")
        for c in df.columns[1:]:
            frames.append((pd.to_datetime(c.split("_")[1], format="%Y%m%d"),
                           pd.Series(pd.to_numeric(df[c], errors="coerce").values, index=idx)))
    def window(s: pd.Series, q: pd.Period, offsets: range) -> list[float] | None:
        pts = [q + k for k in offsets]
        if all(p in s.index and pd.notna(s[p]) for p in pts):
            return [float(s[p]) for p in pts]
        return None

    # Keep only vintages that report the four quarters ending in their own quarter,
    # then take the one closest to mid-quarter.
    usable = [(d, s) for d, s in frames if window(s, pd.Period(d, "Q"), range(-3, 1)) is not None]
    pick = mid_quarter_pick(pd.Series([d for d, _ in usable]))
    out = {}
    for q, i in pick.items():
        s = usable[i][1]
        back = window(s, q, range(-3, 1))
        fwd = window(s, q, range(1, 5))
        out[q] = {"gb_infl_4q": float(np.mean(back)), "gb_infl_f4": float(np.mean(fwd)) if fwd else np.nan}
    return pd.DataFrame.from_dict(out, orient="index")


def kanzig_quarterly(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Monthly")
    df.columns = ["date", "oil_surprise", "oil_news_shock"]
    df["quarter"] = pd.PeriodIndex(pd.to_datetime(df["date"].str.replace("M", "-")), freq="Q")
    return df.groupby("quarter")[["oil_surprise", "oil_news_shock"]].sum()


def main(refresh: bool) -> None:
    manifest = load_manifest()
    waterfall: list[dict] = []

    fred: dict[str, pd.Series] = {}
    for sid in FRED_IDS:
        path = RAW / "fred" / f"{sid}.csv"
        url = FRED_URL.format(sid)
        dl = fetch(url, path, refresh=refresh)
        fred[sid] = read_fred(path)
        record(manifest, f"fred_{sid}", url, path, len(fred[sid]), dl)
        waterfall.append({"step": f"raw FRED {sid}", "observations": len(fred[sid]),
                          "first": str(fred[sid].index.min().date()), "last": str(fred[sid].index.max().date())})

    rt = {}
    for prefix, fname in [("ROUTPUT", "routputqvqd.xlsx"), ("P", "pqvqd.xlsx")]:
        path = RAW / "philly_rtdsm" / fname
        url = PHIL.format(fname)
        dl = fetch(url, path, refresh=refresh)
        rt[prefix] = rtdsm_vintage(path, prefix, VINTAGE)
        record(manifest, f"rtdsm_{prefix}", url, path, len(rt[prefix]), dl)
        waterfall.append({"step": f"raw RTDSM {prefix} vintage {VINTAGE}", "observations": len(rt[prefix]),
                          "first": str(rt[prefix].index.min()), "last": str(rt[prefix].index.max())})

    gb_files = {
        "gap_prev": "Greenbook_Output_Gap_Web_Previous_Format.xlsx",
        "gap_new": "Greenbook_Output_Gap_DH_Web.xlsx",
        "pgdp": "GBweb/Gbweb_gPGDP_Column_Format.zip",
    }
    gb_paths = {}
    for key, fname in gb_files.items():
        path = RAW / "greenbook" / Path(fname).name
        url = GB.format(fname)
        dl = fetch(url, path, refresh=refresh)
        gb_paths[key] = path
        record(manifest, f"greenbook_{key}", url, path, 0, dl)
    gap_rt, overlap_corr = greenbook_gap(gb_paths["gap_prev"], gb_paths["gap_new"])
    infl_rt = greenbook_inflation(gb_paths["pgdp"], RAW / "greenbook" / "gpgdp_unzipped")
    manifest["sources"]["greenbook_gap_prev"]["rows"] = int(len(gap_rt))
    manifest["sources"]["greenbook_pgdp"]["rows"] = int(len(infl_rt))
    waterfall.append({"step": "Greenbook real-time gap, one vintage per quarter", "observations": len(gap_rt),
                      "first": str(gap_rt.index.min()), "last": str(gap_rt.index.max())})
    waterfall.append({"step": "Greenbook real-time deflator inflation, one vintage per quarter",
                      "observations": len(infl_rt), "first": str(infl_rt.index.min()), "last": str(infl_rt.index.max())})

    kpath = RAW / "kanzig" / "oilSupplyNewsShocks_2025M12.xlsx"
    dl = fetch(KANZIG, kpath, refresh=refresh)
    oil = kanzig_quarterly(kpath)
    record(manifest, "kanzig_oil_supply_news", KANZIG, kpath, len(oil), dl)
    waterfall.append({"step": "Kanzig oil supply series, quarterly sums", "observations": len(oil),
                      "first": str(oil.index.min()), "last": str(oil.index.max())})
    save_manifest(manifest)

    # Revised (current-vintage) quarterly series.
    q = pd.DataFrame({
        "ffr": to_quarter_mean(fred["FEDFUNDS"]),
        "unrate": to_quarter_mean(fred["UNRATE"]),
        "cpi": to_quarter_mean(fred["CPIAUCSL"]),
        "core_pce": to_quarter_mean(fred["PCEPILFE"]),
    })
    for sid, name in [("GDPC1", "rgdp"), ("GDPDEF", "deflator"), ("GDPPOT", "potential"), ("NROU", "nrou")]:
        s = fred[sid].copy()
        s.index = s.index.to_period("Q")
        q[name] = s
    q = q.sort_index()
    q = q[q.index >= pd.Period("1959Q1", "Q")]
    q = q[q["rgdp"].notna() | q["ffr"].notna()]

    q["infl_4q"] = 100 * (q["deflator"] / q["deflator"].shift(4) - 1)
    q["infl_q_ann"] = 100 * ((q["deflator"] / q["deflator"].shift(1)) ** 4 - 1)
    q["infl_f4"] = 100 * (q["deflator"].shift(-4) / q["deflator"] - 1)
    q["cpi_infl_4q"] = 100 * (q["cpi"] / q["cpi"].shift(4) - 1)
    q["core_pce_infl_4q"] = 100 * (q["core_pce"] / q["core_pce"].shift(4) - 1)
    q["gap_cbo"] = 100 * (q["rgdp"] - q["potential"]) / q["potential"]
    q["unemp_gap"] = q["unrate"] - q["nrou"]
    q["gap_taylor_trend"] = taylor_gap(np.log(q["rgdp"].dropna())).reindex(q.index)

    # The vintage Taylor saw: 1992Q4 Real-Time Data Set, 1987 dollars and fixed-weight deflator.
    v = pd.DataFrame({"y": rt["ROUTPUT"], "p": rt["P"]}).dropna()
    v["infl_4q"] = 100 * (v["p"] / v["p"].shift(4) - 1)
    v["gap"] = taylor_gap(np.log(v["y"]))
    q["rt92_infl_4q"] = v["infl_4q"].reindex(q.index)
    q["rt92_gap"] = v["gap"].reindex(q.index)
    q.loc[q.index < pd.Period("1984Q1", "Q"), ["rt92_infl_4q", "rt92_gap"]] = np.nan

    q = q.join(gap_rt, how="left").join(infl_rt, how="left").join(oil, how="left")
    q.loc[q.index >= pd.Period("1975Q1", "Q"), ["oil_surprise", "oil_news_shock"]] = (
        q.loc[q.index >= pd.Period("1975Q1", "Q"), ["oil_surprise", "oil_news_shock"]].fillna(0.0))
    q.loc[q.index > oil.index.max(), ["oil_surprise", "oil_news_shock"]] = np.nan

    q["ffr_lag1"] = q["ffr"].shift(1)
    q["zlb"] = (q["ffr"] < 0.25).astype(int)
    q.index.name = "quarter"
    q = q.reset_index()
    q["chair"] = q["quarter"].apply(history.chair_for)
    q["episode"] = q["quarter"].astype(str).map(history.EPISODES).fillna("")
    q = q[q["ffr"].notna()].reset_index(drop=True)

    waterfall.append({"step": "merged quarterly panel (policy rate observed)", "observations": len(q),
                      "first": str(q["quarter"].min()), "last": str(q["quarter"].max())})
    core_ok = q[["ffr", "infl_4q", "gap_cbo"]].notna().all(axis=1)
    waterfall.append({"step": "core variables non-missing (rate, inflation, CBO gap)", "observations": int(core_ok.sum()),
                      "first": str(q.loc[core_ok, "quarter"].min()), "last": str(q.loc[core_ok, "quarter"].max())})
    waterfall.append({"step": "zero-lower-bound quarters (rate below 0.25)", "observations": int(q["zlb"].sum()),
                      "first": str(q.loc[q["zlb"] == 1, "quarter"].min()), "last": str(q.loc[q["zlb"] == 1, "quarter"].max())})
    need = {
        "taylor_1987_1992": ["ffr", "rt92_infl_4q", "rt92_gap"],
        "greenspan_bernanke": ["ffr", "infl_4q", "gap_cbo"],
        "pre_volcker": ["ffr", "infl_4q", "gap_cbo"],
        "cgg_pre_volcker": ["ffr", "infl_4q", "gap_cbo"],
        "cgg_volcker_greenspan": ["ffr", "infl_4q", "gap_cbo"],
        "volcker_greenspan": ["ffr", "infl_4q", "gap_cbo"],
        "post_2008": ["ffr", "infl_4q", "gap_cbo"],
        "full": ["ffr", "infl_4q", "gap_cbo"],
        "full_nonzlb": ["ffr", "infl_4q", "gap_cbo"],
        "realtime_greenbook": ["ffr", "gb_infl_4q", "gb_gap", "infl_4q", "gap_cbo"],
        "oil_iv": ["ffr", "infl_4q", "gap_cbo", "oil_surprise"],
    }
    for name, cols in need.items():
        m = history.sample_mask(q, name) & q[cols].notna().all(axis=1)
        waterfall.append({"step": f"estimation sample {name}", "observations": int(m.sum()),
                          "first": str(q.loc[m, "quarter"].min()), "last": str(q.loc[m, "quarter"].max())})

    # Validation: fail loudly on anything implausible.
    assert q["quarter"].is_unique, "duplicate quarters"
    assert q["ffr"].between(0, 20).all(), "policy rate out of range"
    assert q["infl_4q"].dropna().between(-3, 15).all(), "deflator inflation out of range"
    assert q["gap_cbo"].dropna().between(-15, 10).all(), "CBO gap out of range"
    assert q["gb_gap"].dropna().between(-12, 8).all(), "Greenbook gap out of range"
    assert overlap_corr > 0.95, f"Greenbook gap formats disagree (corr {overlap_corr:.3f})"
    rt92 = q.loc[history.sample_mask(q, "taylor_1987_1992"), ["rt92_infl_4q", "rt92_gap"]]
    assert rt92.notna().all().all() and len(rt92) == 23, "1992 vintage does not cover Taylor's window"

    missing = q[["ffr", "infl_4q", "gap_cbo", "gb_infl_4q", "gb_gap", "oil_surprise", "unemp_gap"]].isna().mean()
    pd.DataFrame({"variable": missing.index, "share_missing_full_panel": missing.values.round(4)}).to_csv(
        RESULTS / "missingness.csv", index=False)

    out = q.copy()
    out["quarter"] = out["quarter"].astype(str)
    keep = ["quarter", "chair", "episode", "zlb", "ffr", "ffr_lag1", "infl_4q", "infl_q_ann", "infl_f4",
            "cpi_infl_4q", "core_pce_infl_4q", "gap_cbo", "gap_taylor_trend", "unemp_gap", "unrate", "nrou",
            "rt92_infl_4q", "rt92_gap", "gb_infl_4q", "gb_infl_f4", "gb_gap", "oil_surprise", "oil_news_shock"]
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_parquet(PANEL, index=False)
    pd.DataFrame(waterfall).to_csv(RESULTS / "sample_waterfall.csv", index=False)
    numbers.update("data", {"greenbook_gap_overlap_corr": overlap_corr, "panel_quarters": len(out),
                            "panel_first": out["quarter"].iloc[0], "panel_last": out["quarter"].iloc[-1],
                            "trend_growth_pct": 100 * TREND_GROWTH})
    print(f"panel: {len(out)} quarters {out['quarter'].iloc[0]} to {out['quarter'].iloc[-1]}; "
          f"Greenbook gap overlap corr {overlap_corr:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download every source")
    main(ap.parse_args().refresh)
