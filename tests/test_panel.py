"""Validation tests on the committed analytical panel."""
import numpy as np
import pandas as pd
import pytest

from core import history
from core.io import PANEL


@pytest.fixture(scope="module")
def panel():
    df = pd.read_parquet(PANEL)
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    return df


def test_quarters_unique_and_contiguous(panel):
    q = panel["quarter"]
    assert q.is_unique
    assert ((q - q.shift(1)).dropna().apply(lambda d: d.n) == 1).all()


def test_ranges(panel):
    assert panel["ffr"].between(0, 20).all()
    assert panel["infl_4q"].dropna().between(-3, 15).all()
    assert panel["gap_cbo"].dropna().between(-15, 10).all()
    assert set(panel["chair"]) <= {c for c, _ in history.CHAIRS}


def test_zlb_quarters(panel):
    z = panel.loc[panel["zlb"] == 1, "quarter"].astype(str)
    assert z.min() == "2009Q1" and "2015Q4" in set(z) and "2016Q1" not in set(z)


def test_taylor_window_vintage_complete(panel):
    w = panel[history.sample_mask(panel, "taylor_1987_1992")]
    assert len(w) == 23
    assert w[["rt92_infl_4q", "rt92_gap"]].notna().all().all()


def test_rule_tracks_rate_on_1992_vintage(panel):
    """Regression test for the Stage 1 replication of Taylor's Figure 1."""
    w = panel[history.sample_mask(panel, "taylor_1987_1992")]
    rule = 1 + 1.5 * w["rt92_infl_4q"] + 0.5 * w["rt92_gap"]
    assert np.sqrt(np.mean((w["ffr"] - rule) ** 2)) < 0.5
