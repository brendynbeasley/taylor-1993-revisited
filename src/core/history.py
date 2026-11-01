"""Institutional calendar: Fed chair tenures, estimation samples, and dated episodes.

Chair start dates follow the Board of Governors' list of members since 1914
(federalreserve.gov/aboutthefed/bios/board/boardmembership.htm). Episode labels
mark widely documented events used only to name quarters in tables and figures.
"""
from __future__ import annotations

import pandas as pd

CHAIRS: list[tuple[str, str]] = [
    ("Martin", "1951-04-02"),
    ("Burns", "1970-02-01"),
    ("Miller", "1978-03-08"),
    ("Volcker", "1979-08-06"),
    ("Greenspan", "1987-08-11"),
    ("Bernanke", "2006-02-01"),
    ("Yellen", "2014-02-03"),
    ("Powell", "2018-02-05"),
]

# name: (first quarter, last quarter or None for latest, exclude zero-lower-bound quarters)
SAMPLES: dict[str, tuple[str, str | None, bool]] = {
    "taylor_1987_1992": ("1987Q1", "1992Q3", False),
    "greenspan_bernanke": ("1987Q3", "2007Q4", False),
    "pre_volcker": ("1965Q1", "1979Q2", False),
    "cgg_pre_volcker": ("1960Q1", "1979Q2", False),
    "cgg_volcker_greenspan": ("1979Q3", "1996Q4", False),
    "volcker_greenspan": ("1979Q3", "2007Q4", False),
    "post_2008": ("2008Q1", None, False),
    "full": ("1965Q1", None, False),
    "full_nonzlb": ("1965Q1", None, True),
    "realtime_greenbook": ("1987Q3", "2020Q4", True),
    "oil_iv": ("1983Q3", None, True),
}

EPISODES: dict[str, str] = {
    "1973Q4": "OPEC oil embargo",
    "1974Q1": "OPEC oil embargo",
    "1979Q4": "Volcker's October 1979 shift to reserve targeting",
    "1980Q2": "Carter credit controls",
    "1981Q2": "Volcker disinflation peak rates",
    "1981Q3": "Volcker disinflation peak rates",
    "1987Q4": "October 1987 stock market crash",
    "1990Q3": "Iraq invades Kuwait",
    "1994Q1": "1994 tightening cycle begins",
    "1998Q4": "LTCM and Russian default",
    "2001Q4": "September 11 and the 2001 recession",
    "2003Q3": "Fed funds at 1 percent, deflation concerns",
    "2004Q2": "'Considerable period' language before liftoff",
    "2008Q4": "Lehman Brothers collapse",
    "2020Q2": "COVID-19 shutdown",
    "2021Q4": "Inflation broadens, policy still at zero",
    "2022Q2": "Inflation peak, 75 basis point hikes begin",
}


def chair_for(q: pd.Period) -> str:
    """Return the chair in office at the midpoint of quarter ``q``."""
    mid = q.start_time + (q.end_time - q.start_time) / 2
    current = CHAIRS[0][0]
    for name, start in CHAIRS:
        if mid >= pd.Timestamp(start):
            current = name
    return current


def sample_mask(df: pd.DataFrame, name: str) -> pd.Series:
    """Boolean mask selecting a named estimation sample from the quarterly panel."""
    first, last, drop_zlb = SAMPLES[name]
    q = df["quarter"]
    m = q >= pd.Period(first, "Q")
    if last is not None:
        m &= q <= pd.Period(last, "Q")
    if drop_zlb:
        m &= ~df["zlb"].astype(bool)
    return m
