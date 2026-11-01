"""Single source of truth for every number quoted in the prose.

Each analysis script writes its headline values here under a namespaced key.
The audit script checks that numbers in README.md and TECHNICAL.md trace back
to this file, the results CSVs, or the cited original-paper benchmarks.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .io import RESULTS

PATH = RESULTS / "key_numbers.json"


def _clean(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def update(section: str, values: dict[str, Any]) -> None:
    """Merge ``values`` into ``section`` of key_numbers.json."""
    data = json.loads(PATH.read_text()) if PATH.exists() else {}
    data.setdefault(section, {}).update(_clean(values))
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")


def load() -> dict[str, Any]:
    return json.loads(PATH.read_text()) if PATH.exists() else {}
