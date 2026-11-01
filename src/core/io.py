"""Paths, downloads with retry, SHA-256 hashing, and the provenance manifest."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MANIFEST = RAW / "manifest.json"
PANEL = PROCESSED / "analytical_panel.parquet"

HEADERS = {"User-Agent": "Mozilla/5.0 (academic replication pipeline; github.com/brendynbeasley)"}


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, *, refresh: bool = False, retries: int = 6, timeout: int = 120) -> bool:
    """Download ``url`` to ``dest`` with exponential backoff.

    Returns True if a network download happened, False if a cached copy was used.
    """
    if dest.exists() and not refresh:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    wait = 2.0
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            if r.headers.get("content-type", "").startswith("text/html") and not url.endswith(".html"):
                raise ValueError(f"expected a data file, got HTML from {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return True
        except (requests.RequestException, ValueError) as err:
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {err}") from err
            time.sleep(wait)
            wait *= 2
    return False


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"sources": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest["sources"] = dict(sorted(manifest["sources"].items()))
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def record(manifest: dict, name: str, url: str, path: Path, rows: int, downloaded: bool) -> None:
    """Add or refresh one manifest entry; keep the original timestamp for cached files."""
    prev = manifest["sources"].get(name, {})
    digest = sha256(path)
    stamp = prev.get("retrieved_utc") if (prev.get("sha256") == digest and not downloaded) else None
    manifest["sources"][name] = {
        "url": url,
        "file": str(path.relative_to(ROOT)),
        "retrieved_utc": stamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": int(rows),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }
