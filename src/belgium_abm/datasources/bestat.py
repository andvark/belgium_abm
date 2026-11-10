"""Statbel BeStat open API client (published 'views' as CSV), with raw cache + sha256.

BeStat (https://bestat.statbel.fgov.be) is Statbel's own open-data API. Statbel's website
download pages are behind an anti-bot challenge and are deliberately NOT scraped.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://bestat.statbel.fgov.be/bestat/api/views"
USER_AGENT = "belgium-abm-data-fetch/0.2 (research tool)"


def fetch_view_csv(view_id: str, cache_dir: Path, *, offline: bool = False) -> tuple[pd.DataFrame, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"bestat_{view_id}.csv"
    if not path.is_file():
        if offline:
            raise RuntimeError(f"BeStat view {view_id} not cached and offline mode requested")
        req = urllib.request.Request(f"{BASE}/{view_id}/result/CSV", headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                path.write_bytes(r.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"BeStat view {view_id}: request failed: {exc}") from exc
    raw = path.read_bytes()
    return pd.read_csv(io.BytesIO(raw)), hashlib.sha256(raw).hexdigest()
