"""Minimal Eurostat dissemination-API client (JSON-stat 2.0) with an on-disk raw cache.

Eurostat republishes the national statistical institutes' official statistics (for Belgium:
Statbel / the Belgian Labour Force Survey). Only the open, documented API is used; nothing here
bypasses an access control. Every response is cached verbatim with its sha256 so a build is
reproducible offline and auditable.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
USER_AGENT = "belgium-abm-data-fetch/0.2 (research tool)"


class FetchError(RuntimeError):
    pass


def _cache_name(code: str, params: dict[str, list[str] | str]) -> str:
    key = json.dumps(params, sort_keys=True)
    return f"{code}__{hashlib.sha1(key.encode()).hexdigest()[:10]}.json"


def fetch_raw(code: str, params: dict[str, list[str] | str], cache_dir: Path, *, offline: bool = False) -> tuple[dict, str]:
    """Return (json-stat dict, sha256 of raw bytes). Uses the cache when present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / _cache_name(code, params)
    if path.is_file():
        raw = path.read_bytes()
    else:
        if offline:
            raise FetchError(f"{code}: not cached and offline mode requested ({path.name})")
        pairs: list[tuple[str, str]] = [("lang", "en")]
        for k, v in sorted(params.items()):
            for item in ([v] if isinstance(v, str) else v):
                pairs.append((k, item))
        url = f"{BASE}/{code}?{urllib.parse.urlencode(pairs)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
        except urllib.error.URLError as exc:
            raise FetchError(f"{code}: request failed: {exc}") from exc
        path.write_bytes(raw)
    data = json.loads(raw)
    if "error" in data:
        raise FetchError(f"{code}: API error {data['error']}")
    return data, hashlib.sha256(raw).hexdigest()


def to_frame(js: dict) -> pd.DataFrame:
    """Flatten a JSON-stat dataset to a long DataFrame (dimension codes + 'value')."""
    ids: list[str] = js["id"]
    dims = []
    for d in ids:
        idx = js["dimension"][d]["category"]["index"]
        codes = sorted(idx, key=lambda c: idx[c]) if isinstance(idx, dict) else list(idx)
        dims.append(codes)
    values = js["value"]
    status = js.get("status", {})
    rows = []
    for pos, combo in enumerate(itertools.product(*dims)):
        v = values.get(str(pos)) if isinstance(values, dict) else values[pos]
        if v is None:
            continue
        rows.append((*combo, float(v), status.get(str(pos), "") if isinstance(status, dict) else ""))
    return pd.DataFrame(rows, columns=[*ids, "value", "flag"])
