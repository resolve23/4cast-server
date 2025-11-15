import os, time, math, requests
from typing import List, Dict, Optional

BASE = os.getenv("ASTER_BASE", "https://fapi.asterdex.com").rstrip("/")
TTL  = int(os.getenv("SCORING_CACHE_TTL", "15"))
_cache = {"ts": 0.0, "items": []}

def _to_f(v):
    try:
        x = float(v)
        return 0.0 if math.isnan(x) or math.isinf(x) else x
    except: return 0.0

def fetch_rankings(symbol_whitelist: Optional[List[str]] = None) -> List[Dict]:
    now = time.time()
    if not symbol_whitelist and _cache["items"] and now - _cache["ts"] < TTL:
        return list(_cache["items"])

    url = f"{BASE}/fapi/v1/ticker/24hr"
    resp = requests.get(url, timeout=7)
    resp.raise_for_status()
    arr = resp.json()
    if isinstance(arr, dict):
        arr = arr.get("items") or arr.get("data") or []

    
    if symbol_whitelist:
        wl = {s.upper() for s in symbol_whitelist}
        arr = [x for x in arr if str(x.get("symbol","")).upper() in wl]

    items = []
    for x in arr:
        sym = str(x.get("symbol","")).upper()
        if not sym: continue
        vol_q = _to_f(x.get("quoteVolume")) 
        trades = int(_to_f(x.get("count")))
        chg = x.get("priceChangePercent")
        chg = None if chg is None else round(_to_f(chg), 4)
        items.append({
            "symbol": sym,
            "vol24q": vol_q,
            "trades24h": trades,
            "change_pct_24h": chg,
        })

    items.sort(key=lambda i: (i["vol24q"], i["trades24h"]), reverse=True)


    if items:
        vmin, vmax = min(i["vol24q"] for i in items), max(i["vol24q"] for i in items)
        def scale(v): return 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)
    else:
        def scale(_): return 0.0

    ranked = []
    for idx, it in enumerate(items, start=1):
        ranked.append({
            "rank": idx,
            "symbol": it["symbol"],
            "score": round(scale(it["vol24q"]), 6),
            "volume_24h": round(it["vol24q"], 6),
            "trades_24h": it["trades24h"],
            "change_pct_24h": it["change_pct_24h"],
        })

    if not symbol_whitelist:
        _cache["ts"] = now
        _cache["items"] = list(ranked)
    return ranked
