from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import httpx

# ---------- Config ----------
ASTER_BASE_URL = os.getenv("ASTER_BASE_URL", "https://fapi.asterdex.com")
HTTP_TIMEOUT_SECS = float(os.getenv("ASTER_HTTP_TIMEOUT_SECS", "5"))
CACHE_TTL_SEC = int(os.getenv("ASTER_CACHE_TTL_SEC", "1"))

# ---------- Simple memory cache ----------
class _Cache:
    def __init__(self):
        self.data: Dict[str, Tuple[float, object]] = {}

    def get(self, key: str):
        hit = self.data.get(key)
        if not hit:
            return None
        ts, val = hit
        if time.time() - ts > CACHE_TTL_SEC:
            return None
        return val

    def set(self, key: str, val: object):
        self.data[key] = (time.time(), val)

_cache = _Cache()

def _client() -> httpx.Client:
    return httpx.Client(base_url=ASTER_BASE_URL, timeout=HTTP_TIMEOUT_SECS)


def get_mark_price(symbol: str) -> Optional[float]:
    """
    Preferred "live" price for fairness: GET /fapi/v1/premiumIndex?symbol=SYMBOL
    Returns float or None on failure.
    """
    key = f"premiumIndex:{symbol}"
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        with _client() as c:
            r = c.get("/fapi/v1/premiumIndex", params={"symbol": symbol})
            r.raise_for_status()
            data = r.json()
            price = float(data["markPrice"])
            _cache.set(key, price)
            return price
    except Exception:
        return None

def get_last_price(symbol: str) -> Optional[float]:
    key = f"lastPrice:{symbol}"
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        with _client() as c:
            r = c.get("/fapi/v1/ticker/price", params={"symbol": symbol})
            r.raise_for_status()
            data = r.json()
            price = float(data["price"])
            _cache.set(key, price)
            return price
    except Exception:
        return None

def get_live_price(symbol: str) -> float:
    p = get_mark_price(symbol)
    if p is None:
        p = get_last_price(symbol)
    if p is None:
        raise RuntimeError(f"Unable to fetch price for {symbol}")
    return p

def get_24h_stats(symbol: str) -> Dict[str, Optional[float]]:
    key = f"24hr:{symbol}"
    cached = _cache.get(key)
    if cached is not None:
        return cached
    out = {"lastPrice": None, "priceChangePercent": None, "volume": None, "quoteVolume": None}
    try:
        with _client() as c:
            r = c.get("/fapi/v1/ticker/24hr", params={"symbol": symbol})
            r.raise_for_status()
            d = r.json()
            out = {
                "lastPrice": float(d.get("lastPrice")) if d.get("lastPrice") is not None else None,
                "priceChangePercent": float(d.get("priceChangePercent")) if d.get("priceChangePercent") is not None else None,
                "volume": float(d.get("volume")) if d.get("volume") is not None else None,
                "quoteVolume": float(d.get("quoteVolume")) if d.get("quoteVolume") is not None else None,
            }
            _cache.set(key, out)
            return out
    except Exception:
        return out

def get_klines(symbol: str, interval: str, limit: int = 2) -> List[List]:
    key = f"klines:{symbol}:{interval}:{limit}"
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        with _client() as c:
            r = c.get("/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
            r.raise_for_status()
            arr = r.json()
            _cache.set(key, arr)
            return arr
    except Exception:
        return []

def get_mark_klines(symbol: str, interval: str, limit: int = 2) -> List[List]:
    key = f"markK:{symbol}:{interval}:{limit}"
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        with _client() as c:
            r = c.get("/fapi/v1/markPriceKlines", params={"symbol": symbol, "interval": interval, "limit": limit})
            r.raise_for_status()
            arr = r.json()
            _cache.set(key, arr)
            return arr
    except Exception:
        return []


def previous_candle_color(symbol: str, interval: str = "1m", use_mark_price: bool = True) -> Optional[str]:
    kl = get_mark_klines(symbol, interval, limit=2) if use_mark_price else get_klines(symbol, interval, limit=2)
    if not kl:
        return None
    last = kl[-1] 
    o = float(last[1]); c = float(last[4])
    return "GREEN" if c >= o else "RED"

def pct_change_over_window(symbol: str, interval: str = "5m", use_mark_price: bool = True) -> Optional[float]:
    kl = get_mark_klines(symbol, interval, limit=2) if use_mark_price else get_klines(symbol, interval, limit=2)
    if len(kl) < 2:
        return None
    prev = float(kl[-2][4])
    last = float(kl[-1][4])
    if prev == 0:
        return None
    return round((last - prev) / prev * 100.0, 4)

def live_mini_card(symbol: str) -> Dict:
    price = get_live_price(symbol)
    prev1m = previous_candle_color(symbol, "1m", True)
    ch5 = pct_change_over_window(symbol, "5m", True)
    stats = get_24h_stats(symbol)
    return {
        "symbol": symbol,
        "price": price,
        "prev1mColor": prev1m,
        "change5mPct": ch5,
        "stats24h": {
            "volume": stats["volume"],
            "quoteVolume": stats["quoteVolume"],
            "priceChangePercent": stats["priceChangePercent"],
        }
    }

def candle_open_close_snapshot(symbol: str, interval: str = "1m", use_mark_price: bool = True) -> Dict:
    kl = get_mark_klines(symbol, interval, limit=1) if use_mark_price else get_klines(symbol, interval, limit=1)
    if not kl:
        raise RuntimeError("No candle data")
    k = kl[-1]
    return {
        "open_time": int(k[0]),
        "open": float(k[1]),
        "close": float(k[4]),
        "close_time": int(k[6]),
        "interval": interval,
        "series": "mark" if use_mark_price else "contract"
    }
