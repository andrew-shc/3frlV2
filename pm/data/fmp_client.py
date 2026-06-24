"""FMP API client — async 5-minute bar fetcher for S&P 500 constituents."""
from __future__ import annotations

import asyncio
import os
from datetime import date
from typing import Optional
from urllib.parse import quote

import aiohttp
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FMP_BASE = "https://financialmodelingprep.com/stable"
_SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 tickers from Wikipedia."""
    import requests
    from io import StringIO
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    html = requests.get(_SP500_WIKI, headers=headers, timeout=15).text
    tickers = pd.read_html(StringIO(html))[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


class _RateLimiter:
    """Token-bucket rate limiter. Must be created inside an async context."""

    def __init__(self, rate_per_minute: int) -> None:
        self._rate = rate_per_minute / 60.0
        self._tokens = float(rate_per_minute)
        self._last = asyncio.get_running_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            self._tokens += (now - self._last) * self._rate
            self._tokens = min(self._tokens, self._rate * 60)
            self._last = now
            if self._tokens < 1.0:
                await asyncio.sleep((1.0 - self._tokens) / self._rate)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


async def _fetch_5min(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    limiter: _RateLimiter,
    symbol: str,
    from_date: str,
    to_date: str,
    api_key: str,
) -> tuple[str, pd.DataFrame]:
    url = (
        f"{FMP_BASE}/historical-chart/5min"
        f"?symbol={quote(symbol, safe='')}&from={from_date}&to={to_date}&apikey={api_key}"
    )
    async with sem:
        await limiter.acquire()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status == 429:
                    print(f"  [rate limit] {symbol}, retrying after 10s...")
                    await asyncio.sleep(10)
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r2:
                        r = r2
                data = await r.json()
                if not data or not isinstance(data, list):
                    return symbol, pd.DataFrame()
                df = pd.DataFrame(data)
                df.insert(0, "symbol", symbol)
                df["date"] = pd.to_datetime(df["date"])
                return symbol, df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(f"  [warn] {symbol}: {e}")
            return symbol, pd.DataFrame()


async def _poll_all(
    tickers: list[str],
    from_date: str,
    to_date: str,
    api_key: str,
    max_concurrent: int = 10,
    requests_per_minute: int = 250,
) -> dict[str, pd.DataFrame]:
    limiter = _RateLimiter(requests_per_minute)
    sem = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=max_concurrent)) as session:
        results = await asyncio.gather(*[
            _fetch_5min(session, sem, limiter, sym, from_date, to_date, api_key)
            for sym in tickers
        ])
    return {sym: df for sym, df in results}


def poll_sp500_5min(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    api_key: Optional[str] = None,
    max_concurrent: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch 5-minute OHLCV bars for all current S&P 500 constituents + ^GSPC.

    Returns:
        stocks_df : long-format DataFrame [symbol, date, open, high, low, close, volume]
        index_df  : same format for ^GSPC (falls back to SPY if unavailable)
    """
    api_key = api_key or os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("FMP_API_KEY not set")

    today = date.today().isoformat()
    from_date = from_date or today
    to_date = to_date or today

    print("Fetching S&P 500 tickers from Wikipedia...")
    tickers = get_sp500_tickers()
    print(f"  {len(tickers)} tickers found")

    all_symbols = tickers + ["^GSPC", "SPY"]
    print(f"Polling {len(all_symbols)} symbols ({from_date} → {to_date}) at 5-min resolution...")

    results = asyncio.run(_poll_all(all_symbols, from_date, to_date, api_key, max_concurrent))

    index_df = results.pop("^GSPC", pd.DataFrame())
    spy_df = results.pop("SPY", pd.DataFrame())
    if index_df.empty:
        index_df = spy_df

    non_empty = sum(1 for df in results.values() if not df.empty)
    print(f"  Received data for {non_empty}/{len(results)} stocks")

    stocks_df = pd.concat(
        [df for df in results.values() if not df.empty],
        ignore_index=True,
    )
    return stocks_df, index_df
