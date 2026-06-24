"""
Bulk historical downloader — S&P 500 5-min bars, up to 5 years.

Strategy
--------
• Splits each symbol's date range into 90-day chunks to stay within
  FMP's response-size limits for intraday data.
• Downloads all chunks for a symbol sequentially; up to --workers
  symbols run concurrently.
• Saves each completed symbol to dataset/hist/symbols/<SYM>.parquet so a
  crashed/interrupted run resumes from where it left off.
• After all downloads finish, streams per-symbol files into a single
  merged parquet via pyarrow (never holds the full dataset in RAM).

Usage
-----
    python -m pm.data.download
    python -m pm.data.download --years 5 --workers 20 --rpm 250
    python -m pm.data.download --no-merge
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from pm.data.fmp_client import get_sp500_tickers, _RateLimiter

load_dotenv()

FMP_BASE = "https://financialmodelingprep.com/stable"
DATA_ROOT = Path("dataset/hist")
SYM_DIR = DATA_ROOT / "symbols"
MERGED_OUT = DATA_ROOT / "sp500_5min_5yr.parquet"
LOG_FILE = DATA_ROOT / "download_log.json"

OHLCV_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]

_MERGE_SCHEMA = pa.schema([
    pa.field("symbol", pa.large_utf8()),
    pa.field("date",   pa.timestamp("us")),
    pa.field("open",   pa.float64()),
    pa.field("high",   pa.float64()),
    pa.field("low",    pa.float64()),
    pa.field("close",  pa.float64()),
    pa.field("volume", pa.float64()),
])


def date_chunks(from_date: date, to_date: date, chunk_days: int = 90) -> list[tuple[str, str]]:
    """Split [from_date, to_date] into non-overlapping chunks of chunk_days."""
    chunks: list[tuple[str, str]] = []
    cur = from_date
    while cur <= to_date:
        end = min(cur + timedelta(days=chunk_days - 1), to_date)
        chunks.append((cur.isoformat(), end.isoformat()))
        cur = end + timedelta(days=1)
    return chunks


async def _fetch_chunk(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    limiter: _RateLimiter,
    symbol: str,
    from_date: str,
    to_date: str,
    api_key: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    from urllib.parse import quote
    url = (
        f"{FMP_BASE}/historical-chart/5min"
        f"?symbol={quote(symbol, safe='')}&from={from_date}&to={to_date}&apikey={api_key}"
    )
    for attempt in range(1, max_retries + 1):
        async with sem:
            await limiter.acquire()
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        wait = 15 * attempt
                        print(f"  [429] {symbol} {from_date}  retry in {wait}s ...")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        print(f"  [HTTP {resp.status}] {symbol} {from_date} — skipping")
                        return pd.DataFrame()
                    data = await resp.json()
            except Exception as exc:
                wait = 5 * attempt
                print(f"  [err] {symbol} {from_date}: {exc}  retry in {wait}s ...")
                await asyncio.sleep(wait)
                continue

        if not data or not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.insert(0, "symbol", symbol)
        df["date"] = pd.to_datetime(df["date"])
        df = df[[c for c in OHLCV_COLS if c in df.columns]]
        return df.sort_values("date").reset_index(drop=True)

    print(f"  [failed] {symbol} {from_date} after {max_retries} attempts — skipping")
    return pd.DataFrame()


async def _download_symbol(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    limiter: _RateLimiter,
    symbol: str,
    chunks: list[tuple[str, str]],
    api_key: str,
    out_path: Path,
) -> tuple[str, int]:
    parts: list[pd.DataFrame] = []
    for from_d, to_d in chunks:
        chunk_df = await _fetch_chunk(session, sem, limiter, symbol, from_d, to_d, api_key)
        if not chunk_df.empty:
            parts.append(chunk_df)

    if not parts:
        return symbol, 0

    combined = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    combined.to_parquet(out_path, index=False)
    return symbol, len(combined)


async def _run(
    tickers: list[str],
    chunks: list[tuple[str, str]],
    api_key: str,
    workers: int,
    rpm: int,
) -> dict[str, int]:
    SYM_DIR.mkdir(parents=True, exist_ok=True)

    done: dict[str, int] = {}
    todo: list[str] = []
    for sym in tickers:
        p = SYM_DIR / f"{sym}.parquet"
        if p.exists():
            try:
                n = len(pd.read_parquet(p, columns=["date"]))
                if n > 0:
                    done[sym] = n
                    continue
            except Exception:
                pass
        todo.append(sym)

    if done:
        print(f"  Resuming: {len(done)} symbols already done, {len(todo)} remaining.")
    if not todo:
        print("  All symbols already downloaded.")
        return done

    queue: asyncio.Queue[str] = asyncio.Queue()
    for sym in todo:
        await queue.put(sym)

    results: dict[str, int] = dict(done)
    completed = [0]
    start_t = time.monotonic()
    sem = asyncio.Semaphore(workers)
    limiter = _RateLimiter(rpm)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=workers)) as session:
        async def worker() -> None:
            while True:
                try:
                    sym = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                _, n_rows = await _download_symbol(
                    session, sem, limiter, sym, chunks, api_key, SYM_DIR / f"{sym}.parquet"
                )
                results[sym] = n_rows
                completed[0] += 1
                cnt = completed[0]
                if cnt % 10 == 0 or cnt == len(todo):
                    elapsed = time.monotonic() - start_t
                    rate = cnt / elapsed if elapsed > 0 else 0
                    remaining = (len(todo) - cnt) / rate if rate > 0 else 0
                    print(
                        f"  [{cnt:>4}/{len(todo)}]  "
                        f"{elapsed/60:5.1f} min elapsed  "
                        f"~{remaining/60:.0f} min remaining  "
                        f"({rate:.1f} sym/s)"
                    )
                queue.task_done()

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(workers)])

    return results


def merge_to_single(sym_dir: Path, out_path: Path) -> int:
    """
    Concatenate all per-symbol parquets into one sorted parquet.
    Uses pyarrow ParquetWriter so only one symbol is in RAM at a time.
    Returns total row count.
    """
    sym_files = sorted(sym_dir.glob("*.parquet"))
    if not sym_files:
        print("  No symbol files found — nothing to merge.")
        return 0

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    for p in sym_files:
        try:
            table = pq.read_table(str(p)).cast(_MERGE_SCHEMA)
        except Exception as e:
            print(f"  [merge warn] skipping {p.name}: {e}")
            continue
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), _MERGE_SCHEMA, compression="snappy")
        writer.write_table(table)
        total_rows += len(table)

    if writer:
        writer.close()
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download S&P 500 5-min history from FMP")
    parser.add_argument("--years",    type=int, default=5)
    parser.add_argument("--workers",  type=int, default=20)
    parser.add_argument("--rpm",      type=int, default=250)
    parser.add_argument("--chunk",    type=int, default=90)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise SystemExit("FMP_API_KEY not set in environment / .env")

    to_date = date.today()
    from_date = date(to_date.year - args.years, to_date.month, to_date.day)
    chunks = date_chunks(from_date, to_date, chunk_days=args.chunk)

    print(f"\nFetching S&P 500 tickers ...")
    tickers = get_sp500_tickers()
    n_sym = len(tickers)
    n_calls = n_sym * len(chunks)
    est_min = n_calls / args.rpm

    print(f"\n{'═'*60}")
    print(f"  5-min historical download  —  pre-flight summary")
    print(f"{'─'*60}")
    print(f"  Date range    {from_date}  →  {to_date}  ({args.years} years)")
    print(f"  Symbols       {n_sym}")
    print(f"  Chunks/symbol {len(chunks)}  ({args.chunk}-day windows)")
    print(f"  Total API calls  ~{n_calls:,}")
    print(f"  Rate limit    {args.rpm} req/min  →  ~{est_min:.0f} min estimated")
    print(f"  Workers       {args.workers}")
    print(f"  Output dir    {SYM_DIR}")
    print(f"  Merged file   {MERGED_OUT}")
    print(f"{'═'*60}\n")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    results = asyncio.run(_run(tickers, chunks, api_key, workers=args.workers, rpm=args.rpm))
    elapsed = time.monotonic() - t0

    succeeded = {s: n for s, n in results.items() if n > 0}
    failed = [s for s, n in results.items() if n == 0]
    total_rows = sum(succeeded.values())
    disk_mb = sum(
        (SYM_DIR / f"{s}.parquet").stat().st_size
        for s in succeeded if (SYM_DIR / f"{s}.parquet").exists()
    ) / 1e6

    print(f"\n{'─'*60}")
    print(f"  Download complete in {elapsed/60:.1f} min")
    print(f"  Symbols with data : {len(succeeded)}/{n_sym}")
    print(f"  Total rows        : {total_rows:,}")
    if failed:
        print(f"  Failed symbols    : {len(failed)}  ({', '.join(failed[:10])}"
              + (" ..." if len(failed) > 10 else "") + ")")
    print(f"  Disk used (symbols/) : {disk_mb:.0f} MB")

    LOG_FILE.write_text(json.dumps({
        "date": date.today().isoformat(),
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "succeeded": len(succeeded),
        "failed": failed,
        "total_rows": total_rows,
        "disk_mb": round(disk_mb, 1),
    }, indent=2))

    if not args.no_merge:
        print(f"\nMerging {len(succeeded)} symbol files → {MERGED_OUT} ...")
        t1 = time.monotonic()
        merged_rows = merge_to_single(SYM_DIR, MERGED_OUT)
        merge_s = time.monotonic() - t1
        merged_mb = MERGED_OUT.stat().st_size / 1e6 if MERGED_OUT.exists() else 0
        print(f"  Merged {merged_rows:,} rows in {merge_s:.1f}s")
        print(f"  Merged file : {merged_mb:.0f} MB  →  {MERGED_OUT}")

    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
