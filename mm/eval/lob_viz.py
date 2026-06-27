"""
LOB data visualization and animation — logged to W&B.

Four outputs (all keys prefixed mm/lob/):
  mm/lob/market_overview  — 4-panel static chart: mid, spread, trade vol, OFI
  mm/lob/heatmap          — bid + ask volume heatmaps over time × price offset
  mm/lob/snapshot         — single-tick LOB depth chart
  mm/lob/animation        — animated GIF: LOB depth + mid history evolving over time

Usage
-----
  python -m mm.eval.lob_viz --ticker MSFT
  python -m mm.eval.lob_viz --ticker MSFT --start 500 --frames 400 --fps 15
"""
from __future__ import annotations

import argparse
import os
import time
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

from mm.data.loader import load_lobster

load_dotenv()

DATASET_DIR = "dataset/mm"
_N_VIZ = 5   # default LOB levels to show in depth charts


# ---------------------------------------------------------------------------
# Frame-level data extraction
# ---------------------------------------------------------------------------

def _lob_snap(ob_row: pd.Series, n_levels: int):
    """Return (bid_prices, bid_vols, ask_prices, ask_vols) for one ob row."""
    bp, bv, ap, av = [], [], [], []
    for i in range(1, n_levels + 1):
        p = ob_row.get(f"bid_p{i}", np.nan)
        s = ob_row.get(f"bid_s{i}", 0.0)
        if np.isfinite(p):
            bp.append(float(p)); bv.append(float(s))
        p = ob_row.get(f"ask_p{i}", np.nan)
        s = ob_row.get(f"ask_s{i}", 0.0)
        if np.isfinite(p):
            ap.append(float(p)); av.append(float(s))
    return bp, bv, ap, av


# ---------------------------------------------------------------------------
# 1. Market overview (static, 4 panels)
# ---------------------------------------------------------------------------

def plot_market_overview(
    ob: pd.DataFrame,
    msg: pd.DataFrame,
    start_t: int,
    end_t: int,
) -> plt.Figure:
    """Mid-price, spread, trade volume, and order-flow imbalance."""
    times   = msg["time"].iloc[start_t:end_t].values
    mids    = ob["mid"].iloc[start_t:end_t].values
    spreads = ob["spread"].iloc[start_t:end_t].fillna(0).values

    types = msg["type"].iloc[start_t:end_t].values
    dirs  = msg["direction"].iloc[start_t:end_t].values
    sizes = msg["size"].iloc[start_t:end_t].values.astype(float)

    is_trade = np.isin(types, [4, 5])
    buy_vol  = np.where(is_trade & (dirs == 1),  sizes, 0.0)
    sell_vol = np.where(is_trade & (dirs == -1), sizes, 0.0)

    win = 50
    buy_roll  = pd.Series(buy_vol).rolling(win, min_periods=1).sum().values
    sell_roll = pd.Series(sell_vol).rolling(win, min_periods=1).sum().values
    ofi = (buy_roll - sell_roll) / (buy_roll + sell_roll + 1e-8)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(times, mids, color="#1f77b4", linewidth=0.8)
    axes[0].fill_between(times, mids - spreads / 2, mids + spreads / 2,
                         alpha=0.2, color="#1f77b4")
    axes[0].set_ylabel("Price ($)")
    axes[0].set_title("Mid-price ± half-spread")
    axes[0].grid(alpha=0.3)

    spread_bps = np.where(mids > 0, spreads / mids * 1e4, np.nan)
    axes[1].plot(times, spread_bps, color="#ff7f0e", linewidth=0.8)
    axes[1].set_ylabel("Spread (bps)")
    axes[1].set_title("Bid-ask spread")
    axes[1].grid(alpha=0.3)

    w = max(1.0, (times[-1] - times[0]) / len(times) * 0.8) if len(times) > 1 else 1.0
    axes[2].bar(times, buy_vol,  color="#2ca02c", alpha=0.7, width=w, label="Buy (mkt sell hit bid)")
    axes[2].bar(times, -sell_vol, color="#d62728", alpha=0.7, width=w, label="Sell (mkt buy hit ask)")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylabel("Volume")
    axes[2].set_title("Executed trade volume per tick")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)

    axes[3].plot(times, ofi, color="#9467bd", linewidth=0.8)
    axes[3].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[3].set_ylim(-1.1, 1.1)
    axes[3].set_ylabel("OFI")
    axes[3].set_title(f"Order-flow imbalance (rolling {win}-tick)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(alpha=0.3)

    fig.suptitle(f"Market overview — ticks {start_t}–{end_t}", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. LOB heatmap (bid + ask, time × price-offset)
# ---------------------------------------------------------------------------

def plot_lob_heatmap(
    ob: pd.DataFrame,
    msg: pd.DataFrame,
    start_t: int,
    end_t: int,
    n_levels: int = _N_VIZ,
    tick_size: float = 0.01,
) -> plt.Figure:
    """
    Heatmap where rows = price offset from mid (in ticks), columns = tick index,
    colour = resting volume.  Bid side (green) and ask side (red) shown separately.
    """
    T = end_t - start_t
    max_off = n_levels + 1   # offsets: -(max_off)..+(max_off)
    n_rows = 2 * max_off + 1
    mid_row = max_off        # row index for offset = 0

    bid_grid = np.zeros((n_rows, T), dtype=np.float32)
    ask_grid = np.zeros((n_rows, T), dtype=np.float32)

    for col, t in enumerate(range(start_t, end_t)):
        row = ob.iloc[t]
        mid = float(row.get("mid", np.nan))
        if not np.isfinite(mid):
            continue
        for i in range(1, n_levels + 1):
            bp = row.get(f"bid_p{i}", np.nan)
            bs = row.get(f"bid_s{i}", 0.0)
            if np.isfinite(bp) and np.isfinite(bs):
                off = int(round((bp - mid) / tick_size))
                ri = mid_row + off
                if 0 <= ri < n_rows:
                    bid_grid[ri, col] = float(bs)

            ap = row.get(f"ask_p{i}", np.nan)
            as_ = row.get(f"ask_s{i}", 0.0)
            if np.isfinite(ap) and np.isfinite(as_):
                off = int(round((ap - mid) / tick_size))
                ri = mid_row + off
                if 0 <= ri < n_rows:
                    ask_grid[ri, col] = float(as_)

    # Downsample columns for readability
    MAX_COLS = 800
    if T > MAX_COLS:
        step = T // MAX_COLS
        bid_grid = bid_grid[:, ::step]
        ask_grid = ask_grid[:, ::step]

    def _vmax(g):
        pos = g[g > 0]
        return float(np.percentile(pos, 95)) if len(pos) > 0 else 1.0

    offsets = list(range(-max_off, max_off + 1))
    fig, (ax_b, ax_a) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    im_b = ax_b.imshow(bid_grid, aspect="auto", origin="lower",
                       cmap="Greens", interpolation="nearest",
                       vmin=0, vmax=_vmax(bid_grid))
    ax_b.set_yticks(range(n_rows)[::2])
    ax_b.set_yticklabels([f"{offsets[i]:+d}" for i in range(n_rows)[::2]], fontsize=7)
    ax_b.axhline(mid_row, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_b.set_ylabel("Price offset (ticks)")
    ax_b.set_title("Bid-side resting volume (green = more volume)")
    plt.colorbar(im_b, ax=ax_b, label="Volume")

    im_a = ax_a.imshow(ask_grid, aspect="auto", origin="lower",
                       cmap="Reds", interpolation="nearest",
                       vmin=0, vmax=_vmax(ask_grid))
    ax_a.set_yticks(range(n_rows)[::2])
    ax_a.set_yticklabels([f"{offsets[i]:+d}" for i in range(n_rows)[::2]], fontsize=7)
    ax_a.axhline(mid_row, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_a.set_ylabel("Price offset (ticks)")
    ax_a.set_title("Ask-side resting volume (red = more volume)")
    ax_a.set_xlabel("Tick (downsampled)")
    plt.colorbar(im_a, ax=ax_a, label="Volume")

    fig.suptitle(f"LOB heatmap — ticks {start_t}–{end_t}", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Single-tick LOB depth snapshot
# ---------------------------------------------------------------------------

def plot_lob_snapshot(
    ob: pd.DataFrame,
    msg: pd.DataFrame,
    t: int,
    n_levels: int = _N_VIZ,
    tick_size: float = 0.01,
    history_window: int = 300,
) -> plt.Figure:
    """Side-by-side: LOB depth bar chart + mid-price history up to t."""
    fig, (ax_d, ax_m) = plt.subplots(1, 2, figsize=(14, 5))

    ob_row = ob.iloc[t]
    mid = float(ob_row.get("mid", np.nan))
    bid_p, bid_v, ask_p, ask_v = _lob_snap(ob_row, n_levels)

    def tick_off(p): return (p - mid) / tick_size if np.isfinite(mid) else 0.0

    ax_d.barh([tick_off(p) for p in bid_p], [-v for v in bid_v],
              color="#2ca02c", alpha=0.85, height=0.7, label="Bid")
    ax_d.barh([tick_off(p) for p in ask_p], ask_v,
              color="#d62728", alpha=0.85, height=0.7, label="Ask")
    ax_d.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_d.axvline(0, color="black", linewidth=0.4, alpha=0.3)
    all_v = bid_v + ask_v
    if all_v:
        lim = max(all_v) * 1.1
        ax_d.set_xlim(-lim, lim)
    ax_d.set_xlabel("Volume  (bids ← | → asks)")
    ax_d.set_ylabel("Price offset from mid (ticks)")
    ax_d.set_title(f"LOB depth   t={t}   mid={mid:.4f}")
    ax_d.legend(fontsize=8)
    ax_d.grid(alpha=0.3)
    ax_d.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(x):.0f}"))

    t0 = max(0, t - history_window)
    times   = msg["time"].iloc[t0 : t + 1].values
    mids    = ob["mid"].iloc[t0 : t + 1].values
    spreads = ob["spread"].iloc[t0 : t + 1].fillna(0).values
    ax_m.plot(times, mids, color="#1f77b4", linewidth=1.0, label="Mid")
    ax_m.fill_between(times, mids - spreads / 2, mids + spreads / 2,
                      alpha=0.2, color="#1f77b4", label="±½ spread")
    trd = msg["type"].iloc[t0 : t + 1].isin([4, 5])
    if trd.any():
        ax_m.scatter(msg["time"].iloc[t0 : t + 1][trd].values,
                     ob["mid"].iloc[t0 : t + 1][trd].values,
                     s=10, color="orange", zorder=5, alpha=0.8, label="Trade")
    ax_m.axvline(float(msg["time"].iloc[t]), color="red",
                 linewidth=1.0, linestyle="--", alpha=0.7, label="Now")
    ax_m.set_xlabel("Time (s)")
    ax_m.set_ylabel("Price ($)")
    ax_m.set_title(f"Mid-price history (last {history_window} ticks)")
    ax_m.legend(fontsize=8)
    ax_m.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. LOB animation
# ---------------------------------------------------------------------------

def animate_lob(
    ob: pd.DataFrame,
    msg: pd.DataFrame,
    start_t: int,
    end_t: int,
    n_levels: int = _N_VIZ,
    tick_size: float = 0.01,
    fps: int = 10,
    out_path: str | None = None,
    history_window: int = 200,
) -> str:
    """
    Animate the LOB depth + mid-price history over [start_t, end_t).
    Saves a GIF to out_path and returns the path.
    """
    if out_path is None:
        out_path = os.path.join(tempfile.gettempdir(), "lob_animation.gif")

    # Pre-compute stable axis limits
    slice_ob = ob.iloc[start_t:end_t]
    max_vol = 1.0
    for i in range(1, n_levels + 1):
        for col in (f"bid_s{i}", f"ask_s{i}"):
            if col in slice_ob.columns:
                v = slice_ob[col].max()
                if np.isfinite(v):
                    max_vol = max(max_vol, float(v))

    all_mids = slice_ob["mid"].dropna().values
    if len(all_mids) > 0:
        m_lo = float(np.nanmin(all_mids))
        m_hi = float(np.nanmax(all_mids))
    else:
        m_lo, m_hi = 0.0, 1.0
    pad = max((m_hi - m_lo) * 0.1, tick_size * 2)

    fig, (ax_d, ax_m) = plt.subplots(1, 2, figsize=(13, 4))
    fig.tight_layout(pad=2.5)

    def _draw(frame_idx: int):
        t = start_t + frame_idx
        if t >= len(ob):
            return
        ob_row = ob.iloc[t]
        mid = float(ob_row.get("mid", np.nan))
        bid_p, bid_v, ask_p, ask_v = _lob_snap(ob_row, n_levels)

        ax_d.clear()
        if np.isfinite(mid):
            def off(p): return (p - mid) / tick_size
            ax_d.barh([off(p) for p in bid_p], [-v for v in bid_v],
                      color="#2ca02c", alpha=0.85, height=0.7)
            ax_d.barh([off(p) for p in ask_p], ask_v,
                      color="#d62728", alpha=0.85, height=0.7)
        ax_d.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax_d.set_xlim(-max_vol * 1.1, max_vol * 1.1)
        ax_d.set_ylim(-n_levels - 0.5, n_levels + 0.5)
        ax_d.set_xlabel("Volume")
        ax_d.set_ylabel("Price offset (ticks)")
        t_sec = float(msg["time"].iloc[t])
        ax_d.set_title(f"LOB  t={t}  mid={mid:.4f}  [{t_sec:.1f}s]")
        ax_d.grid(alpha=0.3)
        ax_d.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(x):.0f}"))

        ax_m.clear()
        t0 = max(start_t, t - history_window)
        times   = msg["time"].iloc[t0 : t + 1].values
        mids    = ob["mid"].iloc[t0 : t + 1].values
        spreads = ob["spread"].iloc[t0 : t + 1].fillna(0).values
        ax_m.plot(times, mids, color="#1f77b4", linewidth=1.0)
        ax_m.fill_between(times, mids - spreads / 2, mids + spreads / 2,
                          alpha=0.2, color="#1f77b4")
        trd = msg["type"].iloc[t0 : t + 1].isin([4, 5])
        if trd.any():
            ax_m.scatter(msg["time"].iloc[t0 : t + 1][trd].values,
                         ob["mid"].iloc[t0 : t + 1][trd].values,
                         s=8, color="orange", zorder=5, alpha=0.9)
        ax_m.axvline(t_sec, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_m.set_xlim(float(msg["time"].iloc[t0]), float(msg["time"].iloc[t]) + 1)
        ax_m.set_ylim(m_lo - pad, m_hi + pad)
        ax_m.set_xlabel("Time (s)")
        ax_m.set_ylabel("Price ($)")
        ax_m.set_title("Mid-price + trades")
        ax_m.grid(alpha=0.3)

    n_frames = end_t - start_t
    anim = manimation.FuncAnimation(
        fig, _draw, frames=n_frames, interval=1000 // fps, blit=False
    )
    anim.save(out_path, writer="pillow", fps=fps, dpi=80)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def log_lob_viz(
    ob: pd.DataFrame,
    msg: pd.DataFrame,
    wandb_run,
    start_t: int = 0,
    n_frames: int = 300,
    n_levels: int = _N_VIZ,
    tick_size: float = 0.01,
    fps: int = 10,
    scratchpad: str | None = None,
) -> None:
    """
    Generate all LOB visualizations and log to W&B under mm/lob/*.
    """
    end_t = min(start_t + n_frames, len(ob) - 1)
    print(f"  LOB viz: ticks {start_t}–{end_t}  ({end_t - start_t} ticks, {n_levels} levels)")

    if scratchpad is None:
        scratchpad = tempfile.gettempdir()

    print("  [1/4] Market overview...")
    fig = plot_market_overview(ob, msg, start_t, end_t)
    wandb_run.log({"mm/lob/market_overview": wandb.Image(fig)})
    plt.close(fig)

    print("  [2/4] LOB heatmap...")
    fig = plot_lob_heatmap(ob, msg, start_t, end_t,
                           n_levels=n_levels, tick_size=tick_size)
    wandb_run.log({"mm/lob/heatmap": wandb.Image(fig)})
    plt.close(fig)

    print("  [3/4] LOB snapshot...")
    t_snap = start_t + (end_t - start_t) // 2
    fig = plot_lob_snapshot(ob, msg, t_snap,
                            n_levels=n_levels, tick_size=tick_size)
    wandb_run.log({"mm/lob/snapshot": wandb.Image(fig)})
    plt.close(fig)

    print(f"  [4/4] LOB animation ({end_t - start_t} frames @ {fps} fps)...")
    gif_path = os.path.join(scratchpad, "lob_animation.gif")
    animate_lob(ob, msg, start_t, end_t,
                n_levels=n_levels, tick_size=tick_size,
                fps=fps, out_path=gif_path)
    wandb_run.log({"mm/lob/animation": wandb.Video(gif_path, fps=fps, format="gif")})
    print(f"  Animation → {gif_path}")


def _find_lobster_files(ticker: str, levels: int = 10):
    import glob
    msgs = sorted(glob.glob(f"{DATASET_DIR}/{ticker}_*_message_{levels}.csv"))
    obs  = sorted(glob.glob(f"{DATASET_DIR}/{ticker}_*_orderbook_{levels}.csv"))
    if not msgs or not obs:
        # Try other depths
        for depth in (50, 5, 1):
            msgs = sorted(glob.glob(f"{DATASET_DIR}/{ticker}_*_message_{depth}.csv"))
            obs  = sorted(glob.glob(f"{DATASET_DIR}/{ticker}_*_orderbook_{depth}.csv"))
            if msgs and obs:
                return msgs[0], obs[0], depth
        raise FileNotFoundError(f"No LOBSTER files for {ticker} in {DATASET_DIR}")
    return msgs[0], obs[0], levels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LOB data and log to W&B")
    parser.add_argument("--ticker",  default="MSFT",  help="Ticker symbol")
    parser.add_argument("--start",   type=int, default=0,   help="Start tick index")
    parser.add_argument("--frames",  type=int, default=300, help="Number of ticks to visualize")
    parser.add_argument("--fps",     type=int, default=10,  help="Animation FPS")
    parser.add_argument("--levels",  type=int, default=_N_VIZ, help="LOB levels to display")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    msg_path, ob_path, actual_levels = _find_lobster_files(args.ticker)
    print(f"Loading {args.ticker}: {msg_path}")
    msg, ob = load_lobster(msg_path, ob_path, n_levels=actual_levels)
    print(f"  {len(msg)} ticks loaded")

    # Infer tick size from 5th-percentile spread
    spreads = ob["spread"].dropna()
    spreads = spreads[spreads > 0]
    tick_size = float(np.percentile(spreads, 5)) if len(spreads) > 0 else 0.01
    tick_size = max(tick_size, 1e-4)
    print(f"  tick_size={tick_size:.4f}")

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "3frlV2"),
        name=args.run_name or f"mm-lob-viz-{args.ticker}-{int(time.time())}",
        group="mm-lob-viz",
        tags=["mm", "lob", "visualization", args.ticker],
        config={
            "ticker":   args.ticker,
            "start_t":  args.start,
            "n_frames": args.frames,
            "fps":      args.fps,
            "n_levels": args.levels,
            "tick_size": tick_size,
        },
    )

    scratchpad = "/tmp/claude-1000/-home-ahc-Documents-3frlV2/485ba1bd-97e6-4f7d-a677-79531969069d/scratchpad"
    os.makedirs(scratchpad, exist_ok=True)

    log_lob_viz(
        ob, msg, run,
        start_t=args.start,
        n_frames=args.frames,
        n_levels=args.levels,
        tick_size=tick_size,
        fps=args.fps,
        scratchpad=scratchpad,
    )

    run.finish()
    print("Done.")
