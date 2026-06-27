"""
Market-replay order matching simulator.

The agent places resting limit orders at K levels on each side.  At every
tick we replay the historical message stream and check for fills using a
simplified price-time priority rule:

  - An agent bid at price p is filled when ask_p1 crosses ≤ p (aggressive)
    OR when a sell execution (type 4/5, direction -1) occurs at price ≤ p
    and enough pre-existing queue volume has been consumed.
  - Symmetric rule for ask orders.

Queue tracking
--------------
When the agent places an order at level i at tick t, we record:
  queue_ahead[i] = current queue size at that level (volume that has priority)
Each execution event at that level reduces queue_ahead by the executed size.
The order is filled (up to our resting volume) once queue_ahead ≤ 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AgentOrder:
    side: str        # "bid" or "ask"
    level: int       # 1-based level index
    price: float
    volume: int
    queue_ahead: float = 0.0
    filled: int = 0


@dataclass
class ExchangeState:
    cash: float = 0.0
    inventory: int = 0
    realized_pnl: float = 0.0    # profit on *closed* portions of trades
    last_fill_bid: int = 0
    last_fill_ask: int = 0
    orders: list[AgentOrder] = field(default_factory=list)
    position_cost: float = 0.0   # total cost basis of current open position


class MarketReplay:
    """Stateful simulator; call reset() at episode start."""

    def __init__(
        self,
        msg: pd.DataFrame,
        ob: pd.DataFrame,
        n_levels: int = 10,
        tick_size: float = 0.01,
    ) -> None:
        self.msg = msg.reset_index(drop=True)
        self.ob = ob.reset_index(drop=True)
        self.n_levels = n_levels
        self.tick = tick_size
        self.T = len(ob)
        self._t: int = 0
        self.state = ExchangeState()
        # Stable reference price
        self._p_ref: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, t_start: int) -> None:
        self._t = t_start
        self.state = ExchangeState()
        mid = self.ob["mid"].iloc[t_start]
        self._p_ref = self._round_to_tick(float(mid))

    def step(
        self,
        bid_prices: np.ndarray,   # shape [K]
        bid_volumes: np.ndarray,  # shape [K]
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> dict:
        """
        Update the agent's desired order book positions then advance one tick.

        Returns fill info and updated LOB state.
        """
        self.state.last_fill_bid = 0
        self.state.last_fill_ask = 0

        # Process current tick's event FIRST (with existing orders, at their current
        # prices) so execution events match our order prices before any reconcile shift.
        # Then reconcile orders for the next tick using the post-event LOB.
        self._process_tick()
        self._reconcile_orders(bid_prices, bid_volumes, ask_prices, ask_volumes,
                               pre_event_t=self._t)
        self._update_reference_price()
        self._t += 1

        mid = float(self.ob["mid"].iloc[self._t - 1])
        unrealized = self.state.inventory * (mid - self._entry_price())
        return {
            "fills_bid": self.state.last_fill_bid,
            "fills_ask": self.state.last_fill_ask,
            "realized_pnl": self.state.realized_pnl,
            "unrealized_pnl": unrealized,
            "inventory": self.state.inventory,
            "p_ref": self._p_ref,
            "mid": mid,
        }

    @property
    def t(self) -> int:
        return self._t

    @property
    def p_ref(self) -> float:
        return self._p_ref

    def queue_positions(self, n_levels: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (bid_queue, ask_queue) normalised queue positions, shape [n_levels]."""
        bid_q = np.zeros(n_levels, dtype=np.float32)
        ask_q = np.zeros(n_levels, dtype=np.float32)
        for o in self.state.orders:
            idx = o.level - 1
            if idx >= n_levels:
                continue
            q_total = o.queue_ahead + o.volume
            norm = o.queue_ahead / (q_total + 1e-8)
            if o.side == "bid":
                bid_q[idx] = norm
            else:
                ask_q[idx] = norm
        return bid_q, ask_q

    def agent_volumes(self, n_levels: int) -> tuple[np.ndarray, np.ndarray]:
        """Resting agent volumes per level, shape [n_levels]."""
        bid_v = np.zeros(n_levels, dtype=np.float32)
        ask_v = np.zeros(n_levels, dtype=np.float32)
        for o in self.state.orders:
            idx = o.level - 1
            if idx >= n_levels:
                continue
            vol = float(o.volume - o.filled)
            if o.side == "bid":
                bid_v[idx] = vol
            else:
                ask_v[idx] = vol
        return bid_v, ask_v

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _round_to_tick(self, p: float) -> float:
        return round(p / self.tick) * self.tick

    def _entry_price(self) -> float:
        """Average fill price (cost basis) for the current open position."""
        inv = self.state.inventory
        if inv == 0:
            return float(self.ob["mid"].iloc[max(0, self._t - 1)])
        return self.state.position_cost / abs(inv)

    def _reconcile_orders(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
        pre_event_t: int = 0,
    ) -> None:
        """Cancel/place orders to match the agent's desired positions."""
        ob_row = self.ob.iloc[pre_event_t]

        # Build target dicts keyed by (side, level)
        targets: dict[tuple[str, int], tuple[float, int]] = {}
        for lvl, (p, v) in enumerate(zip(bid_prices, bid_volumes), start=1):
            if v > 0:
                targets[("bid", lvl)] = (float(p), int(v))
        for lvl, (p, v) in enumerate(zip(ask_prices, ask_volumes), start=1):
            if v > 0:
                targets[("ask", lvl)] = (float(p), int(v))

        # Remove orders no longer desired or too far from target price.
        # Within 1.5 ticks: sticky quote — update price but preserve queue
        # position so accumulated queue depletion carries across LOB shifts.
        keep = []
        for o in self.state.orders:
            key = (o.side, o.level)
            if key in targets and abs(targets[key][0] - o.price) < self.tick * 1.5:
                o.price = targets[key][0]   # follow the market; queue_ahead kept
                keep.append(o)
                del targets[key]
            # else: cancel (just drop it)
        self.state.orders = keep

        # Place new orders
        for (side, lvl), (price, vol) in targets.items():
            queue_size = self._current_queue_at(ob_row, side, price)
            self.state.orders.append(AgentOrder(
                side=side, level=lvl, price=price,
                volume=vol, queue_ahead=queue_size,
            ))

    def _current_queue_at(self, ob_row: pd.Series, side: str, price: float) -> float:
        """Approximate queue ahead = volume at that price level in current LOB."""
        col_p = "bid_p" if side == "bid" else "ask_p"
        col_s = "bid_s" if side == "bid" else "ask_s"
        for i in range(1, self.n_levels + 1):
            lp = ob_row.get(f"{col_p}{i}", np.nan)
            if np.isfinite(lp) and abs(lp - price) < self.tick * 0.5:
                return float(ob_row.get(f"{col_s}{i}", 0) or 0)
        return 0.0

    def _apply_fill(self, side: str, price: float, fill: int) -> None:
        """
        Update cash, inventory, realized_pnl and position_cost for one fill.

        position_cost tracks the total cost basis of the current open position
        (positive for both longs and shorts — divide by |inventory| to get avg
        entry price).  realized_pnl accumulates only on position-*closing* fills.
        """
        inv = self.state.inventory

        if side == "bid":                          # we bought
            if inv < 0:                            # currently short
                close = min(fill, -inv)
                avg_short = self.state.position_cost / (-inv)
                self.state.realized_pnl += (avg_short - price) * close
                self.state.position_cost -= avg_short * close
                extra = fill - close               # volume that opens new long
                if extra > 0:
                    self.state.position_cost = price * extra
            else:                                  # flat or long
                self.state.position_cost += price * fill
            self.state.inventory += fill
            self.state.cash      -= price * fill
            self.state.last_fill_bid += fill

        else:                                      # side == "ask", we sold
            if inv > 0:                            # currently long
                close = min(fill, inv)
                avg_long = self.state.position_cost / inv
                self.state.realized_pnl += (price - avg_long) * close
                self.state.position_cost -= avg_long * close
                extra = fill - close               # volume that opens new short
                if extra > 0:
                    self.state.position_cost = price * extra
            else:                                  # flat or short
                self.state.position_cost += price * fill
            self.state.inventory -= fill
            self.state.cash      += price * fill
            self.state.last_fill_ask += fill

    def _process_tick(self) -> None:
        """Process current message events and fill matching agent orders."""
        t = self._t
        msg_row = self.msg.iloc[t]
        ob_row = self.ob.iloc[t]

        event_type = int(msg_row["type"])
        event_price = float(msg_row["price"])
        event_size = float(msg_row["size"])
        event_dir = int(msg_row["direction"])

        best_ask = float(ob_row.get("ask_p1", np.inf))
        best_bid = float(ob_row.get("bid_p1", -np.inf))

        for o in self.state.orders:
            remaining = o.volume - o.filled
            if remaining <= 0:
                continue

            if o.side == "bid":
                # Aggressive: ask crossed below our bid price
                if best_ask <= o.price + self.tick * 0.1:
                    fill = remaining
                elif event_type in (4, 5) and event_dir == 1:
                    # direction=1: buy limit order executed (sell mkt hit our bid)
                    if abs(event_price - o.price) < self.tick * 0.5:
                        consume = min(event_size, o.queue_ahead)
                        o.queue_ahead -= consume
                        if o.queue_ahead <= 0:
                            leftover = event_size - consume
                            fill = min(remaining, max(0.0, leftover))
                        else:
                            fill = 0
                    else:
                        fill = 0
                else:
                    fill = 0
            else:  # ask
                # Aggressive: bid crossed above our ask price
                if best_bid >= o.price - self.tick * 0.1:
                    fill = remaining
                elif event_type in (4, 5) and event_dir == -1:
                    # direction=-1: sell limit order executed (buy mkt hit our ask)
                    if abs(event_price - o.price) < self.tick * 0.5:
                        consume = min(event_size, o.queue_ahead)
                        o.queue_ahead -= consume
                        if o.queue_ahead <= 0:
                            leftover = event_size - consume
                            fill = min(remaining, max(0.0, leftover))
                        else:
                            fill = 0
                    else:
                        fill = 0
                else:
                    fill = 0

            fill = int(fill)
            if fill > 0:
                o.filled += fill
                self._apply_fill(o.side, o.price, fill)

        # Remove fully-filled orders
        self.state.orders = [o for o in self.state.orders if o.filled < o.volume]

    def _update_reference_price(self) -> None:
        t = self._t
        if t >= self.T:
            return
        ob_row = self.ob.iloc[t]
        msg_row = self.msg.iloc[t]
        event_type = int(msg_row["type"])
        event_price = float(msg_row["price"])
        event_dir = int(msg_row["direction"])

        best_ask = float(ob_row.get("ask_p1", np.nan))
        best_bid = float(ob_row.get("bid_p1", np.nan))
        if not (np.isfinite(best_ask) and np.isfinite(best_bid)):
            return
        mid = (best_ask + best_bid) / 2.0
        spread = best_ask - best_bid

        # Stable reference price update rules from paper:
        # Changes only when: aggressive order within spread (Q_0 was empty),
        # cancellation of last order at best, or market order consuming best.
        if event_type in (4, 5):
            # Execution — update ref toward mid
            new_ref = self._round_to_tick(mid)
            if abs(new_ref - self._p_ref) >= self.tick:
                self._p_ref = new_ref
        elif event_type == 1 and spread < self.tick * 2:
            # New order inside spread
            new_ref = self._round_to_tick(mid)
            self._p_ref = new_ref
