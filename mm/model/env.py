"""
MarketMakingEnv — gymnasium environment for the IMM agent.

Observation (flat float32 vector):
  market window  x: [F × L]  (passed to TCSA/SL inside the actor)
  private state s_p: [priv_dim]

Action (float32):
  [m*, δ*, φ_bid_1..K, φ_ask_1..K]  — decoded by env into actual orders

Reward:
  R = PnL_step - γ·IP + β·CT
  PnL_step = realized + floating change
  IP = |z| if |z| > C, else 0          (inventory penalty)
  CT = β · (fill_bid + fill_ask)        (transaction bonus)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mm.data.features import F_DIM
from mm.sim.exchange import MarketReplay


class MarketMakingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_matrix: np.ndarray,    # [T, F_DIM] pre-computed
        msg,                            # pd.DataFrame
        ob,                             # pd.DataFrame
        cfg: dict,
        t_min: int | None = None,       # earliest allowed episode start tick
        t_max: int | None = None,       # latest allowed episode start tick
    ) -> None:
        super().__init__()
        self.X = feature_matrix         # [T, F]
        self.msg = msg
        self.ob = ob
        self.cfg = cfg

        self.K = cfg["n_quote_levels"]
        self.L = cfg["tcsa_seq_len"]
        self.V = cfg["total_volume"]
        self.T = len(ob)
        self.ep_len = cfg["episode_ticks"]

        priv_dim = 1 + 4 * self.K + 7
        self._priv_dim = priv_dim
        obs_dim = F_DIM * self.L + priv_dim
        action_dim = 2 + 2 * self.K

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        self.exchange = MarketReplay(
            msg=msg, ob=ob,
            n_levels=cfg["lob_levels"],
            tick_size=cfg["tick_size"],
        )

        # Episode-start constraints for train/test splitting
        self._t_min: int = t_min if t_min is not None else self.L
        self._t_max: int = t_max if t_max is not None else (self.T - self.ep_len - 1)

        self._t_start: int = 0
        self._t: int = 0
        self._ep_start: int = 0
        self._day_start: float = 0.0
        self._day_end: float = 0.0
        self._prev_pnl: float = 0.0

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Random episode start within the allowed split window
        lo = self._t_min
        hi = max(lo + 1, self._t_max)
        t_start = self.np_random.integers(lo, hi)
        self._ep_start = int(t_start)
        self._t = int(t_start)

        # Day boundaries from message timestamps
        self._day_start = float(self.msg["time"].iloc[0])
        self._day_end = float(self.msg["time"].iloc[-1])

        self.exchange.reset(t_start)
        self._prev_pnl = 0.0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        tick  = self.cfg["tick_size"]
        p_ref = self.exchange.p_ref

        # Current LOB best bid/ask — quotes must anchor here, not to p_ref.
        # Anchoring to p_ref (mid) places quotes inside the spread, which
        # never receive passive fills because no execution events happen there.
        ob_row   = self.ob.iloc[self._t]
        best_bid = float(ob_row.get("bid_p1", p_ref - tick))
        best_ask = float(ob_row.get("ask_p1", p_ref + tick))
        if not np.isfinite(best_bid): best_bid = p_ref - tick
        if not np.isfinite(best_ask): best_ask = p_ref + tick

        # m* ∈ [-1,1]: symmetric shift of both quotes (inventory/signal skew)
        m_star = float(action[0]) * self.cfg["max_offset_ticks"] * tick

        # δ* ∈ [0,1]: extra ticks OUTSIDE best bid/ask (0 = quote at touch)
        delta_outside = float(np.abs(action[1])) * self.cfg["max_spread_ticks"] * tick

        phi_bid_raw = action[2 : 2 + self.K]
        phi_ask_raw = action[2 + self.K : 2 + 2 * self.K]
        # phi values are already (softmax) probabilities from the actor;
        # clip to [0,1] and renormalise without applying exp again.
        phi_bid = np.maximum(0.0, phi_bid_raw); phi_bid /= phi_bid.sum() + 1e-8
        phi_ask = np.maximum(0.0, phi_ask_raw); phi_ask /= phi_ask.sum() + 1e-8

        bid_center = best_bid - delta_outside + m_star
        ask_center = best_ask + delta_outside + m_star

        # Clamp: never quote inside the current spread.
        # Inside-spread passive orders can't fill (no execution events there).
        bid_center = min(bid_center, best_bid)
        ask_center = max(ask_center, best_ask)

        # Guard against NaN
        if not np.isfinite(bid_center):
            bid_center = best_bid
        if not np.isfinite(ask_center):
            ask_center = best_ask

        bid_prices  = np.array([
            round((bid_center - i * tick) / tick) * tick for i in range(self.K)
        ], dtype=np.float64)
        ask_prices  = np.array([
            round((ask_center + i * tick) / tick) * tick for i in range(self.K)
        ], dtype=np.float64)

        bid_volumes = np.where(phi_bid > 1e-6,
                              np.maximum(1, np.round(phi_bid * self.V)), 0).astype(int)
        ask_volumes = np.where(phi_ask > 1e-6,
                              np.maximum(1, np.round(phi_ask * self.V)), 0).astype(int)

        info = self.exchange.step(bid_prices, bid_volumes, ask_prices, ask_volumes)

        reward = self._reward(info)
        self._prev_pnl = info["realized_pnl"] + info["unrealized_pnl"]
        self._t += 1

        done = (self._t - self._ep_start) >= self.ep_len or self._t >= self.T - 1
        obs = self._obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, False, info

    # ------------------------------------------------------------------
    def _reward(self, info: dict) -> float:
        pnl_now = info["realized_pnl"] + info["unrealized_pnl"]
        pnl_step = pnl_now - self._prev_pnl

        # Normalise by tick × V so rewards are O(1) regardless of stock price.
        # A perfect 1-tick round-trip on half the volume → reward ≈ +0.5.
        norm = self.cfg["tick_size"] * self.V + 1e-8
        pnl_step_norm = pnl_step / norm

        z = abs(info["inventory"])
        C = self.cfg["inventory_limit"]
        ip = self.cfg["inventory_penalty"] * max(0, z - C)

        ct = self.cfg["fill_bonus"] * (info["fills_bid"] + info["fills_ask"])
        return float(pnl_step_norm - ip + ct)

    def _obs(self) -> np.ndarray:
        t = self._t
        # Market window
        x_window = self.X[max(0, t - self.L) : t]
        if len(x_window) < self.L:
            pad = np.zeros((self.L - len(x_window), F_DIM), dtype=np.float32)
            x_window = np.concatenate([pad, x_window], axis=0)
        x_flat = x_window.ravel()   # [F * L]

        # Private state
        s_p = self._private_state(t)
        return np.concatenate([x_flat, s_p]).astype(np.float32)

    def _private_state(self, t: int) -> np.ndarray:
        st = self.exchange.state
        tick = self.cfg["tick_size"]
        norm_inv = self.cfg["inventory_limit"]
        V = float(self.V)

        z = np.array([st.inventory / norm_inv], dtype=np.float32)

        bid_q, ask_q = self.exchange.queue_positions(self.K)
        bid_v, ask_v = self.exchange.agent_volumes(self.K)
        bid_v = bid_v / (V + 1e-8)
        ask_v = ask_v / (V + 1e-8)

        ep_elapsed = (t - self._ep_start) / max(self.ep_len, 1)
        t_sec = float(self.msg["time"].iloc[t])
        day_elapsed = (t_sec - self._day_start) / max(self._day_end - self._day_start, 1.0)
        time_feats = np.array([1.0 - ep_elapsed, 1.0 - day_elapsed], dtype=np.float32)

        mid = float(self.ob["mid"].iloc[t])
        scale = mid * norm_inv * V + 1e-8
        pnl_r = np.array([st.realized_pnl / scale], dtype=np.float32)
        pnl_u = np.array([(st.inventory * (mid - self.exchange._entry_price())) / scale],
                         dtype=np.float32)

        fills = np.array([
            st.last_fill_bid / (V + 1e-8),
            st.last_fill_ask / (V + 1e-8),
        ], dtype=np.float32)

        p_ref = self.exchange.p_ref
        p_tilde = mid  # approximation for reference staleness
        staleness = np.array([abs(p_tilde - p_ref) / (tick * 10 + 1e-8)], dtype=np.float32)

        return np.concatenate([z, bid_q, ask_q, bid_v, ask_v,
                               time_feats, pnl_r, pnl_u, fills, staleness])

    # ------------------------------------------------------------------
    @staticmethod
    def split_obs(
        obs: "np.ndarray | torch.Tensor",
        f_dim: int,
        seq_len: int,
        priv_dim: int,
    ):
        """Split flat obs into (x_window [B,F,L], s_p [B,priv_dim])."""
        import torch
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        x_size = f_dim * seq_len
        x = obs[..., :x_size].reshape(-1, f_dim, seq_len)
        s_p = obs[..., x_size:]
        return x, s_p

    @property
    def priv_dim(self) -> int:
        return self._priv_dim
