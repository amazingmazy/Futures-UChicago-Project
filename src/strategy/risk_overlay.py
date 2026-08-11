"""Composable risk overlays on top of the CL/BZ z-score signal.

Two independent overlays, both causal by construction, both scored through
the exact same PnL/cost engine (``src.strategy.backtest.run_backtest``) that
already validates the base signal -- an overlay is just a different
``next_session_position`` series fed into that engine, not a parallel PnL
implementation:

* **Volatility targeting.** Scales the position by the inverse of the
  spread's own trailing realized volatility, so a quiet spread gets sized up
  and a stormy one gets sized down, targeting a roughly constant *dollar*
  risk per day rather than a constant *lot* size. Deliberately measures the
  volatility of the raw 1x-lot spread move itself (a market-data quantity),
  not of the strategy's own scaled PnL -- sizing off your own PnL's
  volatility is circular (the scale depends on the vol of a series that
  depends on the scale).
* **Drawdown gate.** Cuts exposure by a fixed fraction once cumulative PnL
  has fallen a set percentage off its running peak, and restores full size
  once it has recovered most of the way back -- the same entry/exit
  hysteresis idea ``SignalConfig``'s ``entry_z``/``exit_z`` already uses,
  applied to equity instead of the z-score. This is a genuine sequential,
  path-dependent computation (today's gate state depends on yesterday's
  equity), so it runs as a single forward loop, the same style
  ``src.strategy.signals.generate_positions`` already uses for the entry/exit
  state machine -- not vectorized, but O(n) and easy to reason about
  causally.

Composing the two (vol target first, then the drawdown gate on the
vol-targeted position) and running both through ``run_backtest`` afterward
means: with neither overlay enabled, ``run_overlay_backtest`` is byte-identical
to calling ``run_backtest`` directly (tested explicitly below).

Run with::

    uv run python -m src.strategy.risk_overlay

Outputs:

* ``outputs/tables/risk_overlay_summary.csv`` -- baseline vs. vol-targeted vs.
  drawdown-gated vs. both, on the CL/BZ baseline signal.
* ``outputs/figures/23_risk_overlay_equity.png``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.panel import FIGURES_DIR, TABLES_DIR
from src.models.spread import SpreadModel, load_model, load_spread
from src.strategy.backtest import (
    CONTRACT_SIZE_BBL,
    CostConfig,
    contract_notional_usd,
    reference_notional_usd,
    run_backtest,
    summarize_backtest,
)
from src.strategy.signals import SignalConfig, generate_signals

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SUMMARY_PATH = TABLES_DIR / "risk_overlay_summary.csv"


@dataclass(frozen=True)
class VolTargetConfig:
    """Scale the position to target a constant daily dollar vol.

    ``target_daily_vol_usd`` has no principled default -- it depends on the
    book size a trader actually wants, so it is a required argument, not
    defaulted.
    """

    target_daily_vol_usd: float
    lookback: int = 63  # ~1 trading quarter: a tactical, fast-moving estimate
    min_scale: float = 0.25
    max_scale: float = 2.0


@dataclass(frozen=True)
class DrawdownGateConfig:
    """Cut exposure after a drawdown, restore it after a partial recovery.

    ``trigger_pct``/``recovery_pct`` are both negative (fractions of
    ``reference_notional``); ``recovery_pct`` must be less severe than
    ``trigger_pct`` for the hysteresis to do anything (mirrors
    ``SignalConfig``'s ``exit_z < entry_z`` requirement).
    """

    trigger_pct: float = -0.15
    recovery_pct: float = -0.05
    de_risked_scale: float = 0.5


# --------------------------------------------------------------------------
# Volatility targeting
# --------------------------------------------------------------------------


def spread_daily_dollar_vol(
    spread: pd.Series, lookback: int, contract_size: float = CONTRACT_SIZE_BBL
) -> pd.Series:
    """Trailing realized dollar vol of a 1x-lot spread position.

    Uses the same shift-then-roll idiom ``rolling_zscore`` already uses
    (``src/strategy/signals.py``): the window ending at *t* only ever
    contains moves realized strictly before *t*, so this is available before
    trading on any given day, not after.
    """
    unit_pnl = spread.diff() * contract_size
    return unit_pnl.shift(1).rolling(lookback).std()


def vol_target_scale(
    spread: pd.Series, config: VolTargetConfig, contract_size: float = CONTRACT_SIZE_BBL
) -> pd.Series:
    """Position multiplier: ``target_daily_vol_usd / trailing_vol``, clipped.

    NaN during the lookback warm-up (and if a realized vol of exactly zero
    ever occurs) resolves to a neutral ``1.0`` scale, matching how
    ``SignalConfig``'s own machinery treats an undefined statistic as "do
    nothing" rather than an error.
    """
    vol = spread_daily_dollar_vol(spread, config.lookback, contract_size)
    raw_scale = config.target_daily_vol_usd / vol.replace(0.0, np.nan)
    return raw_scale.clip(lower=config.min_scale, upper=config.max_scale).fillna(1.0)


# --------------------------------------------------------------------------
# Drawdown gate
# --------------------------------------------------------------------------


def apply_drawdown_gate(
    unit_pnl: pd.Series,
    position: pd.Series,
    reference_notional: float,
    config: DrawdownGateConfig = DrawdownGateConfig(),
) -> pd.Series:
    """Scale ``position`` down after a drawdown, back up after a recovery.

    Sequential by construction: the gate scale applied on day *t* is decided
    entirely from equity accumulated strictly before *t* (the loop uses the
    scale carried over from the previous iteration *before* updating it), so
    a large move on day *t* itself can influence day *t+1*'s exposure but
    never day *t*'s own -- see the causality test in
    ``tests/test_risk_overlay.py``.

    Equity here is **gross** of transaction/financing costs -- costs are a
    couple of bps, an order of magnitude smaller than the drawdown
    thresholds this gate reacts to, and computing them inline would require
    duplicating ``run_backtest``'s cost formulas inside a sequential loop.
    ``run_overlay_backtest`` applies real costs afterward, once the gated
    position series is finalized, via the same ``run_backtest`` every other
    position series in this repo is scored through.
    """
    gate_scale = 1.0
    cum_equity = 0.0
    peak_equity = 0.0
    gated_values: list[float] = []

    positions = position.astype(float)
    pnl = unit_pnl.astype(float)
    for date in positions.index:
        gated = positions.loc[date] * gate_scale
        gated_values.append(gated)

        move = pnl.loc[date]
        gross_pnl = gated * move if pd.notna(move) else 0.0
        cum_equity += gross_pnl
        peak_equity = max(peak_equity, cum_equity)
        drawdown_pct = (cum_equity - peak_equity) / reference_notional if reference_notional else 0.0

        if drawdown_pct <= config.trigger_pct:
            gate_scale = config.de_risked_scale
        elif drawdown_pct >= config.recovery_pct:
            gate_scale = 1.0
        # else: hold the current gate_scale (hysteresis band)

    return pd.Series(gated_values, index=position.index, name="gated_position")


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def run_overlay_backtest(
    signals: pd.DataFrame,
    model: SpreadModel,
    vol_config: VolTargetConfig | None = None,
    drawdown_config: DrawdownGateConfig | None = None,
    cost_config: CostConfig = CostConfig(),
) -> pd.DataFrame:
    """Apply the requested overlays (vol target, then drawdown gate) to
    ``next_session_position`` and price the result through ``run_backtest``.

    With both configs ``None`` this is byte-identical to calling
    ``run_backtest(signals, model, cost_config)`` directly -- asserted as a
    regression test, since every overlay here composes with, rather than
    replaces, the base signal. (An early return, rather than a no-op float
    cast, is what makes it byte-identical: casting an int position column to
    float and back is a no-op in value but not in dtype.)
    """
    if vol_config is None and drawdown_config is None:
        return run_backtest(signals, model, cost_config)

    frame = signals.copy()
    position = frame["next_session_position"].astype(float)

    if vol_config is not None:
        scale = vol_target_scale(frame["spread"], vol_config)
        position = position * scale

    if drawdown_config is not None:
        leg_y_col = model.leg_y.lower()
        leg_x_col = model.leg_x.lower()
        reference_notional = float(
            contract_notional_usd(frame[leg_y_col], frame[leg_x_col], model.beta).mean()
        )
        unit_pnl = frame["spread"].diff() * CONTRACT_SIZE_BBL
        position = apply_drawdown_gate(unit_pnl, position, reference_notional, drawdown_config)

    frame["next_session_position"] = position
    return run_backtest(frame, model, cost_config)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_overlay_equity(overlay_frames: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, frame in overlay_frames.items():
        ax.plot(frame.index, frame["net_cum_pnl_usd"], linewidth=1.2, label=label)
    ax.set_ylabel("cumulative net PnL ($)")
    ax.set_title("CL/BZ baseline signal: risk overlays compared")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    spread_frame = load_spread()
    signals = generate_signals(spread_frame, model, SignalConfig())
    cost_config = CostConfig()
    reference_notional = reference_notional_usd(spread_frame, model)

    vol_config = VolTargetConfig(target_daily_vol_usd=reference_notional * 0.01, lookback=63)
    drawdown_config = DrawdownGateConfig()

    frames = {
        "baseline (no overlay)": run_overlay_backtest(signals, model, None, None, cost_config),
        "vol-targeted": run_overlay_backtest(signals, model, vol_config, None, cost_config),
        "drawdown-gated": run_overlay_backtest(signals, model, None, drawdown_config, cost_config),
        "vol-targeted + drawdown-gated": run_overlay_backtest(
            signals, model, vol_config, drawdown_config, cost_config
        ),
    }

    summary_table = pd.DataFrame(
        {label: summarize_backtest(frame, reference_notional, cost_config=cost_config) for label, frame in frames.items()}
    ).T
    summary_table.to_csv(SUMMARY_PATH)
    print("=== RISK OVERLAY COMPARISON (CL/BZ baseline signal, full sample) ===")
    print(summary_table[["sharpe_gross", "sharpe_net", "max_drawdown_pct", "annualized_vol_pct"]].to_string())

    plot_overlay_equity(frames, FIGURES_DIR / "23_risk_overlay_equity.png")

    print(f"\nTable  -> {SUMMARY_PATH}")
    print(f"Figure -> {FIGURES_DIR / '23_risk_overlay_equity.png'}")
    print("\nSee this module's docstring for the methodology write-up.")


if __name__ == "__main__":
    main()
