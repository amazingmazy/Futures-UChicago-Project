"""Regime-conditioned position sizing, derived entirely from CL/BZ's own
price/vol history -- no external data source.

There is no macro or volatility-index data anywhere in this repo, only the
Databento futures settlement prices already ingested (see
``src.data.ingest``). So "regime" here means a classification built from the
spread's own trailing realized volatility (and, optionally, its own trend),
not an external market-state signal.

**The one place a naive implementation of this leaks, and the fix.**
Classifying "today is a high-vol day" against a *full-sample* quantile of
realized vol is a lookahead bug -- it requires knowing the eventual all-time
vol distribution before the sample has finished. This module uses
``pandas``' native ``.expanding().quantile(q)`` on the already-lagged
trailing-vol series from ``src.strategy.risk_overlay.spread_daily_dollar_vol``
instead: the threshold used on date *t* is computed only from vol
observations available by *t*, exactly what a trader computing this live,
day by day, would have had.

The optional trend gate is shipped **off by default**
(``RegimeConfig.trend_gate_enabled=False``) and should stay that way absent
further study: suppressing exposure during a strong trend is in real tension
with a *mean-reversion* signal. A strong trend is sometimes exactly the
regime where a spread is least likely to revert soon -- but it is also the
exact situation a naive z-score rule kept adding to a losing position in,
during April 2020 (see ``docs/backtest.md``). This module treats that as a
genuine open question, not a proven feature.

Run with::

    uv run python -m src.strategy.regime

Outputs:

* ``outputs/tables/regime_summary.csv`` -- baseline vs. vol-regime-scaled
  (and, if enabled, vol+trend-gated) on the CL/BZ baseline signal.
* ``outputs/figures/24_regime_equity.png``.
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
from src.strategy.backtest import CostConfig, reference_notional_usd, run_backtest, summarize_backtest
from src.strategy.risk_overlay import spread_daily_dollar_vol
from src.strategy.signals import SignalConfig, generate_signals

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SUMMARY_PATH = TABLES_DIR / "regime_summary.csv"

#: Matches the repo's existing ROLLING_WINDOW stability-diagnostic convention
#: (src.analysis.exploratory_analysis) -- a slower, "regime" lookback, versus
#: risk_overlay's faster 63-day tactical vol-target lookback.
DEFAULT_VOL_LOOKBACK = 252


@dataclass(frozen=True)
class RegimeConfig:
    vol_lookback: int = DEFAULT_VOL_LOOKBACK
    vol_low_quantile: float = 0.25
    vol_high_quantile: float = 0.75
    low_vol_scale: float = 1.0
    high_vol_scale: float = 0.5
    #: Off by default -- see module docstring.
    trend_gate_enabled: bool = False
    trend_lookback: int = 63
    trend_z_threshold: float = 1.0
    trend_scale: float = 0.5


# --------------------------------------------------------------------------
# Regime classification
# --------------------------------------------------------------------------


def classify_vol_regime(spread: pd.Series, config: RegimeConfig = RegimeConfig()) -> pd.Series:
    """{'low', 'mid', 'high'} per date, from expanding quantiles of the
    already-causal trailing vol -- never a full-sample quantile.

    ``vol.expanding().quantile(q)`` at date *t* uses only vol observations
    up to and including *t*; since ``spread_daily_dollar_vol`` is itself
    already lagged (``shift(1)`` under the hood), this introduces no
    additional leakage -- it answers "is today's vol high relative to the
    distribution of vol observed by today," exactly what could be computed
    live.
    """
    vol = spread_daily_dollar_vol(spread, config.vol_lookback)
    low_threshold = vol.expanding(min_periods=config.vol_lookback).quantile(config.vol_low_quantile)
    high_threshold = vol.expanding(min_periods=config.vol_lookback).quantile(config.vol_high_quantile)

    regime = pd.Series("mid", index=spread.index, dtype=object)
    regime[vol <= low_threshold] = "low"
    regime[vol >= high_threshold] = "high"
    regime[vol.isna()] = "mid"
    return regime


def classify_trend_regime(spread: pd.Series, config: RegimeConfig = RegimeConfig()) -> pd.Series:
    """{'trending', 'flat'} per date: is the spread's own lagged deviation
    from its trailing mean unusually large relative to its trailing vol.

    Causal via the same shift-then-roll idiom used throughout this repo:
    both the trailing mean/std and the deviation being judged use only data
    strictly before *t*.
    """
    lagged = spread.shift(1)
    trailing_mean = lagged.rolling(config.trend_lookback).mean()
    trailing_std = lagged.rolling(config.trend_lookback).std().replace(0.0, np.nan)
    z = (lagged - trailing_mean) / trailing_std
    trending = z.abs() >= config.trend_z_threshold
    return pd.Series(np.where(trending, "trending", "flat"), index=spread.index)


def regime_scale(spread: pd.Series, config: RegimeConfig = RegimeConfig()) -> pd.Series:
    """Position multiplier from the vol regime, further reduced during a
    trend if ``config.trend_gate_enabled``."""
    vol_regime = classify_vol_regime(spread, config)
    scale = vol_regime.map({"low": config.low_vol_scale, "mid": 1.0, "high": config.high_vol_scale}).astype(
        float
    )

    if config.trend_gate_enabled:
        trend_regime = classify_trend_regime(spread, config)
        scale = scale.where(trend_regime != "trending", scale * config.trend_scale)

    return scale.fillna(1.0)


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def run_regime_backtest(
    signals: pd.DataFrame,
    model: SpreadModel,
    config: RegimeConfig | None = None,
    cost_config: CostConfig = CostConfig(),
) -> pd.DataFrame:
    """Scale ``next_session_position`` by the regime multiplier and price
    the result through ``run_backtest``.

    ``config=None`` is byte-identical to calling ``run_backtest`` directly
    (tested), matching ``risk_overlay.run_overlay_backtest``'s convention.
    """
    if config is None:
        return run_backtest(signals, model, cost_config)

    frame = signals.copy()
    scale = regime_scale(frame["spread"], config)
    frame["next_session_position"] = frame["next_session_position"].astype(float) * scale
    return run_backtest(frame, model, cost_config)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_regime_equity(frames: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, frame in frames.items():
        ax.plot(frame.index, frame["net_cum_pnl_usd"], linewidth=1.2, label=label)
    ax.set_ylabel("cumulative net PnL ($)")
    ax.set_title("CL/BZ baseline signal: vol-regime scaling")
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

    frames = {
        "baseline (no regime scaling)": run_regime_backtest(signals, model, None, cost_config),
        "vol-regime scaled": run_regime_backtest(signals, model, RegimeConfig(), cost_config),
        "vol-regime + trend gate (exploratory)": run_regime_backtest(
            signals, model, RegimeConfig(trend_gate_enabled=True), cost_config
        ),
    }

    summary_table = pd.DataFrame(
        {label: summarize_backtest(frame, reference_notional, cost_config=cost_config) for label, frame in frames.items()}
    ).T
    summary_table.to_csv(SUMMARY_PATH)
    print("=== VOL-REGIME OVERLAY (CL/BZ baseline signal, full sample) ===")
    print(summary_table[["sharpe_gross", "sharpe_net", "max_drawdown_pct", "annualized_vol_pct"]].to_string())

    plot_regime_equity(frames, FIGURES_DIR / "24_regime_equity.png")

    print(f"\nTable  -> {SUMMARY_PATH}")
    print(f"Figure -> {FIGURES_DIR / '24_regime_equity.png'}")
    print("\nSee this module's docstring for the methodology write-up.")


if __name__ == "__main__":
    main()
