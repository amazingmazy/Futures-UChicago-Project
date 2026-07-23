"""Signal generation for the selected CL/BZ spread (issue #5).

This stage consumes the artifact written by ``src.models.spread`` and turns the
modeled spread into reproducible trading instructions. It deliberately does not
compute PnL; issue #6 owns execution assumptions, costs, and performance.

The baseline rule is a rolling z-score mean-reversion strategy:

* normalize today's spread against a trailing window that is shifted by one
  session, so the statistic only uses information known before today's
  settlement;
* go **long the spread** when the z-score is unusually negative: long CL and
  short ``beta`` BZ;
* go **short the spread** when the z-score is unusually positive: short CL and
  long ``beta`` BZ;
* exit when the spread has reverted close to its rolling mean.

Run with::

    uv run python -m src.strategy.signals

Outputs:

* ``data/processed/signals_cl_bz.parquet`` -- per-date z-score, position, and
  hedge-leg targets.
* ``outputs/tables/signal_summary.csv`` -- small review table for the PR.
* ``outputs/figures/10_cl_bz_signals.png`` -- z-score with entry/exit markers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.panel import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR
from src.models.spread import LEG_X, LEG_Y, SpreadModel, load_model, load_spread

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SIGNALS_PATH = PROCESSED_DIR / "signals_cl_bz.parquet"
DEFAULT_ZSCORE_WINDOW = 126


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for the baseline mean-reversion signal.

    ``entry_z`` and ``exit_z`` are absolute z-score thresholds. The defaults are
    intentionally modest: issue #4 measured a 4.1-day full-sample half-life and
    an 8-9 day recent half-life, so the first backtest should not wait for only
    extreme two-sigma moves before it has any trades to evaluate.
    """

    window: int = DEFAULT_ZSCORE_WINDOW
    min_periods: int | None = None
    entry_z: float = 1.5
    exit_z: float = 0.5
    lag_rolling_stats: bool = True

    def resolved_min_periods(self) -> int:
        """Return the minimum observations required for a z-score."""
        return self.window if self.min_periods is None else self.min_periods


# --------------------------------------------------------------------------
# Validation and z-score construction
# --------------------------------------------------------------------------


def validate_config(config: SignalConfig) -> None:
    """Raise ``ValueError`` if a signal configuration is internally invalid."""
    if config.window < 2:
        raise ValueError("window must be at least 2 sessions")
    if config.entry_z <= 0:
        raise ValueError("entry_z must be positive")
    if config.exit_z < 0:
        raise ValueError("exit_z cannot be negative")
    if config.exit_z >= config.entry_z:
        raise ValueError("exit_z must be smaller than entry_z")

    min_periods = config.resolved_min_periods()
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")
    if min_periods > config.window:
        raise ValueError("min_periods cannot exceed window")


def rolling_zscore(spread: pd.Series, config: SignalConfig = SignalConfig()) -> pd.Series:
    """Compute a trailing rolling z-score for the spread.

    By default the rolling mean and standard deviation are shifted by one row
    before scoring today's spread. That avoids look-ahead bias: today's
    settlement may trigger a signal, but it is not also allowed to define the
    mean and volatility against which it is judged.
    """
    validate_config(config)
    series = spread.astype(float)
    history = series.shift(1) if config.lag_rolling_stats else series
    mean = history.rolling(config.window, min_periods=config.resolved_min_periods()).mean()
    std = history.rolling(config.window, min_periods=config.resolved_min_periods()).std(ddof=0)
    std = std.replace(0.0, np.nan)
    return ((series - mean) / std).rename("zscore")


# --------------------------------------------------------------------------
# Position logic
# --------------------------------------------------------------------------


def generate_positions(zscore: pd.Series, config: SignalConfig = SignalConfig()) -> pd.Series:
    """Turn z-scores into target spread positions.

    Position convention:

    * ``+1`` = long spread = long CL, short ``beta`` BZ.
    * ``-1`` = short spread = short CL, long ``beta`` BZ.
    * ``0`` = flat.

    The state machine exits an existing trade before considering a new one on a
    later row; it does not reverse from long to short on the same settlement.
    That keeps issue #5 focused on clear signals and leaves execution details to
    the backtest issue.
    """
    validate_config(config)
    positions: list[int] = []
    position = 0

    for value in zscore.astype(float):
        if np.isnan(value):
            position = 0
        elif position == 0:
            if value <= -config.entry_z:
                position = 1
            elif value >= config.entry_z:
                position = -1
        elif position == 1:
            if value >= -config.exit_z:
                position = 0
        elif position == -1:
            if value <= config.exit_z:
                position = 0
        else:  # defensive; callers should only ever reach -1/0/+1
            raise ValueError(f"unknown position state: {position}")
        positions.append(position)

    return pd.Series(positions, index=zscore.index, name="target_position", dtype="int64")


def generate_signals(
    spread_frame: pd.DataFrame,
    model: SpreadModel,
    config: SignalConfig = SignalConfig(),
) -> pd.DataFrame:
    """Create the full per-date signal table from a saved spread artifact.

    The input is the frame produced by ``fit_spread_model`` / ``load_spread``.
    It must contain a ``spread`` column. If the CL/BZ price columns are also
    present, they are copied through for easier review and plotting.
    """
    if "spread" not in spread_frame.columns:
        raise KeyError("spread_frame must contain a 'spread' column")

    validate_config(config)
    frame = spread_frame.copy()
    zscore = rolling_zscore(frame["spread"], config)
    target_position = generate_positions(zscore, config)
    previous_target = target_position.shift(1).fillna(0).astype("int64")

    out = pd.DataFrame(index=frame.index)
    for column in [model.leg_y.lower(), model.leg_x.lower(), "spread"]:
        if column in frame.columns:
            out[column] = frame[column]

    out["zscore"] = zscore
    out["target_position"] = target_position
    # Position that a next-stage backtest may apply to next-session returns if it
    # assumes signals are observed after settlement and executed one session later.
    out["next_session_position"] = previous_target
    out["position_change"] = target_position.diff().fillna(target_position).astype("int64")
    out["entry"] = (previous_target == 0) & (target_position != 0)
    out["exit"] = (previous_target != 0) & (target_position == 0)
    out["signal"] = np.select(
        [
            out["entry"] & (target_position == 1),
            out["entry"] & (target_position == -1),
            out["exit"],
        ],
        ["enter_long_spread", "enter_short_spread", "exit"],
        default="hold",
    )

    # Hedge-leg targets in contracts, matching the spread model y - alpha - beta*x.
    out[f"{model.leg_y.lower()}_contracts"] = target_position.astype(float)
    out[f"{model.leg_x.lower()}_contracts"] = -target_position.astype(float) * model.beta
    return out


# --------------------------------------------------------------------------
# Summary, I/O, and figure
# --------------------------------------------------------------------------


def summarize_signals(signals: pd.DataFrame, config: SignalConfig) -> pd.Series:
    """Small machine-readable summary for docs and PR review."""
    entries = signals.loc[signals["entry"]]
    exits = signals.loc[signals["exit"]]
    active = signals["target_position"] != 0

    return pd.Series(
        {
            **{f"config_{k}": v for k, v in asdict(config).items()},
            "sample_start": str(signals.index.min().date()) if len(signals) else "",
            "sample_end": str(signals.index.max().date()) if len(signals) else "",
            "n_sessions": int(len(signals)),
            "n_scored_sessions": int(signals["zscore"].notna().sum()),
            "n_entries": int(len(entries)),
            "n_long_entries": int((entries["target_position"] == 1).sum()),
            "n_short_entries": int((entries["target_position"] == -1).sum()),
            "n_exits": int(len(exits)),
            "pct_time_in_market": float(active.mean()) if len(signals) else float("nan"),
            "mean_abs_entry_zscore": float(entries["zscore"].abs().mean())
            if len(entries)
            else float("nan"),
            "final_target_position": int(signals["target_position"].iloc[-1])
            if len(signals)
            else 0,
        },
        name="value",
    )


def save_signals(signals: pd.DataFrame, path: Path = SIGNALS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(path)


def load_signals(path: Path = SIGNALS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No signal artifact at {path}.\n"
            "Run the signal step first:\n"
            "    uv run python -m src.strategy.signals"
        )
    return pd.read_parquet(path)


def plot_signals(signals: pd.DataFrame, config: SignalConfig, path: Path) -> None:
    """Plot rolling z-score with entry/exit markers for review."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(signals.index, signals["zscore"], linewidth=1.0, label="rolling z-score")
    ax.axhline(config.entry_z, linestyle="--", linewidth=1.0, label=f"entry +/-{config.entry_z:g}")
    ax.axhline(-config.entry_z, linestyle="--", linewidth=1.0)
    ax.axhline(config.exit_z, linestyle=":", linewidth=1.0, label=f"exit +/-{config.exit_z:g}")
    ax.axhline(-config.exit_z, linestyle=":", linewidth=1.0)
    ax.axhline(0, linewidth=0.8)

    long_entries = signals[signals["signal"] == "enter_long_spread"]
    short_entries = signals[signals["signal"] == "enter_short_spread"]
    exits = signals[signals["signal"] == "exit"]
    ax.scatter(long_entries.index, long_entries["zscore"], marker="^", s=36, label="long-spread entry")
    ax.scatter(short_entries.index, short_entries["zscore"], marker="v", s=36, label="short-spread entry")
    ax.scatter(exits.index, exits["zscore"], marker="x", s=30, label="exit")

    ax.set_title(f"{LEG_Y}/{LEG_X} rolling z-score signal rule")
    ax.set_ylabel("z-score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, ncols=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    config = SignalConfig()
    model = load_model()
    spread = load_spread()
    signals = generate_signals(spread, model, config)
    save_signals(signals)

    summary = summarize_signals(signals, config)
    summary.to_csv(TABLES_DIR / "signal_summary.csv", header=True)
    plot_signals(signals, config, FIGURES_DIR / "10_cl_bz_signals.png")

    print("=== SIGNAL GENERATION (rolling z-score mean reversion) ===")
    print(summary.to_string())
    print(f"\nSignals -> {SIGNALS_PATH}")
    print(f"Summary -> {TABLES_DIR / 'signal_summary.csv'}")
    print(f"Figure  -> {FIGURES_DIR / '10_cl_bz_signals.png'}")
    print("\nNext (issue #6): apply next_session_position to spread returns and evaluate PnL.")


if __name__ == "__main__":
    main()
