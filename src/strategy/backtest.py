"""Backtest and performance evaluation for the CL/BZ spread strategy (issue #6).

This is the terminal stage of the pipeline. It consumes the per-date signal
table written by ``src.strategy.signals`` and turns it into PnL, cost-adjusted
performance, and an out-of-sample validation of the z-score entry/exit
thresholds that stage leaves configurable.

Three things this stage adds that issue #5 explicitly deferred (see
``docs/signals.md``, "Risk handoff"):

* **Execution and PnL.** Positions are applied via ``next_session_position``
  (decided at t, executed at t+1), turned into dollar PnL using the fact that
  ``spread = CL - alpha - beta * BZ`` with ``alpha`` constant, so
  ``position * spread.diff()`` is exactly the PnL of the hedged CL/BZ book --
  no need to touch ``cl_contracts``/``bz_contracts`` separately.
* **Costs.** A transaction cost on turnover and a financing/carry cost while a
  position is held. Both are explicit, documented placeholders -- issue #6's
  own discussion says financing costs, portfolio scalability, and latency all
  need real input from the professor, not just a plausible guess. Every
  headline metric is reported gross *and* net so sensitivity to these two
  knobs stays visible rather than hidden in one number.
* **The threshold sweep, validated out-of-sample.** ``entry_z``/``exit_z`` are
  swept over a grid and scored in-sample (a descriptive sensitivity surface,
  reported but never the headline), then re-selected walk-forward: each
  calendar-year fold picks its config using only prior data, and that config
  is applied unmodified to the following fold. The concatenated out-of-sample
  curve is the honest "does this actually work" result. This mirrors
  ``subsample_stability`` in ``src.models.spread`` -- refitting from several
  start dates -- applied to threshold selection instead of the hedge ratio.

April 2020 (CL settled at -$37.63 on 2020-04-20) is never filtered out of the
headline numbers; see ``docs/backtest.md``'s dedicated section on it.

Run with::

    uv run python -m src.strategy.backtest

Outputs:

* ``data/processed/backtest_cl_bz.parquet`` -- per-date PnL, costs, and
  cumulative equity for the baseline (default-threshold) run.
* ``outputs/tables/backtest_grid.csv`` -- in-sample entry/exit threshold grid.
* ``outputs/tables/backtest_walkforward.csv`` -- one row per walk-forward
  fold: the config it selected and that fold's out-of-sample performance.
* ``outputs/tables/backtest_summary.csv`` -- headline comparison: baseline,
  best in-sample config, walk-forward out-of-sample, and a capped-risk
  sensitivity row.
* ``outputs/figures/11_threshold_grid_heatmap.png``,
  ``12_walkforward_equity_curve.png``, ``13_walkforward_threshold_stability.png``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.panel import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR
from src.models.spread import SpreadModel, load_model, load_spread
from src.strategy.signals import DEFAULT_ZSCORE_WINDOW, SignalConfig, generate_signals

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

BACKTEST_PATH = PROCESSED_DIR / "backtest_cl_bz.parquet"
GRID_PATH = TABLES_DIR / "backtest_grid.csv"
WALKFORWARD_PATH = TABLES_DIR / "backtest_walkforward.csv"
SUMMARY_PATH = TABLES_DIR / "backtest_summary.csv"

CONTRACT_SIZE_BBL = 1000.0
TRADING_DAYS_PER_YEAR = 252

# Both grids include the SignalConfig() defaults (1.5, 0.5) exactly as a cell,
# so the baseline run below is literally grid cell (1.5, 0.5) rather than a
# separately computed number that could drift out of sync with it.
ENTRY_Z_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
EXIT_Z_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)

#: ~2 trading years: enough to warm up the 126-session rolling window with
#: room to spare, and to give a non-degenerate Sharpe estimate to select on.
MIN_TRAIN_SESSIONS = 504
#: Floor before a trailing partial fold gets merged into the previous one
#: instead of being reported as its own noisy fold.
MIN_TEST_SESSIONS = 60

#: Multiple of the full-sample daily spread-move std used for the one
#: supplementary capped-risk diagnostic row -- illustrative only, not a
#: modeling claim; see docs/backtest.md's April 2020 section.
RISK_CAP_STD_MULTIPLE = 8.0


@dataclass(frozen=True)
class CostConfig:
    """Provisional trading-cost assumptions.

    Issue #6's own discussion says financing costs, portfolio scalability, and
    latency all need real input from the professor -- these bps figures are
    documented placeholders, not researched numbers. Every metric downstream
    is computed gross *and* net so the sensitivity to these two knobs stays
    visible rather than being baked silently into one number.
    """

    transaction_cost_bps: float = 2.0
    financing_bps_per_day: float = 0.5
    apply_costs: bool = True


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: train strictly before ``test_start``, evaluate on
    ``[test_start, test_end]``."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_sessions: int
    n_test_sessions: int


# --------------------------------------------------------------------------
# PnL and cost engine
# --------------------------------------------------------------------------


def contract_notional_usd(
    leg_y: pd.Series, leg_x: pd.Series, beta: float, contract_size: float = CONTRACT_SIZE_BBL
) -> pd.Series:
    """Gross two-leg notional in dollars: ``(|leg_y| + beta*|leg_x|) * contract_size``.

    Uses absolute value deliberately: CL settled at -$37.63 on 2020-04-20, and
    a signed notional there would be economically meaningless. This is a
    normalization convenience for cost and percent-return calculations, not a
    real margin or capital estimate -- see the "portfolio scalability" caveat
    in docs/backtest.md (this repo has no order-book depth data to size real
    capacity).
    """
    return (leg_y.abs() + beta * leg_x.abs()) * contract_size


def compute_turnover(next_session_position: pd.Series) -> pd.Series:
    """Absolute change in the *executed* position, session over session.

    Costs must land on the day money actually moves, so this is computed on
    ``next_session_position`` (what was traded) rather than ``target_position``
    (what was decided).
    """
    position = next_session_position.astype(float)
    return position.diff().abs().fillna(position.abs())


def run_backtest(
    signals: pd.DataFrame,
    model: SpreadModel,
    cost_config: CostConfig = CostConfig(),
    contract_size: float = CONTRACT_SIZE_BBL,
) -> pd.DataFrame:
    """Turn a signal table into daily PnL, costs, and cumulative equity.

    ``signals`` must be the output of ``src.strategy.signals.generate_signals``
    (needs ``<leg_y>``, ``<leg_x>``, ``spread``, ``next_session_position``,
    lowercased leg names matching ``model.leg_y``/``model.leg_x``). Since
    ``spread = leg_y - alpha - beta*leg_x`` with ``alpha`` constant,
    ``position * spread.diff()`` is exactly the dollar PnL of the hedged book,
    up to the ``contract_size`` dollar-per-point multiplier of one ``leg_y``
    lot (default: CL/BZ's 1,000 bbl/$-per-bbl contract).

    ``contract_size`` is a units choice, not a modeling one: Sharpe, percent
    returns, and percent drawdown are all scale-invariant to it as long as
    the *same* value is also passed to ``reference_notional_usd`` for the
    percentage-based metrics -- only the absolute dollar figures move. For a
    pair whose two legs have different dollar-per-point values (e.g. ZQ/SR3,
    see the ``src.strategy.run_zq_sr3`` docstring), this is necessarily an
    approximation using one leg's reference value, not a resolution of that
    pair's own documented DV01 mismatch.
    """
    leg_y_col = model.leg_y.lower()
    leg_x_col = model.leg_x.lower()
    missing = [c for c in (leg_y_col, leg_x_col) if c not in signals.columns]
    if missing:
        raise KeyError(
            f"signals is missing leg column(s) {missing} for model legs "
            f"{model.leg_y}/{model.leg_x}; did you call generate_signals with "
            "a spread_frame that carries both leg price columns?"
        )

    frame = signals.copy()
    position = frame["next_session_position"].astype(float)
    notional = contract_notional_usd(frame[leg_y_col], frame[leg_x_col], model.beta, contract_size)
    turnover = compute_turnover(frame["next_session_position"])

    gross_pnl = position * frame["spread"].diff() * contract_size
    if cost_config.apply_costs:
        transaction_cost = turnover * cost_config.transaction_cost_bps / 1e4 * notional
        financing_cost = position.abs() * cost_config.financing_bps_per_day / 1e4 * notional
    else:
        transaction_cost = pd.Series(0.0, index=frame.index)
        financing_cost = pd.Series(0.0, index=frame.index)
    net_pnl = gross_pnl - transaction_cost - financing_cost

    frame["notional_usd"] = notional
    frame["turnover"] = turnover
    frame["gross_pnl_usd"] = gross_pnl.fillna(0.0)
    frame["transaction_cost_usd"] = transaction_cost.fillna(0.0)
    frame["financing_cost_usd"] = financing_cost.fillna(0.0)
    frame["net_pnl_usd"] = net_pnl.fillna(0.0)
    frame["gross_cum_pnl_usd"] = frame["gross_pnl_usd"].cumsum()
    frame["net_cum_pnl_usd"] = frame["net_pnl_usd"].cumsum()
    return frame


# --------------------------------------------------------------------------
# Performance metrics
# --------------------------------------------------------------------------


def reference_notional_usd(
    spread_frame: pd.DataFrame, model: SpreadModel, contract_size: float = CONTRACT_SIZE_BBL
) -> float:
    """Fixed normalization denominator for "percent return" figures.

    The mean full-sample two-leg gross notional. Used everywhere a dollar PnL
    is expressed as a percentage so every reported percentage shares one
    documented denominator -- explicitly not a margin or capital claim. Pass
    the same ``contract_size`` used for the corresponding ``run_backtest``
    call so percentage-based metrics stay internally consistent.
    """
    leg_y = spread_frame[model.leg_y.lower()]
    leg_x = spread_frame[model.leg_x.lower()]
    return float(contract_notional_usd(leg_y, leg_x, model.beta, contract_size).mean())


def sharpe_ratio(daily_pnl: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe ratio on a daily dollar PnL series.

    Scale-invariant to a fixed position size, so no notional is needed here.
    """
    pnl = daily_pnl.dropna()
    if len(pnl) < 2:
        return float("nan")
    std = pnl.std()
    if std == 0:
        return float("nan")
    return float(pnl.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity_level: pd.Series) -> float:
    """Most negative peak-to-trough fractional drawdown of an equity level series."""
    if len(equity_level) == 0:
        return float("nan")
    running_max = equity_level.cummax()
    drawdown = equity_level / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(annualized_return_pct: float, max_dd_pct: float) -> float:
    if not max_dd_pct or np.isnan(max_dd_pct):
        return float("nan")
    return float(annualized_return_pct / abs(max_dd_pct))


def identify_trades(position: pd.Series) -> pd.Series:
    """Integer id per contiguous nonzero run of ``position``; NaN while flat.

    A "trade" is a maximal run of consecutive nonzero values. The signal state
    machine never reverses sign on the same session (it exits to flat first),
    so a plain flat/nonzero boundary is sufficient to delimit runs.
    """
    pos = position.astype(float)
    active = pos != 0
    starts = active & ~active.shift(1, fill_value=False)
    trade_id = starts.cumsum().where(active)
    return trade_id


def trade_level_stats(
    pnl_frame: pd.DataFrame,
    position_col: str = "next_session_position",
    pnl_col: str = "net_pnl_usd",
) -> pd.DataFrame:
    """One row per round-trip trade: entry/exit dates, side, holding period, PnL."""
    columns = ["trade_id", "entry_date", "exit_date", "side", "n_sessions_held", "trade_net_pnl_usd"]
    trade_id = identify_trades(pnl_frame[position_col])
    if trade_id.notna().sum() == 0:
        return pd.DataFrame(columns=columns)

    grouped = pnl_frame.assign(trade_id=trade_id).dropna(subset=["trade_id"]).groupby("trade_id")
    rows = []
    for tid, group in grouped:
        rows.append(
            {
                "trade_id": int(tid),
                "entry_date": group.index[0],
                "exit_date": group.index[-1],
                "side": "long_spread" if group[position_col].iloc[0] > 0 else "short_spread",
                "n_sessions_held": len(group),
                "trade_net_pnl_usd": float(group[pnl_col].sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def apply_daily_loss_cap(pnl_frame: pd.DataFrame, cap_usd: float) -> pd.DataFrame:
    """Clip daily net losses beyond ``cap_usd`` (gains are never touched).

    Illustrative sensitivity diagnostic only -- see docs/backtest.md's April
    2020 section. Never used for the headline numbers.
    """
    frame = pnl_frame.copy()
    frame["net_pnl_usd"] = frame["net_pnl_usd"].clip(lower=-abs(cap_usd))
    frame["net_cum_pnl_usd"] = frame["net_pnl_usd"].cumsum()
    return frame


def summarize_backtest(
    pnl_frame: pd.DataFrame,
    reference_notional: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    cost_config: CostConfig | None = None,
) -> pd.Series:
    """The shared metrics record: sample dates, return/vol/Sharpe, drawdown,
    trade stats, turnover. Reused for the baseline row, every grid row, every
    walk-forward fold row, and the final summary rows.
    """
    n_sessions = len(pnl_frame)
    total_gross = float(pnl_frame["gross_pnl_usd"].sum()) if n_sessions else float("nan")
    total_net = float(pnl_frame["net_pnl_usd"].sum()) if n_sessions else float("nan")

    # Arithmetic (non-compounding) annualization: appropriate for a fixed-size
    # 1-lot spread book, not a reinvested/compounding portfolio.
    total_return_pct = total_net / reference_notional if reference_notional else float("nan")
    annualized_return_pct = (
        total_return_pct * periods_per_year / n_sessions if n_sessions else float("nan")
    )
    if reference_notional and n_sessions > 1:
        daily_return = pnl_frame["net_pnl_usd"] / reference_notional
        annualized_vol_pct = float(daily_return.std() * np.sqrt(periods_per_year))
    else:
        annualized_vol_pct = float("nan")

    if n_sessions:
        equity_level = reference_notional + pnl_frame["net_pnl_usd"].cumsum()
        max_dd_pct = max_drawdown(equity_level)
    else:
        max_dd_pct = float("nan")

    trades = trade_level_stats(pnl_frame)
    n_trades = len(trades)
    hit_rate = float((trades["trade_net_pnl_usd"] > 0).mean()) if n_trades else float("nan")
    avg_holding = float(trades["n_sessions_held"].mean()) if n_trades else float("nan")

    if n_sessions:
        active_sessions = int((pnl_frame["next_session_position"] != 0).sum())
        pct_time_in_market = active_sessions / n_sessions
        annualized_turnover = float(pnl_frame["turnover"].sum()) * periods_per_year / n_sessions
    else:
        pct_time_in_market = float("nan")
        annualized_turnover = float("nan")

    result = {
        "sample_start": str(pnl_frame.index.min().date()) if n_sessions else "",
        "sample_end": str(pnl_frame.index.max().date()) if n_sessions else "",
        "n_sessions": n_sessions,
        "total_gross_pnl_usd": total_gross,
        "total_net_pnl_usd": total_net,
        "total_return_pct": total_return_pct,
        "annualized_return_pct": annualized_return_pct,
        "annualized_vol_pct": annualized_vol_pct,
        "sharpe_gross": sharpe_ratio(pnl_frame["gross_pnl_usd"], periods_per_year),
        "sharpe_net": sharpe_ratio(pnl_frame["net_pnl_usd"], periods_per_year),
        "max_drawdown_pct": max_dd_pct,
        "calmar_net": calmar_ratio(annualized_return_pct, max_dd_pct),
        "n_round_trip_trades": n_trades,
        "hit_rate": hit_rate,
        "avg_holding_sessions": avg_holding,
        "annualized_turnover": annualized_turnover,
        "pct_time_in_market": pct_time_in_market,
    }
    if cost_config is not None:
        result["cost_transaction_bps"] = cost_config.transaction_cost_bps
        result["cost_financing_bps_per_day"] = cost_config.financing_bps_per_day
    return pd.Series(result, name="value")


# --------------------------------------------------------------------------
# Threshold grid (in-sample, descriptive only)
# --------------------------------------------------------------------------


def build_grid_frames(
    spread_frame: pd.DataFrame,
    model: SpreadModel,
    entry_grid: tuple[float, ...] = ENTRY_Z_GRID,
    exit_grid: tuple[float, ...] = EXIT_Z_GRID,
    window: int = DEFAULT_ZSCORE_WINDOW,
    cost_config: CostConfig = CostConfig(),
    contract_size: float = CONTRACT_SIZE_BBL,
) -> dict[tuple[float, float], pd.DataFrame]:
    """Run the full backtest once per valid ``(entry_z, exit_z)`` combo on the
    full sample. Computed once and reused by both the in-sample grid summary
    and every walk-forward fold -- that reuse is what keeps the walk-forward
    selection leak-free (see ``run_walkforward``).
    """
    frames: dict[tuple[float, float], pd.DataFrame] = {}
    for entry_z in entry_grid:
        for exit_z in exit_grid:
            if exit_z >= entry_z:
                continue
            config = SignalConfig(window=window, entry_z=entry_z, exit_z=exit_z)
            signals = generate_signals(spread_frame, model, config)
            frames[(entry_z, exit_z)] = run_backtest(signals, model, cost_config, contract_size)
    return frames


def summarize_grid(
    grid_frames: dict[tuple[float, float], pd.DataFrame],
    reference_notional: float,
    cost_config: CostConfig = CostConfig(),
) -> pd.DataFrame:
    """One row per grid combo -> ``outputs/tables/backtest_grid.csv``."""
    rows = []
    for (entry_z, exit_z), frame in grid_frames.items():
        summary = summarize_backtest(frame, reference_notional, cost_config=cost_config)
        rows.append({"entry_z": entry_z, "exit_z": exit_z, **summary.to_dict()})
    return pd.DataFrame(rows).sort_values(["entry_z", "exit_z"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Walk-forward expanding-window cross-validation
# --------------------------------------------------------------------------


def build_walkforward_folds(
    index: pd.DatetimeIndex,
    min_train_sessions: int = MIN_TRAIN_SESSIONS,
    min_test_sessions: int = MIN_TEST_SESSIONS,
) -> list[Fold]:
    """Sequential, expanding-window, calendar-year folds.

    Candidate boundaries are each calendar-year start within the sample.
    Boundaries whose preceding training slice is shorter than
    ``min_train_sessions`` are dropped -- those early years become pure
    warm-up for the first eligible fold rather than a hardcoded start date. A
    trailing fold shorter than ``min_test_sessions`` is merged into the
    previous fold instead of being dropped or reported as its own noisy fold.
    """
    index = index.sort_values()
    year_starts = pd.date_range(index.min(), index.max(), freq="YS", tz=index.tz)
    boundaries = [b for b in year_starts if b > index.min()]
    boundaries = [b for b in boundaries if int((index < b).sum()) >= min_train_sessions]

    if not boundaries:
        raise ValueError(
            "Not enough history for a single walk-forward fold: need at least "
            f"{min_train_sessions} sessions before the first eligible year boundary "
            f"(have {len(index)} sessions total)."
        )

    sentinel_end = index.max() + pd.Timedelta(days=1)
    test_starts = boundaries + [sentinel_end]

    folds: list[Fold] = []
    for i, test_start in enumerate(boundaries):
        test_end_exclusive = test_starts[i + 1]
        test_index = index[(index >= test_start) & (index < test_end_exclusive)]
        if len(test_index) == 0:
            continue
        train_index = index[index < test_start]
        folds.append(
            Fold(
                fold_id=len(folds),
                train_start=index.min(),
                train_end=test_start,
                test_start=test_index.min(),
                test_end=test_index.max(),
                n_train_sessions=len(train_index),
                n_test_sessions=len(test_index),
            )
        )

    # Merge a too-short trailing fold into the previous one rather than
    # reporting it as its own noisy fold or silently dropping real data.
    while len(folds) > 1 and folds[-1].n_test_sessions < min_test_sessions:
        last = folds.pop()
        prev = folds[-1]
        merged_test_index = index[(index >= prev.test_start) & (index <= last.test_end)]
        folds[-1] = Fold(
            fold_id=prev.fold_id,
            train_start=prev.train_start,
            train_end=prev.train_end,
            test_start=prev.test_start,
            test_end=last.test_end,
            n_train_sessions=prev.n_train_sessions,
            n_test_sessions=len(merged_test_index),
        )

    first_test_start = folds[0].test_start
    expected_coverage = index[index >= first_test_start]
    actual_coverage = pd.DatetimeIndex([])
    for fold in folds:
        actual_coverage = actual_coverage.append(
            index[(index >= fold.test_start) & (index <= fold.test_end)]
        )
    assert list(actual_coverage) == list(expected_coverage), (
        "walk-forward folds must fully and exactly cover the out-of-sample range"
    )

    return folds


def score_fold_configs(
    grid_frames: dict[tuple[float, float], pd.DataFrame],
    fold: Fold,
    reference_notional: float,
    cost_config: CostConfig = CostConfig(),
) -> pd.DataFrame:
    """Evaluate every grid combo on ``fold``'s training period only (rows dated
    strictly before ``fold.test_start``), dropping the rolling-window
    cold-start rows so the un-scored warm-up period doesn't dilute the Sharpe
    estimate.
    """
    rows = []
    for (entry_z, exit_z), frame in grid_frames.items():
        train = frame.loc[frame.index < fold.test_start].dropna(subset=["zscore"])
        summary = summarize_backtest(train, reference_notional, cost_config=cost_config)
        rows.append({"entry_z": entry_z, "exit_z": exit_z, **summary.to_dict()})
    return pd.DataFrame(rows)


def select_fold_winner(fold_scores: pd.DataFrame) -> tuple[float, float]:
    """Argmax on net Sharpe with a deterministic tie-break; falls back to the
    signal module's own defaults if every combo is degenerate on this fold
    (defensive only -- not expected on real data at this sample size).
    """
    scored = fold_scores.dropna(subset=["sharpe_net"])
    if scored.empty:
        print(
            "WARNING: no grid combo had a defined Sharpe on this fold's training "
            "period; falling back to SignalConfig() defaults."
        )
        default = SignalConfig()
        return (default.entry_z, default.exit_z)
    ranked = scored.sort_values(
        ["sharpe_net", "total_net_pnl_usd", "entry_z", "exit_z"],
        ascending=[False, False, True, True],
    )
    winner = ranked.iloc[0]
    return (float(winner["entry_z"]), float(winner["exit_z"]))


def run_walkforward(
    spread_frame: pd.DataFrame,
    model: SpreadModel,
    entry_grid: tuple[float, ...] = ENTRY_Z_GRID,
    exit_grid: tuple[float, ...] = EXIT_Z_GRID,
    window: int = DEFAULT_ZSCORE_WINDOW,
    cost_config: CostConfig = CostConfig(),
    grid_frames: dict[tuple[float, float], pd.DataFrame] | None = None,
    contract_size: float = CONTRACT_SIZE_BBL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward expanding-window selection of ``(entry_z, exit_z)``.

    For each fold, the winning config is chosen using only rows dated before
    the fold's test period (``score_fold_configs``/``select_fold_winner``),
    then applied by reading that config's already-computed out-of-sample rows
    for the fold's test window. Since ``rolling_zscore``/``generate_positions``
    are purely causal, a value dated *t* is identical whether computed on the
    full series or any slice ending at or after *t* -- so this selection
    cannot be affected by anything dated after the fold's test window.
    """
    if grid_frames is None:
        grid_frames = build_grid_frames(
            spread_frame, model, entry_grid, exit_grid, window, cost_config, contract_size
        )

    reference_notional = reference_notional_usd(spread_frame, model, contract_size)
    folds = build_walkforward_folds(spread_frame.index, MIN_TRAIN_SESSIONS, MIN_TEST_SESSIONS)

    fold_rows = []
    oos_slices = []
    for fold in folds:
        fold_scores = score_fold_configs(grid_frames, fold, reference_notional, cost_config)
        entry_z, exit_z = select_fold_winner(fold_scores)
        oos = grid_frames[(entry_z, exit_z)].loc[fold.test_start : fold.test_end]
        oos_summary = summarize_backtest(oos, reference_notional, cost_config=cost_config)
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "train_start": str(fold.train_start.date()),
                "train_end": str(fold.train_end.date()),
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                "n_train_sessions": fold.n_train_sessions,
                "n_test_sessions": fold.n_test_sessions,
                "selected_entry_z": entry_z,
                "selected_exit_z": exit_z,
                **oos_summary.to_dict(),
            }
        )
        oos_slices.append(oos)

    walkforward_table = pd.DataFrame(fold_rows)
    concatenated = pd.concat(oos_slices).sort_index()
    concatenated["gross_cum_pnl_usd"] = concatenated["gross_pnl_usd"].cumsum()
    concatenated["net_cum_pnl_usd"] = concatenated["net_pnl_usd"].cumsum()
    return walkforward_table, concatenated


# --------------------------------------------------------------------------
# Artifact I/O
# --------------------------------------------------------------------------


def save_backtest(frame: pd.DataFrame, path: Path = BACKTEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def load_backtest(path: Path = BACKTEST_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No backtest artifact at {path}.\n"
            "Run the backtest step first:\n"
            "    uv run python -m src.strategy.backtest"
        )
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_threshold_heatmap(grid_summary: pd.DataFrame, path: Path, pair_label: str = "CL/BZ") -> None:
    """Entry/exit threshold sensitivity, in-sample, full sample only."""
    pivot = grid_summary.pivot(index="entry_z", columns="exit_z", values="sharpe_net")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:g}" for r in pivot.index])
    for i, entry_z in enumerate(pivot.index):
        for j, exit_z in enumerate(pivot.columns):
            value = pivot.loc[entry_z, exit_z]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("exit_z")
    ax.set_ylabel("entry_z")
    ax.set_title(
        f"{pair_label} threshold grid: net Sharpe, in-sample, full sample\n"
        "(descriptive only -- see the walk-forward section for the honest result)"
    )
    fig.colorbar(im, ax=ax, label="net Sharpe")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_walkforward_equity(
    oos_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    best_insample_frame: pd.DataFrame,
    path: Path,
    pair_label: str = "CL/BZ",
    highlight_date: pd.Timestamp | None = pd.Timestamp("2020-04-20"),
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(oos_frame.index, oos_frame["net_cum_pnl_usd"], linewidth=1.6, label="walk-forward OOS (headline)")
    ax.plot(
        baseline_frame.index,
        baseline_frame["net_cum_pnl_usd"],
        linewidth=1.0,
        linestyle="--",
        label="baseline default (1.5 / 0.5), full sample, in-sample",
    )
    ax.plot(
        best_insample_frame.index,
        best_insample_frame["net_cum_pnl_usd"],
        linewidth=1.0,
        linestyle=":",
        label="best in-sample config, full sample, in-sample",
    )
    if highlight_date is not None:
        highlight = pd.Timestamp(highlight_date, tz=oos_frame.index.tz)
        if oos_frame.index.min() <= highlight <= oos_frame.index.max():
            ax.axvline(highlight, color="#c0392b", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("cumulative net PnL ($)")
    ax.set_title(f"{pair_label} backtest: walk-forward out-of-sample vs. in-sample configs")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_walkforward_threshold_stability(
    walkforward_table: pd.DataFrame, path: Path, pair_label: str = "CL/BZ"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(walkforward_table))
    ax.step(x, walkforward_table["selected_entry_z"], where="mid", marker="o", label="selected entry_z")
    ax.step(x, walkforward_table["selected_exit_z"], where="mid", marker="s", label="selected exit_z")
    ax.set_xticks(list(x))
    ax.set_xticklabels(walkforward_table["test_start"], rotation=45, ha="right")
    ax.set_ylabel("z-score threshold")
    ax.set_title(f"{pair_label} walk-forward: selected thresholds by fold")
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
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    spread_frame = load_spread()
    cost_config = CostConfig()
    reference_notional = reference_notional_usd(spread_frame, model)

    baseline_signals = generate_signals(spread_frame, model, SignalConfig())
    baseline = run_backtest(baseline_signals, model, cost_config)
    save_backtest(baseline)

    print("=== BASELINE BACKTEST (entry_z=1.5, exit_z=0.5) ===")
    print(summarize_backtest(baseline, reference_notional, cost_config=cost_config).to_string())

    print("\nBuilding threshold grid ...")
    grid_frames = build_grid_frames(spread_frame, model, cost_config=cost_config)
    grid_summary = summarize_grid(grid_frames, reference_notional, cost_config)
    grid_summary.to_csv(GRID_PATH, index=False)
    best_row = grid_summary.sort_values("sharpe_net", ascending=False).iloc[0]
    best_entry_z, best_exit_z = float(best_row["entry_z"]), float(best_row["exit_z"])
    best_insample = grid_frames[(best_entry_z, best_exit_z)]
    print(
        f"Best in-sample config: entry_z={best_entry_z:g}, exit_z={best_exit_z:g}, "
        f"sharpe_net={best_row['sharpe_net']:.3f}"
    )

    print("\nRunning walk-forward cross-validation ...")
    walkforward_table, oos_frame = run_walkforward(
        spread_frame, model, cost_config=cost_config, grid_frames=grid_frames
    )
    walkforward_table.to_csv(WALKFORWARD_PATH, index=False)
    walkforward_summary = summarize_backtest(oos_frame, reference_notional, cost_config=cost_config)
    print("\n=== WALK-FORWARD OUT-OF-SAMPLE (headline) ===")
    print(walkforward_summary.to_string())

    risk_cap_usd = RISK_CAP_STD_MULTIPLE * spread_frame["spread"].diff().std() * CONTRACT_SIZE_BBL
    capped_baseline = apply_daily_loss_cap(baseline, risk_cap_usd)

    summary_rows = {
        "baseline_default": summarize_backtest(baseline, reference_notional, cost_config=cost_config),
        "best_insample": summarize_backtest(best_insample, reference_notional, cost_config=cost_config),
        "walkforward_oos": walkforward_summary,
        "baseline_default_capped": summarize_backtest(
            capped_baseline, reference_notional, cost_config=cost_config
        ),
    }
    summary_table = pd.DataFrame(summary_rows).T
    summary_table.to_csv(SUMMARY_PATH)

    plot_threshold_heatmap(grid_summary, FIGURES_DIR / "11_threshold_grid_heatmap.png")
    plot_walkforward_equity(oos_frame, baseline, best_insample, FIGURES_DIR / "12_walkforward_equity_curve.png")
    plot_walkforward_threshold_stability(
        walkforward_table, FIGURES_DIR / "13_walkforward_threshold_stability.png"
    )

    print(f"\nBacktest    -> {BACKTEST_PATH}")
    print(f"Grid        -> {GRID_PATH}")
    print(f"Walkforward -> {WALKFORWARD_PATH}")
    print(f"Summary     -> {SUMMARY_PATH}")
    print(f"Figures     -> {FIGURES_DIR}")
    print("\nThis is the final pipeline stage -- see docs/backtest.md for the full writeup.")


if __name__ == "__main__":
    main()
