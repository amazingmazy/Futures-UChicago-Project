"""Walk-forward monthly seasonality in the CL/BZ spread.

WTI/Brent seasonality is a real, widely-cited phenomenon (US driving season,
refinery maintenance turnarounds, winter heating demand, the OPEC+ meeting
calendar). This repo has **no OPEC+ meeting calendar or other external
data source** -- only the 9 CME futures roots ingested via Databento (see
``src.data.ingest``). Those narratives are candidate *explanations* for
whatever a month-of-year effect turns out to show, not something this module
can ingest directly. The deliverable here is a **month-of-year** effect
specifically (not day-of-year -- 2015-2026 gives too little history per
calendar day to say anything meaningful about e.g. "March 14th").

The load-bearing design requirement, matching how every other walk-forward
piece in this repo already works: a seasonal effect must be estimated
**expanding-window, using only prior years at each point in time**, never
fit once on the full sample and applied retroactively. ``walkforward_seasonality``
reuses the exact same ``Fold``/``build_walkforward_folds`` machinery
``src.strategy.backtest``'s threshold walk-forward and
``src.strategy.walkforward_beta``'s beta refit both already use, so fold
boundaries are identical across all three mechanisms.

Statistical significance is a **gate**, not a decoration: the trading
adapter (``seasonal_position_multiplier``) only tilts exposure for a
(fold, month) pair when that fold's own training-period estimate flags the
month as significant (p < alpha); every other month is a no-op (multiplier
1.0). This directly implements "test for statistical significance" as
something that changes behavior, not just something reported in a table.

Run with::

    uv run python -m src.models.seasonality

Outputs:

* ``outputs/tables/monthly_seasonality_summary.csv`` -- full-sample fit,
  explicitly descriptive only (see ``docs/seasonality.md``).
* ``outputs/tables/seasonality_walkforward.csv`` -- one row per fold: that
  fold's own (expanding-window) monthly effects, p-values, and which months
  were flagged significant.
* ``outputs/figures/25_monthly_seasonality.png``,
  ``26_seasonal_overlay_equity.png``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever write PNGs, never open a window
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.data.panel import FIGURES_DIR, TABLES_DIR
from src.models.spread import SpreadModel, load_model, load_spread
from src.strategy.backtest import (
    CostConfig,
    Fold,
    MIN_TEST_SESSIONS,
    MIN_TRAIN_SESSIONS,
    build_walkforward_folds,
    reference_notional_usd,
    run_backtest,
    summarize_backtest,
)
from src.strategy.signals import SignalConfig, generate_signals

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

MONTHLY_SUMMARY_PATH = TABLES_DIR / "monthly_seasonality_summary.csv"
WALKFORWARD_PATH = TABLES_DIR / "seasonality_walkforward.csv"


@dataclass(frozen=True)
class MonthlySeasonalityModel:
    """One fit of the month-of-year effect on daily spread changes."""

    as_of: str
    n_obs: int
    month_effect: dict[int, float]  # 1..12 -> mean daily $/bbl spread change
    month_pvalue: dict[int, float]  # per-month t-test p-value vs. zero
    overall_fpvalue: float  # one-way ANOVA: do months differ from each other
    significant_months: list[int]


@dataclass(frozen=True)
class SeasonalOverlayConfig:
    alpha: float = 0.05
    tailwind_scale: float = 1.25
    headwind_scale: float = 0.75


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def fit_monthly_seasonality(
    spread: pd.Series, as_of: pd.Timestamp | None = None, alpha: float = 0.05
) -> MonthlySeasonalityModel:
    """Fit month-of-year dummies on daily spread *changes* (matching the
    repo's existing convention of testing changes rather than levels, e.g.
    ``src.analysis.exploratory_analysis``'s ``change_corr``).

    ``sm.OLS(changes, month_dummies)`` with no separate intercept: each
    dummy's coefficient *is* that month's mean daily change, and its own
    t-test p-value tests whether that mean differs from zero -- exactly the
    per-month significance the walk-forward gate needs. The joint
    ``overall_fpvalue`` is a standard one-way ANOVA across the twelve
    groups (do months differ from *each other*, a different question from
    "does any one month differ from zero").
    """
    changes = spread.diff().dropna()
    if as_of is not None:
        changes = changes.loc[changes.index < as_of]

    months = changes.index.month
    dummies = pd.get_dummies(months, prefix="m").astype(float)
    dummies.index = changes.index
    fit = sm.OLS(changes, dummies).fit()

    month_effect: dict[int, float] = {}
    month_pvalue: dict[int, float] = {}
    for m in range(1, 13):
        col = f"m_{m}"
        if col in fit.params.index:
            month_effect[m] = float(fit.params[col])
            month_pvalue[m] = float(fit.pvalues[col])
        else:
            month_effect[m] = float("nan")
            month_pvalue[m] = float("nan")

    groups = [changes[months == m].to_numpy() for m in range(1, 13) if (months == m).sum() >= 2]
    overall_fpvalue = float(stats.f_oneway(*groups)[1]) if len(groups) >= 2 else float("nan")

    significant_months = [m for m, p in month_pvalue.items() if pd.notna(p) and p < alpha]

    return MonthlySeasonalityModel(
        as_of=str(as_of.date()) if as_of is not None else (str(changes.index.max().date()) if len(changes) else ""),
        n_obs=len(changes),
        month_effect=month_effect,
        month_pvalue=month_pvalue,
        overall_fpvalue=overall_fpvalue,
        significant_months=significant_months,
    )


def walkforward_seasonality(spread: pd.Series, folds: list[Fold]) -> pd.DataFrame:
    """Refit ``fit_monthly_seasonality`` per fold, using only rows strictly
    before that fold's ``test_start`` -- the expanding-window discipline the
    module docstring requires."""
    rows = []
    for fold in folds:
        model = fit_monthly_seasonality(spread, as_of=fold.test_start)
        row = {
            "fold_id": fold.fold_id,
            "test_start": str(fold.test_start.date()),
            "test_end": str(fold.test_end.date()),
            "n_train_obs": model.n_obs,
            "overall_fpvalue": model.overall_fpvalue,
            "n_significant_months": len(model.significant_months),
            "significant_months": ",".join(str(m) for m in model.significant_months),
        }
        for m in range(1, 13):
            row[f"month_{m}_effect"] = model.month_effect[m]
            row[f"month_{m}_pvalue"] = model.month_pvalue[m]
        rows.append(row)
    return pd.DataFrame(rows)


def fold_models_from_walkforward(spread: pd.Series, folds: list[Fold]) -> list[tuple[Fold, MonthlySeasonalityModel]]:
    """The per-fold ``MonthlySeasonalityModel`` objects themselves (not just
    the flattened table), for the trading adapter below."""
    return [(fold, fit_monthly_seasonality(spread, as_of=fold.test_start)) for fold in folds]


# --------------------------------------------------------------------------
# Trading adapter
# --------------------------------------------------------------------------


def seasonal_position_multiplier(
    dates: pd.DatetimeIndex,
    position: pd.Series,
    fold_models: list[tuple[Fold, MonthlySeasonalityModel]],
    config: SeasonalOverlayConfig = SeasonalOverlayConfig(),
) -> pd.Series:
    """Tilt exposure toward months where the *current position's own
    direction* agrees with that fold's significant historical seasonal
    drift, and away from months where it disagrees. A month not flagged
    significant by that fold's own (expanding-window) fit is always a
    no-op -- the multiplier stays 1.0.
    """
    scale = pd.Series(1.0, index=dates)
    for fold, model in fold_models:
        mask = (dates >= fold.test_start) & (dates <= fold.test_end)
        for date in dates[mask]:
            month = date.month
            if month not in model.significant_months:
                continue
            effect = model.month_effect[month]
            pos = position.loc[date]
            if pos == 0 or effect == 0:
                continue
            scale.loc[date] = config.tailwind_scale if pos * effect > 0 else config.headwind_scale
    return scale


def run_seasonal_backtest(
    signals: pd.DataFrame,
    model: SpreadModel,
    fold_models: list[tuple[Fold, MonthlySeasonalityModel]] | None = None,
    config: SeasonalOverlayConfig = SeasonalOverlayConfig(),
    cost_config: CostConfig = CostConfig(),
) -> pd.DataFrame:
    """Apply the seasonal tilt to ``next_session_position`` and price the
    result through ``run_backtest``. ``fold_models=None`` is byte-identical
    to calling ``run_backtest`` directly, matching every other overlay's
    convention in this repo."""
    if fold_models is None:
        return run_backtest(signals, model, cost_config)

    frame = signals.copy()
    position = frame["next_session_position"].astype(float)
    multiplier = seasonal_position_multiplier(frame.index, position, fold_models, config)
    frame["next_session_position"] = position * multiplier
    return run_backtest(frame, model, cost_config)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_monthly_seasonality(model: MonthlySeasonalityModel, path: Path) -> None:
    months = list(range(1, 13))
    effects = [model.month_effect[m] for m in months]
    colors = ["#2c7fb8" if m in model.significant_months else "#bbbbbb" for m in months]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(months, effects, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(months)
    ax.set_xlabel("month")
    ax.set_ylabel("mean daily spread change ($/bbl)")
    ax.set_title(
        f"CL/BZ month-of-year effect, full sample, as of {model.as_of} (n={model.n_obs})\n"
        f"(descriptive only -- blue bars are significant at p<0.05; see the walk-forward section)"
    )
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_seasonal_overlay_equity(baseline: pd.DataFrame, seasonal: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(baseline.index, baseline["net_cum_pnl_usd"], linewidth=1.2, linestyle="--", label="baseline")
    ax.plot(seasonal.index, seasonal["net_cum_pnl_usd"], linewidth=1.4, label="seasonally tilted (walk-forward)")
    ax.set_ylabel("cumulative net PnL ($)")
    ax.set_title("CL/BZ baseline signal: walk-forward seasonal tilt")
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

    full_sample_model = fit_monthly_seasonality(spread_frame["spread"])
    pd.Series(
        {
            **{f"month_{m}_effect": full_sample_model.month_effect[m] for m in range(1, 13)},
            **{f"month_{m}_pvalue": full_sample_model.month_pvalue[m] for m in range(1, 13)},
            "overall_fpvalue": full_sample_model.overall_fpvalue,
            "significant_months": ",".join(str(m) for m in full_sample_model.significant_months),
            "n_obs": full_sample_model.n_obs,
        },
        name="value",
    ).to_csv(MONTHLY_SUMMARY_PATH, header=True)
    print("=== MONTHLY SEASONALITY (full sample, descriptive only) ===")
    print(f"Significant months (p<0.05): {full_sample_model.significant_months}")
    print(f"Overall ANOVA p-value: {full_sample_model.overall_fpvalue:.4f}")
    plot_monthly_seasonality(full_sample_model, FIGURES_DIR / "25_monthly_seasonality.png")

    folds = build_walkforward_folds(spread_frame.index, MIN_TRAIN_SESSIONS, MIN_TEST_SESSIONS)
    walkforward_table = walkforward_seasonality(spread_frame["spread"], folds)
    walkforward_table.to_csv(WALKFORWARD_PATH, index=False)

    fold_models = fold_models_from_walkforward(spread_frame["spread"], folds)
    baseline = run_seasonal_backtest(signals, model, None, cost_config=cost_config)
    seasonal = run_seasonal_backtest(signals, model, fold_models, cost_config=cost_config)

    print("\n=== WALK-FORWARD SEASONAL TILT vs. BASELINE ===")
    comparison = pd.DataFrame(
        {
            "baseline": summarize_backtest(baseline, reference_notional, cost_config=cost_config),
            "seasonal_tilt": summarize_backtest(seasonal, reference_notional, cost_config=cost_config),
        }
    ).T
    print(comparison[["sharpe_gross", "sharpe_net", "max_drawdown_pct"]].to_string())

    plot_seasonal_overlay_equity(baseline, seasonal, FIGURES_DIR / "26_seasonal_overlay_equity.png")

    print(f"\nMonthly summary -> {MONTHLY_SUMMARY_PATH}")
    print(f"Walk-forward    -> {WALKFORWARD_PATH}")
    print(f"Figures         -> {FIGURES_DIR}")
    print("\nSee docs/seasonality.md for the write-up.")


if __name__ == "__main__":
    main()
