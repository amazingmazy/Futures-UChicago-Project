"""Walk-forward hedge-ratio refit: closes the one lookahead gap left in the
threshold walk-forward (issue #6 follow-up).

``src.strategy.backtest``'s walk-forward cross-validation already re-selects
``entry_z``/``exit_z`` per fold using only prior data -- but it does that on
top of a single **static, full-sample OLS beta** from ``src.models.spread``.
That beta is fit on data that includes, for any early fold, years the fold
has not "seen" yet. This module closes that gap: for each walk-forward fold,
the hedge ratio itself is refit on rows strictly before the fold's test
period (reusing ``fit_spread_model`` unchanged), and the resulting
fold-specific spread series is what the threshold grid/selection then runs
on for that fold only.

Nothing in ``src.strategy.backtest`` needs to change for this. Its
``build_grid_frames``/``score_fold_configs``/``select_fold_winner``/
``build_walkforward_folds`` are already pair-agnostic and fold-agnostic --
they just get called once per fold, on a fold-specific spread frame, instead
of once globally on the full-sample fit.

Run with::

    uv run python -m src.strategy.walkforward_beta

Outputs (CL/BZ by default):

* ``data/processed/walkforward_beta_refit_cl_bz.parquet`` -- concatenated
  out-of-sample PnL across folds, each fold priced off its own refit beta.
* ``outputs/tables/backtest_walkforward_beta_refit.csv`` -- one row per fold:
  the refit ``fold_alpha``/``fold_beta``, the selected thresholds, and that
  fold's out-of-sample performance.
* ``outputs/tables/backtest_summary_beta_refit.csv`` -- the static-beta
  walk-forward headline next to this beta-refit one, so the size of the gap
  this closes is a visible number, not a claim.
* ``outputs/figures/14_cl_bz_walkforward_beta_stability.png``,
  ``15_cl_bz_walkforward_beta_refit_equity.png``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever write PNGs, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from src.data.panel import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR, drop_sunday_session, load_panel
from src.models.spread import SpreadModel, fit_spread_model, load_model, load_spread
from src.strategy.backtest import (
    CostConfig,
    ENTRY_Z_GRID,
    EXIT_Z_GRID,
    Fold,
    MIN_TEST_SESSIONS,
    MIN_TRAIN_SESSIONS,
    build_grid_frames,
    build_walkforward_folds,
    reference_notional_usd,
    run_walkforward,
    score_fold_configs,
    select_fold_winner,
    summarize_backtest,
)
from src.strategy.signals import DEFAULT_ZSCORE_WINDOW

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

WALKFORWARD_BETA_PATH = PROCESSED_DIR / "walkforward_beta_refit_cl_bz.parquet"
WALKFORWARD_BETA_TABLE_PATH = TABLES_DIR / "backtest_walkforward_beta_refit.csv"
SUMMARY_BETA_PATH = TABLES_DIR / "backtest_summary_beta_refit.csv"


@dataclass(frozen=True)
class BetaRefitFold:
    """One walk-forward fold together with the hedge ratio refit for it."""

    fold: Fold
    model: SpreadModel
    entry_z: float
    exit_z: float


# --------------------------------------------------------------------------
# Per-fold beta refit
# --------------------------------------------------------------------------


def build_fold_spread_frame(
    panel: pd.DataFrame, leg_y: str, leg_x: str, fold: Fold
) -> tuple[SpreadModel, pd.DataFrame]:
    """Refit the hedge ratio on rows strictly before ``fold.test_start`` and
    apply it across the fold's full train+test range.

    Reuses ``fit_spread_model`` exactly as-is (see ``src.models.spread``) --
    the only new logic here is the ``fold.test_start`` cutoff and applying the
    frozen fold parameters forward via ``SpreadModel.spread()``. The test
    window's spread values are therefore an honest out-of-sample application
    of parameters that never saw the test window, the same causal guarantee
    the z-score itself already has.
    """
    both = panel[[leg_y, leg_x]].dropna()
    train = both.loc[both.index < fold.test_start]
    model, _ = fit_spread_model(train, leg_y, leg_x)

    y = both[leg_y]
    x = both[leg_x]
    frame = pd.DataFrame(
        {
            leg_y.lower(): y,
            leg_x.lower(): x,
            "fitted": model.alpha + model.beta * x,
            "spread": model.spread(y, x),
        }
    )
    return model, frame


def run_walkforward_beta_refit(
    panel: pd.DataFrame,
    leg_y: str,
    leg_x: str,
    entry_grid: tuple[float, ...] = ENTRY_Z_GRID,
    exit_grid: tuple[float, ...] = EXIT_Z_GRID,
    window: int = DEFAULT_ZSCORE_WINDOW,
    cost_config: CostConfig = CostConfig(),
    reference_notional: float | None = None,
    min_train_sessions: int = MIN_TRAIN_SESSIONS,
    min_test_sessions: int = MIN_TEST_SESSIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[BetaRefitFold]]:
    """Walk-forward selection of ``(entry_z, exit_z)`` *and* the hedge ratio.

    Unlike ``src.strategy.backtest.run_walkforward`` (which reuses one
    full-sample-fit ``grid_frames`` cache across every fold), this rebuilds
    the grid **per fold** from a fold-specific refit, because the spread
    series itself -- not just which thresholds win -- differs fold to fold.
    That is roughly a ``len(folds)``-times increase in grid/backtest
    computation versus the static-beta version; each fold's extra OLS refit
    itself is a single cheap ``sm.OLS(...).fit()`` call.

    ``reference_notional``, if not given, is computed from a full-sample fit
    purely as a fixed reporting denominator for percentage-based metrics --
    it does not affect Sharpe (scale-invariant) or which config wins any
    fold, so it introduces no leakage. Pass in the same value used elsewhere
    in ``backtest_summary.csv`` to keep rows comparable.
    """
    both_index = panel[[leg_y, leg_x]].dropna().index
    folds = build_walkforward_folds(both_index, min_train_sessions, min_test_sessions)

    if reference_notional is None:
        full_model, full_frame = fit_spread_model(panel[[leg_y, leg_x]].dropna(), leg_y, leg_x)
        reference_notional = reference_notional_usd(full_frame, full_model)

    fold_rows = []
    oos_slices = []
    beta_refit_folds: list[BetaRefitFold] = []
    for fold in folds:
        fold_model, fold_spread_frame = build_fold_spread_frame(panel, leg_y, leg_x, fold)
        grid_frames_fold = build_grid_frames(
            fold_spread_frame, fold_model, entry_grid, exit_grid, window, cost_config
        )
        fold_scores = score_fold_configs(grid_frames_fold, fold, reference_notional, cost_config)
        entry_z, exit_z = select_fold_winner(fold_scores)
        oos = grid_frames_fold[(entry_z, exit_z)].loc[fold.test_start : fold.test_end]
        oos_summary = summarize_backtest(oos, reference_notional, cost_config=cost_config)

        beta_refit_folds.append(BetaRefitFold(fold=fold, model=fold_model, entry_z=entry_z, exit_z=exit_z))
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "train_start": str(fold.train_start.date()),
                "train_end": str(fold.train_end.date()),
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                "n_train_sessions": fold.n_train_sessions,
                "n_test_sessions": fold.n_test_sessions,
                "fold_alpha": fold_model.alpha,
                "fold_beta": fold_model.beta,
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
    return walkforward_table, concatenated, beta_refit_folds


# --------------------------------------------------------------------------
# Artifact I/O
# --------------------------------------------------------------------------


def save_walkforward_beta(frame: pd.DataFrame, path: Path = WALKFORWARD_BETA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def load_walkforward_beta(path: Path = WALKFORWARD_BETA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No beta-refit walk-forward artifact at {path}.\n"
            "Run this step first:\n"
            "    uv run python -m src.strategy.walkforward_beta"
        )
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_walkforward_beta_stability(
    walkforward_table: pd.DataFrame, path: Path, pair_label: str = "CL/BZ"
) -> None:
    """Refit beta by fold -- the auditable core of this module, analogous to
    ``backtest.plot_walkforward_threshold_stability`` but for the hedge ratio
    instead of the entry/exit thresholds."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(walkforward_table))
    ax.step(x, walkforward_table["fold_beta"], where="mid", marker="o", color="#2c7fb8", label="fold_beta")
    ax.set_xticks(list(x))
    ax.set_xticklabels(walkforward_table["test_start"], rotation=45, ha="right")
    ax.set_ylabel("beta")
    ax.set_title(f"{pair_label} walk-forward: refit hedge ratio by fold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_beta_refit_vs_static_equity(
    beta_refit_oos: pd.DataFrame, static_oos: pd.DataFrame, path: Path, pair_label: str = "CL/BZ"
) -> None:
    """The size of the gap this module closes, made visible: cumulative net
    PnL of the static-beta walk-forward next to the beta-refit walk-forward,
    over their common date range."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        beta_refit_oos.index,
        beta_refit_oos["net_cum_pnl_usd"],
        linewidth=1.6,
        label="walk-forward OOS, beta refit per fold",
    )
    ax.plot(
        static_oos.index,
        static_oos["net_cum_pnl_usd"],
        linewidth=1.2,
        linestyle="--",
        label="walk-forward OOS, static full-sample beta",
    )
    ax.set_ylabel("cumulative net PnL ($)")
    ax.set_title(f"{pair_label}: does refitting beta per fold change the walk-forward result?")
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
    panel = drop_sunday_session(load_panel())
    cost_config = CostConfig()
    reference_notional = reference_notional_usd(spread_frame, model)

    print("=== WALK-FORWARD, BETA REFIT PER FOLD ===")
    walkforward_table, oos_frame, _ = run_walkforward_beta_refit(
        panel, model.leg_y, model.leg_x, cost_config=cost_config, reference_notional=reference_notional
    )
    save_walkforward_beta(oos_frame)
    walkforward_table.to_csv(WALKFORWARD_BETA_TABLE_PATH, index=False)
    beta_refit_summary = summarize_backtest(oos_frame, reference_notional, cost_config=cost_config)
    print(beta_refit_summary.to_string())

    print("\n=== WALK-FORWARD, STATIC FULL-SAMPLE BETA (for comparison) ===")
    static_walkforward_table, static_oos_frame = run_walkforward(spread_frame, model, cost_config=cost_config)
    static_summary = summarize_backtest(static_oos_frame, reference_notional, cost_config=cost_config)
    print(static_summary.to_string())

    summary_table = pd.DataFrame(
        {
            "walkforward_oos_static_beta": static_summary,
            "walkforward_oos_beta_refit": beta_refit_summary,
        }
    ).T
    summary_table.to_csv(SUMMARY_BETA_PATH)

    plot_walkforward_beta_stability(walkforward_table, FIGURES_DIR / "14_cl_bz_walkforward_beta_stability.png")
    plot_beta_refit_vs_static_equity(
        oos_frame, static_oos_frame, FIGURES_DIR / "15_cl_bz_walkforward_beta_refit_equity.png"
    )

    print(f"\nWalk-forward (beta refit) -> {WALKFORWARD_BETA_PATH}")
    print(f"Table                     -> {WALKFORWARD_BETA_TABLE_PATH}")
    print(f"Summary comparison        -> {SUMMARY_BETA_PATH}")
    print(f"Figures                   -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
