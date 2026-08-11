"""Second-pair comparison: run ZQ/SR3 (Fed Funds vs SOFR) through the same
spread-model -> signals -> backtest pipeline built for CL/BZ.

``docs/pair_selection.md`` picked CL/BZ over ZQ/SR3 for good reason -- ZQ/SR3
is the runner-up, and its documented weaknesses are real: cointegration
decays out of sample (Engle-Granger p 0.0003 -> 0.2299 from 2015 to 2023),
48%/37% staleness, an unresolved "DV01 dilemma" (the ~1:1 spread isn't
rate-neutral; the rate-neutral spread isn't stationary), and a rolling hedge
ratio that swings from -0.46 to +2.40. This script does **not** try to fix
any of that. It exists to answer a different question -- issue #6's "final
stretch" ask of whether trading a second, admittedly shakier pair alongside
CL/BZ helps a combined portfolio (see ``src.strategy.portfolio`` and
``docs/second_pair_and_portfolio.md``) -- and to demonstrate that already
pair-agnostic pipeline code needs no duplication to answer it.

Three things this script does differently from a copy-paste of the CL/BZ
scripts:

* **Rate space, not price space.** ZQ and SR3 quote ``100 - rate``.
  ``src.analysis.exploratory_analysis.to_rate_space`` (already used for the
  issue #3 ranking table) converts both legs before fitting, so this script's
  alpha/beta/half-life numbers are directly comparable to that table instead
  of silently redefining units.
* **A DV01 reference, not a bbl reference.** ``CONTRACT_SIZE_BBL`` (1,000
  bbl, $/bbl) means nothing for a rate pair. This uses ZQ's own contract
  DV01 ($41.67/bp, matching the position-sizing convention of "+1 leg_y
  contract" already built into ``generate_signals``) as the reference
  dollar-per-point multiplier. This does **not** resolve the DV01 dilemma --
  SR3's DV01 is $25.00/bp, so the position is still not rate-neutral -- it
  only lets Sharpe/drawdown/correlation be computed and compared on a
  consistent units choice. See ``docs/second_pair_and_portfolio.md``.
* **Beta-refit walk-forward, not the static-beta version.** Given ZQ/SR3's
  own rolling beta is documented as unstable, reporting its walk-forward
  headline off a single full-sample-fit beta would be misleading for
  exactly the pair whose hedge-ratio instability is its flagged weakness.

Run with::

    uv run python -m src.strategy.run_zq_sr3

Outputs (paths suffixed so nothing collides with the CL/BZ artifacts):

* ``data/processed/spread_zq_sr3.parquet``, ``spread_model_zq_sr3.json``
* ``data/processed/signals_zq_sr3.parquet``
* ``data/processed/backtest_zq_sr3.parquet``
* ``data/processed/walkforward_beta_refit_zq_sr3.parquet``
* ``outputs/tables/backtest_grid_zq_sr3.csv``,
  ``backtest_walkforward_zq_sr3.csv``, ``backtest_summary_zq_sr3.csv``
* ``outputs/figures/16_zq_sr3_fit.png``, ``17_zq_sr3_rolling_beta.png``,
  ``18_zq_sr3_signals.png``, ``19_zq_sr3_threshold_grid_heatmap.png``,
  ``20_zq_sr3_walkforward_beta_stability.png``,
  ``21_zq_sr3_walkforward_equity.png``
"""

from __future__ import annotations

import pandas as pd

from src.analysis.exploratory_analysis import DV01_USD_PER_BP, to_rate_space
from src.data.panel import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR, drop_sunday_session, load_panel
from src.models.spread import fit_spread_model, plot_fit_scatter, plot_rolling_beta_vs_static, save_model, save_spread
from src.strategy.backtest import (
    CostConfig,
    build_grid_frames,
    plot_threshold_heatmap,
    plot_walkforward_equity,
    reference_notional_usd,
    run_backtest,
    save_backtest,
    summarize_backtest,
    summarize_grid,
)
from src.strategy.signals import SignalConfig, generate_signals, plot_signals, save_signals, summarize_signals
from src.strategy.walkforward_beta import (
    plot_walkforward_beta_stability,
    run_walkforward_beta_refit,
    save_walkforward_beta,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

LEG_Y = "ZQ"
LEG_X = "SR3"
PAIR_LABEL = "ZQ/SR3"

#: ZQ's contract DV01 in $/bp (see module docstring). Used as the reference
#: dollar-per-point multiplier for the whole spread -- an approximation, not
#: a resolution of this pair's DV01 mismatch.
CONTRACT_SIZE = DV01_USD_PER_BP["ZQ"]

SPREAD_PATH = PROCESSED_DIR / "spread_zq_sr3.parquet"
MODEL_PATH = PROCESSED_DIR / "spread_model_zq_sr3.json"
SIGNALS_PATH = PROCESSED_DIR / "signals_zq_sr3.parquet"
BACKTEST_PATH = PROCESSED_DIR / "backtest_zq_sr3.parquet"
WALKFORWARD_BETA_PATH = PROCESSED_DIR / "walkforward_beta_refit_zq_sr3.parquet"

SIGNAL_SUMMARY_PATH = TABLES_DIR / "signal_summary_zq_sr3.csv"
GRID_PATH = TABLES_DIR / "backtest_grid_zq_sr3.csv"
WALKFORWARD_PATH = TABLES_DIR / "backtest_walkforward_zq_sr3.csv"
SUMMARY_PATH = TABLES_DIR / "backtest_summary_zq_sr3.csv"


def load_rate_space_panel() -> pd.DataFrame:
    """The settlement panel with ZQ/SR3 converted from ``100 - rate`` quotes
    into rate space, matching ``docs/pair_selection.md``'s convention."""
    panel = drop_sunday_session(load_panel())
    return panel.assign(
        **{
            LEG_Y: to_rate_space(panel[LEG_Y], LEG_Y),
            LEG_X: to_rate_space(panel[LEG_X], LEG_X),
        }
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_rate_space_panel()

    model, spread_frame = fit_spread_model(panel, LEG_Y, LEG_X)
    save_model(model, MODEL_PATH)
    save_spread(spread_frame, SPREAD_PATH)
    print("=== ZQ/SR3 SPREAD MODEL (static OLS, rate space) ===")
    print(pd.Series(model.to_dict(), name="value").to_string())
    plot_fit_scatter(spread_frame, model, FIGURES_DIR / "16_zq_sr3_fit.png", unit_label="rate, %")
    plot_rolling_beta_vs_static(spread_frame, model, FIGURES_DIR / "17_zq_sr3_rolling_beta.png")

    config = SignalConfig()
    signals = generate_signals(spread_frame, model, config)
    save_signals(signals, SIGNALS_PATH)
    summarize_signals(signals, config).to_csv(SIGNAL_SUMMARY_PATH, header=True)
    plot_signals(signals, config, FIGURES_DIR / "18_zq_sr3_signals.png", model)

    cost_config = CostConfig()
    reference_notional = reference_notional_usd(spread_frame, model, CONTRACT_SIZE)

    baseline = run_backtest(signals, model, cost_config, CONTRACT_SIZE)
    save_backtest(baseline, BACKTEST_PATH)
    print("\n=== ZQ/SR3 BASELINE BACKTEST (entry_z=1.5, exit_z=0.5) ===")
    print(summarize_backtest(baseline, reference_notional, cost_config=cost_config).to_string())

    print("\nBuilding threshold grid ...")
    grid_frames = build_grid_frames(spread_frame, model, cost_config=cost_config, contract_size=CONTRACT_SIZE)
    grid_summary = summarize_grid(grid_frames, reference_notional, cost_config)
    grid_summary.to_csv(GRID_PATH, index=False)
    best_row = grid_summary.sort_values("sharpe_net", ascending=False).iloc[0]
    best_entry_z, best_exit_z = float(best_row["entry_z"]), float(best_row["exit_z"])
    best_insample = grid_frames[(best_entry_z, best_exit_z)]
    plot_threshold_heatmap(
        grid_summary, FIGURES_DIR / "19_zq_sr3_threshold_grid_heatmap.png", pair_label=PAIR_LABEL
    )

    print(
        "\nRunning walk-forward with beta refit per fold (ZQ/SR3's rolling "
        "beta is documented as unstable -- see docs/pair_selection.md) ..."
    )
    walkforward_table, oos_frame, _ = run_walkforward_beta_refit(
        panel,
        LEG_Y,
        LEG_X,
        cost_config=cost_config,
        reference_notional=reference_notional,
        contract_size=CONTRACT_SIZE,
    )
    walkforward_table.to_csv(WALKFORWARD_PATH, index=False)
    save_walkforward_beta(oos_frame, WALKFORWARD_BETA_PATH)
    walkforward_summary = summarize_backtest(oos_frame, reference_notional, cost_config=cost_config)
    print(walkforward_summary.to_string())

    plot_walkforward_beta_stability(
        walkforward_table, FIGURES_DIR / "20_zq_sr3_walkforward_beta_stability.png", pair_label=PAIR_LABEL
    )
    # April 2020 (negative WTI) has no meaning for a rates pair.
    plot_walkforward_equity(
        oos_frame,
        baseline,
        best_insample,
        FIGURES_DIR / "21_zq_sr3_walkforward_equity.png",
        pair_label=PAIR_LABEL,
        highlight_date=None,
    )

    summary_table = pd.DataFrame(
        {
            "baseline_default": summarize_backtest(baseline, reference_notional, cost_config=cost_config),
            "best_insample": summarize_backtest(best_insample, reference_notional, cost_config=cost_config),
            "walkforward_oos_beta_refit": walkforward_summary,
        }
    ).T
    summary_table.to_csv(SUMMARY_PATH)

    print(f"\nSpread             -> {SPREAD_PATH}")
    print(f"Signals            -> {SIGNALS_PATH}")
    print(f"Backtest           -> {BACKTEST_PATH}")
    print(f"Walk-forward       -> {WALKFORWARD_BETA_PATH}")
    print(f"Tables             -> {TABLES_DIR}")
    print(f"Figures            -> {FIGURES_DIR}")
    print("\nSee docs/second_pair_and_portfolio.md for the write-up, and")
    print("src.strategy.portfolio for combining this with CL/BZ.")


if __name__ == "__main__":
    main()
