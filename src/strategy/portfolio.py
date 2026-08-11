"""Does trading ZQ/SR3 alongside CL/BZ actually diversify anything?

Combines the two pairs' already-out-of-sample walk-forward PnL streams
(``src.strategy.walkforward_beta``'s beta-refit OOS series, for both legs of
the comparison, so neither pair gets an unfair static-beta advantage) into
one equal-risk-weighted portfolio, and reports whether the combination has a
better risk-adjusted profile than either pair traded alone.

This module makes no attempt to make ZQ/SR3 a better strategy -- it stays
exactly as flawed as ``docs/pair_selection.md``/``src.strategy.run_zq_sr3``
already documented. The question here is narrower: given two OOS PnL
streams that already exist, does combining them (with no information fed
back into either pair's own signal generation) reduce risk the way
uncorrelated strategies are supposed to.

Design choices, stated explicitly rather than glossed over:

* **Evaluation window = intersection, not union.** ZQ/SR3's usable sample is
  much shorter than CL/BZ's (see the coverage gap documented in
  ``src.strategy.run_zq_sr3``). Zero-filling the non-overlapping stub would
  overstate any diversification benefit during a period where only one
  strategy is actually trading.
* **Equal-risk weighting, computed once, not walk-forward.** Weights are set
  from each strategy's own already-OOS daily PnL std over the common window
  -- legitimate, since it only combines two finished OOS return streams and
  feeds nothing back into either pair's signal generation, but it is *not
  itself* walk-forward. A true expanding-window portfolio-weight
  re-estimation is natural future work, out of scope here.
* **No portfolio-level reference notional.** Each pair has its own
  contract/DV01-implied notional (see ``src.strategy.backtest``'s
  ``contract_size`` parameter); forcing them into one normalized "%% return"
  would need a third, even more arbitrary reference. Metrics here are
  reported in raw dollars and via Sharpe (already notional-free).

Run with::

    uv run python -m src.strategy.portfolio

(after ``src.strategy.walkforward_beta`` and ``src.strategy.run_zq_sr3`` have
both been run, so their OOS artifacts exist.)

Outputs:

* ``outputs/tables/portfolio_combination.csv`` -- ``cl_bz_alone``,
  ``zq_sr3_alone``, ``combined_equal_risk`` rows plus the PnL correlation.
* ``outputs/figures/22_portfolio_combined_equity.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever write PNGs, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from src.data.panel import FIGURES_DIR, TABLES_DIR
from src.strategy.backtest import TRADING_DAYS_PER_YEAR, sharpe_ratio
from src.strategy.run_zq_sr3 import WALKFORWARD_BETA_PATH as ZQ_SR3_WALKFORWARD_BETA_PATH
from src.strategy.walkforward_beta import WALKFORWARD_BETA_PATH as CL_BZ_WALKFORWARD_BETA_PATH
from src.strategy.walkforward_beta import load_walkforward_beta

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PORTFOLIO_TABLE_PATH = TABLES_DIR / "portfolio_combination.csv"


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------


def combine_strategies(
    pnl_series: dict[str, pd.Series], target_vol: float | None = None
) -> tuple[pd.Series, dict[str, float]]:
    """Equal-risk-weighted combination of two or more daily PnL series.

    Restricts to the intersection of every series' index -- not the union --
    so a period where only one strategy has data never gets zero-filled into
    an artificially calmer (or more diversified-looking) stretch.

    Weights are ``target_vol / std_i``, computed once from each series' own
    standard deviation over the common window. If ``target_vol`` is not
    given, it defaults to the first series' own std, so that series gets
    weight 1.0 and every other series is scaled relative to it.
    """
    names = list(pnl_series.keys())
    if len(names) < 2:
        raise ValueError("combine_strategies needs at least two strategies")

    common_index = pnl_series[names[0]].index
    for name in names[1:]:
        common_index = common_index.intersection(pnl_series[name].index)
    if len(common_index) == 0:
        raise ValueError("no overlapping dates between the given strategies")

    common = {name: series.loc[common_index] for name, series in pnl_series.items()}
    stds = {name: float(series.std()) for name, series in common.items()}
    if target_vol is None:
        target_vol = stds[names[0]]

    weights = {name: (target_vol / std if std else 0.0) for name, std in stds.items()}
    combined = sum(weights[name] * common[name] for name in names)
    combined.name = "combined_net_pnl_usd"
    return combined, weights


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _dollar_drawdown(cum_pnl: pd.Series) -> float:
    """Worst peak-to-trough dollar drawdown of a cumulative PnL series.

    Deliberately not ``backtest.max_drawdown`` (a *percentage* drawdown that
    requires a positive equity-level baseline) -- there is no single
    reference notional at the portfolio level, so this reports dollars off
    a $0 starting point instead.
    """
    if len(cum_pnl) == 0:
        return float("nan")
    running_max = cum_pnl.cummax()
    return float((cum_pnl - running_max).min())


def summarize_pnl_series(pnl: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """The shared metrics record for a raw dollar PnL series with no
    associated reference notional."""
    n_sessions = len(pnl)
    cum_pnl = pnl.cumsum()
    return pd.Series(
        {
            "sample_start": str(pnl.index.min().date()) if n_sessions else "",
            "sample_end": str(pnl.index.max().date()) if n_sessions else "",
            "n_sessions": n_sessions,
            "total_pnl_usd": float(pnl.sum()) if n_sessions else float("nan"),
            "annualized_pnl_usd": float(pnl.mean() * periods_per_year) if n_sessions else float("nan"),
            "annualized_vol_usd": float(pnl.std() * (periods_per_year**0.5)) if n_sessions > 1 else float("nan"),
            "sharpe": sharpe_ratio(pnl, periods_per_year),
            "max_drawdown_usd": _dollar_drawdown(cum_pnl),
        }
    )


def portfolio_diagnostics(pnl_a: pd.Series, pnl_b: pd.Series, combined: pd.Series) -> pd.Series:
    """Correlation of the two source strategies plus a side-by-side Sharpe
    comparison, all over ``combined``'s (already-intersected) date range."""
    a = pnl_a.reindex(combined.index)
    b = pnl_b.reindex(combined.index)
    correlation = float(a.corr(b)) if len(combined) > 1 else float("nan")
    return pd.Series(
        {
            "n_common_sessions": len(combined),
            "pnl_correlation": correlation,
            "sharpe_a_alone": sharpe_ratio(a),
            "sharpe_b_alone": sharpe_ratio(b),
            "sharpe_combined": sharpe_ratio(combined),
        }
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_combined_equity(
    pnl_a: pd.Series,
    pnl_b: pd.Series,
    combined: pd.Series,
    weights: dict[str, float],
    path: Path,
    label_a: str = "CL/BZ",
    label_b: str = "ZQ/SR3",
) -> None:
    """Combined equity next to each pair held alone, both at the *same*
    equal-risk-weighted scale used in the combination.

    Plotting raw, unweighted PnL for the "alone" lines would make ZQ/SR3
    (natural dollar scale a couple hundred dollars) invisible next to CL/BZ
    (tens of thousands) -- true, but uninformative. Scaling both by the
    weights already used in ``combined`` answers the actually interesting
    question: at the position size each pair would need to match the
    other's risk, does holding both together beat holding either alone.
    """
    names = list(weights.keys())
    a = (weights[names[0]] * pnl_a.reindex(combined.index)).cumsum()
    b = (weights[names[1]] * pnl_b.reindex(combined.index)).cumsum()
    combined_cum = combined.cumsum()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(combined_cum.index, combined_cum, linewidth=1.6, label="combined (equal-risk weighted)")
    ax.plot(a.index, a, linewidth=1.0, linestyle="--", label=f"{label_a} alone (same risk-scaled weight)")
    ax.plot(b.index, b, linewidth=1.0, linestyle=":", label=f"{label_b} alone (same risk-scaled weight)")
    ax.set_ylabel("cumulative net PnL ($), equal-risk-weighted scale")
    ax.set_title(f"{label_a} + {label_b}: does combining reduce risk?")
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

    cl_bz = load_walkforward_beta(CL_BZ_WALKFORWARD_BETA_PATH)["net_pnl_usd"]
    zq_sr3 = load_walkforward_beta(ZQ_SR3_WALKFORWARD_BETA_PATH)["net_pnl_usd"]

    combined, weights = combine_strategies({"cl_bz": cl_bz, "zq_sr3": zq_sr3})
    diagnostics = portfolio_diagnostics(cl_bz, zq_sr3, combined)

    print("=== PORTFOLIO COMBINATION (equal-risk weighted, common OOS window) ===")
    print(f"weights: {weights}")
    print(diagnostics.to_string())

    summary_table = pd.DataFrame(
        {
            "cl_bz_alone": summarize_pnl_series(cl_bz.reindex(combined.index)),
            "zq_sr3_alone": summarize_pnl_series(zq_sr3.reindex(combined.index)),
            "combined_equal_risk": summarize_pnl_series(combined),
        }
    ).T
    summary_table.to_csv(PORTFOLIO_TABLE_PATH)

    plot_combined_equity(cl_bz, zq_sr3, combined, weights, FIGURES_DIR / "22_portfolio_combined_equity.png")

    print(f"\nTable   -> {PORTFOLIO_TABLE_PATH}")
    print(f"Figure  -> {FIGURES_DIR / '22_portfolio_combined_equity.png'}")
    print("\nSee docs/second_pair_and_portfolio.md for the write-up.")


if __name__ == "__main__":
    main()
