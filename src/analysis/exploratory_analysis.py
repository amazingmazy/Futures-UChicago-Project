"""Exploratory data analysis and pair selection (issue #3).

Answers the question issue #3 poses: *which two futures contracts should we
model and potentially trade against each other?*

**Recommendation: CL/BZ (WTI vs Brent crude)** -- run on official CME
settlement prices. It dominates every candidate on every dimension that
matters, and it is the only pair whose cointegration survives out of the
2018-2021 regime:

* Engle-Granger and ADF p = 0.0000 on the full 2015-2026 sample, and the
  **only pair that stays under p = 0.05 on every subsample window**
  (2015/2018/2022/2023 starts: 0.0000 / 0.0003 / 0.0087 / 0.0192). Everything
  else -- including ZQ/SR3 -- fails from 2022 or 2023 onward.
* **Half-life 4.1 days**, the shortest in the table (ZQ/SR3: 12.4d).
* **beta = 0.97 with a tight rolling range (0.74-1.14)**, against ZQ/SR3's
  -0.46 to +2.40. A static hedge ratio is defensible here.
* Both legs quote in $/bbl on identical 1,000-barrel contracts, so the ~1:1
  spread that is stationary is *also* (approximately) the dollar-neutral one.
  The DV01 dilemma that plagues ZQ/SR3 (see below) never arises.
* Staleness under 1% on both legs; full 2,894-day sample.
* The economic story is one sentence: two grades of light sweet crude tied
  together by a physical shipping arbitrage.

**ZQ/SR3 (Fed Funds vs SOFR), the pair the README originally assumed, is the
runner-up** -- and on settlement prices its problems are worse than the
close-price analysis suggested. Its hazards, kept here because they document
why it lost:

1. **Cointegration decays out of sample.** EG p goes 0.0003 (full sample) ->
   0.0131 (from 2022) -> 0.2299 (from 2023). It is a claim about 2018-2021,
   not about today.

2. **The sample starts in May 2018, not 2015** (SOFR futures did not exist
   before then), so it has 2,052 usable days against CL/BZ's 2,894.

3. **Both legs are stale much of the time** (ZQ 57%, SR3 37% of settlements
   unchanged) because front-month short-rate contracts are pinned to a policy
   rate that only moves at eight scheduled FOMC meetings a year. The
   daily-change correlation is 0.01 against a level R-squared of 0.99.

4. **The stationary spread is the unhedged one.** ZQ and SR3 have different
   DV01s ($41.67 vs $25.00 per basis point), so the ~1:1 spread that is
   stationary carries net exposure to the *level* of rates, while the
   DV01-neutral spread (beta = 0.60) is not stationary (ADF p = 0.75,
   half-life 302 days). See ``compare_spread_constructions``. CL/BZ has no
   analogue of this problem.

The Treasury curve pairs (ZT/ZF, ZF/ZN, ZN/ZB) fail Engle-Granger outright
(p ~ 0.36-0.40 full-sample; ZN/ZB's marginal 0.046 collapses to 0.31-0.50 on
every later window -- an artifact of the quiet 2015-2017 period).

Run with::

    uv run python -m src.analysis.exploratory_analysis

Writes figures to ``outputs/figures/`` and tables to ``outputs/tables/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------

from src.data.panel import (  # noqa: E402
    FIGURES_DIR,
    PANEL_PATH,
    PROCESSED_DIR,
    ROOT_DIR,
    TABLES_DIR,
    drop_sunday_session,
    load_panel,
)

#: Roots quoted as ``100 - rate`` rather than as a price. Everything else in the
#: panel is a price (Treasury points, $/bbl, $/gal).
RATE_LIKE_ROOTS = frozenset({"ZQ", "SR3"})

#: Contract DV01 in USD per basis point, from the CME contract specs.
#: ZQ is a 30-day contract on $5m notional: 5e6 * 1e-4 * (30/360) = $41.67/bp.
#: SR3 is a 3-month contract on $1m notional: 1e6 * 1e-4 * (90/360) = $25.00/bp.
#: They differ, which is why a 1:1 spread is not risk-neutral. See
#: ``compare_spread_constructions``.
DV01_USD_PER_BP = {"ZQ": 41.67, "SR3": 25.00}

#: Candidate pairs from the shortlist in issue #3. Gold/silver and corn/wheat
#: were also listed there but are not in the ingested panel and sit outside the
#: interest-rate scope the README sets, so they are not evaluated. CL/HO is kept
#: as an out-of-scope control: it is useful to see what a pair of genuinely
#: active, non-policy-linked contracts looks like on the same statistics.
CANDIDATE_PAIRS: list[tuple[str, str]] = [
    ("ZQ", "SR3"),  # Fed Funds vs SOFR    -- the original recommendation
    ("ZT", "ZF"),  # 2Y vs 5Y             -- short-end curve
    ("ZF", "ZN"),  # 5Y vs 10Y            -- belly
    ("ZN", "ZB"),  # 10Y vs 30Y           -- long-end curve
    ("CL", "HO"),  # crude vs heating oil -- control
    ("CL", "BZ"),  # WTI vs Brent         -- same units ($/bbl), same 1,000bbl size
]

#: Subsample start dates for the cointegration robustness check. A pair whose
#: Engle-Granger p-value is only small on the full sample, and blows up on recent
#: subsamples, is not cointegrated in any regime we would trade -- it just
#: happens to have a quiet early period. This is what disqualifies ZN/ZB.
ROBUSTNESS_STARTS = ["2015-01-01", "2018-01-01", "2022-01-01", "2023-01-01"]

#: Rolling window for the hedge-ratio stability check, in trading days (~1 year).
ROLLING_WINDOW = 252

#: Basis points per unit of price for a ``100 - rate`` contract.
BP_PER_PRICE_POINT = 100.0


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------


def to_rate_space(series: pd.Series, root: str) -> pd.Series:
    """Convert a rate-like root to its implied rate; leave prices untouched.

    ZQ and SR3 quote ``100 - rate``. Comparing them to each other in price space
    happens to work, because they share the affine transform -- but the
    regression coefficient and the residual spread are only *interpretable* in
    rate space. Doing the conversion here means the ZQ/SR3 hedge ratio reads as
    "basis points of Fed Funds per basis point of SOFR", and the spread comes out
    in basis points: the unit a rates desk would actually quote.
    """
    return 100.0 - series if root in RATE_LIKE_ROOTS else series


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-root coverage: first and last observation, missing count, staleness.

    ``stale_frac`` -- the fraction of days on which the settlement did not
    change -- is the most diagnostic column here, and nothing in the ingestion
    step surfaces it. It explains why the ZQ/SR3 daily-change correlation is
    0.01 despite a level R-squared of 0.99, and why CL/BZ (both legs under 1%
    stale) is a cleaner pair to model.
    """
    rows = []
    for root in panel.columns:
        present = panel[root].dropna()
        rows.append(
            {
                "root": root,
                "n_obs": len(present),
                "first": present.index.min().date() if len(present) else None,
                "last": present.index.max().date() if len(present) else None,
                "n_missing": int(panel[root].isna().sum()),
                "pct_missing": panel[root].isna().mean(),
                # A zero daily change means the contract did not trade to a new
                # level. For a policy-linked contract between FOMC meetings that
                # is the norm, not a data error.
                "stale_frac": float((present.diff() == 0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("root")


# --------------------------------------------------------------------------
# Pair statistics
# --------------------------------------------------------------------------


@dataclass
class PairStats:
    """One row of the pair-ranking table."""

    pair: str
    n_obs: int
    start: str
    beta: float  # OLS hedge ratio: leg_a ~ const + beta * leg_b
    r_squared: float  # fit on levels -- high for anything that trends together
    change_corr: float  # correlation of daily *changes* -- the honest measure
    eg_pvalue: float  # Engle-Granger cointegration test
    adf_pvalue: float  # ADF on the OLS residual (the spread)
    half_life_days: float  # AR(1) mean-reversion half-life of the spread
    spread_std: float  # dispersion of the spread, in its natural units
    leg_stale_frac: float  # staleness of the worse of the two legs
    beta_min: float  # rolling-window hedge ratio, minimum
    beta_max: float  # rolling-window hedge ratio, maximum

    def as_row(self) -> dict:
        return asdict(self)


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life of a spread, from an AR(1) fit.

    Regress the daily change of the spread on its own lagged level::

        s_t - s_{t-1} = alpha + b * s_{t-1} + e_t

    If ``b < 0`` the series is pulled back toward its mean, and the half-life is
    ``-ln(2) / b``. A non-negative ``b`` means there is no mean reversion to
    measure, so return NaN rather than a meaningless negative number.
    """
    series = spread.dropna()
    lagged = series.shift(1)
    delta = series - lagged
    mask = lagged.notna() & delta.notna()
    if mask.sum() < 2:
        return float("nan")
    slope = sm.OLS(delta[mask], sm.add_constant(lagged[mask])).fit().params.iloc[1]
    return float("nan") if slope >= 0 else float(-np.log(2) / slope)


def rolling_beta(y: pd.Series, x: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Re-estimate the hedge ratio on a rolling window.

    A pair whose beta is flat supports a single static hedge ratio, which is what
    issue #4's baseline model assumes. On settlements, CL/BZ's rolling beta stays
    in a tight 0.74-1.14 band, so a static ratio is defensible. ZQ/SR3's swings
    from -0.46 to +2.40 -- it would need the time-varying/Bayesian extension the
    README lists as optional.
    """
    betas: list[float] = []
    index: list[pd.Timestamp] = []
    for i in range(window, len(y) + 1):
        y_window = y.iloc[i - window : i]
        x_window = x.iloc[i - window : i]
        # A degenerate window -- e.g. a policy rate pinned at zero for a whole
        # year, as in 2021 -- has no variation to regress against. Skip it rather
        # than dividing by something indistinguishable from zero.
        if x_window.std() < 1e-9:
            continue
        betas.append(float(sm.OLS(y_window, sm.add_constant(x_window)).fit().params.iloc[1]))
        index.append(y.index[i - 1])
    return pd.Series(betas, index=pd.DatetimeIndex(index), name="beta")


def analyse_pair(panel: pd.DataFrame, leg_a: str, leg_b: str) -> tuple[PairStats, pd.Series]:
    """Run the full statistical battery on one candidate pair.

    Returns the summary statistics and the spread itself (the OLS residual), so a
    caller can plot the spread without refitting.

    Both legs are moved into rate space first where applicable, so ``beta`` and
    the residual are in interpretable units.
    """
    both = panel[[leg_a, leg_b]].dropna()
    y = to_rate_space(both[leg_a], leg_a)
    x = to_rate_space(both[leg_b], leg_b)

    fit = sm.OLS(y, sm.add_constant(x)).fit()
    spread = fit.resid

    # Correlation of daily *changes*, not levels. Two series that both trend will
    # show a high level correlation whether or not they actually co-move, so a
    # level-based screen would rank almost everything as a good pair. The change
    # correlation is what reveals that ZQ and SR3 barely move together day to day
    # (0.01) even though their levels track almost perfectly.
    changes = both.diff().dropna()
    change_corr = float(changes[leg_a].corr(changes[leg_b]))

    rolling = rolling_beta(y, x)

    # Staleness of the *worse* leg: a pair is only as informative as its
    # least-active contract.
    leg_stale = max(
        float((both[leg_a].diff() == 0).mean()),
        float((both[leg_b].diff() == 0).mean()),
    )

    stats = PairStats(
        pair=f"{leg_a}/{leg_b}",
        n_obs=len(both),
        start=str(both.index.min().date()),
        beta=float(fit.params.iloc[1]),
        r_squared=float(fit.rsquared),
        change_corr=change_corr,
        eg_pvalue=float(coint(y, x)[1]),
        adf_pvalue=float(adfuller(spread, autolag="AIC")[1]),
        half_life_days=half_life(spread),
        spread_std=float(spread.std()),
        leg_stale_frac=leg_stale,
        beta_min=float(rolling.min()) if len(rolling) else float("nan"),
        beta_max=float(rolling.max()) if len(rolling) else float("nan"),
    )
    return stats, spread


def rank_pairs(
    panel: pd.DataFrame, pairs: list[tuple[str, str]] = CANDIDATE_PAIRS
) -> pd.DataFrame:
    """Run every candidate pair through the same battery and tabulate the result."""
    return pd.DataFrame([analyse_pair(panel, a, b)[0].as_row() for a, b in pairs]).set_index("pair")


def cointegration_robustness(
    panel: pd.DataFrame,
    pairs: list[tuple[str, str]] = CANDIDATE_PAIRS,
    starts: list[str] = ROBUSTNESS_STARTS,
) -> pd.DataFrame:
    """Re-run Engle-Granger on progressively later subsamples.

    This is the test that separates a real relationship from a lucky one, and
    it is what decides the recommendation. On settlement prices:

    CL/BZ:   0.0000 -> 0.0003 -> 0.0087 -> 0.0192   (passes every window)
    ZQ/SR3:  0.0003 -> 0.0003 -> 0.0131 -> 0.2299   (fails from 2023)
    ZN/ZB:   0.0463 -> 0.5030 -> 0.3151 -> 0.4880   (never had a relationship)
    ZT/ZF:   0.3808 -> 0.5035 -> 0.7350 -> 0.6993   (never close)

    Three different things are going on. ZN/ZB's marginal full-sample pass is
    carried entirely by the quiet 2015-2017 window and collapses the moment
    that window is dropped. ZQ/SR3 *did* have a strong relationship and it has
    weakened -- what one would expect after the 2022-23 hiking cycle repriced
    the front end. CL/BZ holds on every window, because the relationship is
    enforced by a physical arbitrage (shipping crude across the Atlantic)
    rather than by a policy regime.

    Note that the later windows are shorter and so have less power; some
    p-value drift is expected mechanically. What matters is staying on the
    right side of 0.05, and only CL/BZ does.
    """
    rows = []
    for leg_a, leg_b in pairs:
        row: dict[str, object] = {"pair": f"{leg_a}/{leg_b}"}
        for start in starts:
            both = panel.loc[start:, [leg_a, leg_b]].dropna()
            if len(both) < ROLLING_WINDOW:
                row[f"eg_p_from_{start[:4]}"] = float("nan")
                continue
            y = to_rate_space(both[leg_a], leg_a)
            x = to_rate_space(both[leg_b], leg_b)
            row[f"eg_p_from_{start[:4]}"] = float(coint(y, x)[1])
        rows.append(row)
    return pd.DataFrame(rows).set_index("pair")


def compare_spread_constructions(panel: pd.DataFrame) -> pd.DataFrame:
    """Three ways to build the ZQ/SR3 spread, and the trade-off between them.

    This is the central finding of issue #3, and the thing issue #4 most needs to
    know before it estimates anything.

    * **raw_1to1** -- one ZQ against one SR3. Stationary (ADF p = 0.0000),
      half-life about 12 days. But the legs have different DV01s, so this book is
      long or short the *level* of rates: it is not market-neutral.
    * **ols_beta** -- beta = 0.986, essentially 1:1. Same properties and the same
      problem. The regression recovers the unhedged ratio precisely because that
      is the combination that happens to be stationary.
    * **dv01_neutral** -- beta = 0.60, the ratio that actually neutralises rate
      exposure. And it is **not stationary**: ADF p = 0.75, half-life 302 days.

    So the spread that mean-reverts is not hedged, and the spread that is hedged
    does not mean-revert. No construction gives both. Whichever issue #4 picks, it
    should pick knowingly: a strategy on the 1:1 spread is taking directional rate
    risk it has not accounted for, and a strategy on the DV01-neutral spread has
    no mean reversion to trade.

    Note: this dilemma is specific to ZQ/SR3. For CL/BZ the question does not
    arise -- both legs are 1,000 barrels quoted in $/bbl, so the 1:1 spread and
    the dollar-neutral spread are the same object.
    """
    both = panel[["ZQ", "SR3"]].dropna()
    zq_bp = to_rate_space(both["ZQ"], "ZQ") * BP_PER_PRICE_POINT
    sr3_bp = to_rate_space(both["SR3"], "SR3") * BP_PER_PRICE_POINT

    ols_beta = float(sm.OLS(zq_bp, sm.add_constant(sr3_bp)).fit().params.iloc[1])
    dv01_beta = DV01_USD_PER_BP["SR3"] / DV01_USD_PER_BP["ZQ"]

    constructions = {
        "raw_1to1": (1.0, zq_bp - sr3_bp),
        "ols_beta": (ols_beta, zq_bp - ols_beta * sr3_bp),
        "dv01_neutral": (dv01_beta, zq_bp - dv01_beta * sr3_bp),
    }

    rows = []
    for name, (beta, spread) in constructions.items():
        rows.append(
            {
                "construction": name,
                "beta": beta,
                "spread_std_bp": float(spread.std()),
                "adf_pvalue": float(adfuller(spread.dropna(), autolag="AIC")[1]),
                "half_life_days": half_life(spread),
                "is_dv01_neutral": bool(np.isclose(beta, dv01_beta)),
            }
        )
    return pd.DataFrame(rows).set_index("construction")


def spread_volatility_by_year(
    panel: pd.DataFrame,
    leg_a: str = "ZQ",
    leg_b: str = "SR3",
    scale: float = BP_PER_PRICE_POINT,
) -> pd.Series:
    """Standard deviation of a pair's spread, year by year.

    Included because a static z-score threshold -- the entry rule issue #5
    proposes -- assumes the spread's dispersion is roughly constant. For ZQ/SR3
    it is not: 2021 (ZIRP, both legs frozen) has a standard deviation of ~1.5bp
    against ~40bp for 2022 (the hiking cycle). A ``|z| > 2`` rule calibrated
    on the full sample will essentially never fire in 2021 and will fire
    constantly in 2022. Issue #5 should use a rolling mean and standard
    deviation, not full-sample ones.

    ``scale`` converts the spread into its quoted unit: basis points for the
    rate-like default, 1.0 for a pair already in price units (e.g. CL/BZ in
    $/bbl).
    """
    _, spread = analyse_pair(panel, leg_a, leg_b)
    scaled = spread * scale
    return scaled.groupby(scaled.index.year).std().rename("spread_std")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_normalised_levels(panel: pd.DataFrame, path: Path) -> None:
    """All roots on one axis, rebased to 100 at each root's first observation.

    Rebasing is the only honest way to put a $/gal heating-oil series and a
    Treasury price series on the same chart. It also makes SR3's May-2018 start
    immediately visible, which is half the point of the figure.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for root in panel.columns:
        series = panel[root].dropna()
        if series.empty:
            continue
        ax.plot(series.index, 100 * series / series.iloc[0], label=root, linewidth=1.1)
    ax.set_title("Continuous front-month settlements, rebased to 100 at each root's first observation")
    ax.set_ylabel("Index (first observation = 100)")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_change_correlation(panel: pd.DataFrame, path: Path) -> None:
    """Correlation heatmap of *daily changes* across all roots.

    Deliberately on changes rather than levels. A level-correlation matrix on
    trending series is dominated by the shared trend and shows everything as
    correlated with everything, which is useless as a screen.
    """
    corr = panel.diff().corr()
    values = corr.to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns)
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(values[i, j]) > 0.55 else "black",
            )
    ax.set_title("Correlation of daily changes")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_staleness(coverage: pd.DataFrame, path: Path) -> None:
    """Fraction of days on which each root's settlement did not move.

    The single most important diagnostic in this analysis. ZQ and SR3 sit at 57%
    and 37%; every other root is under 5%. This is why the ZQ/SR3 daily-change
    correlation is 0.01 despite a level R-squared of 0.99, and it is a big part
    of why the recommendation moved away from ZQ/SR3 to CL/BZ.
    """
    ordered = coverage["stale_frac"].sort_values(ascending=False)
    colours = ["#c0392b" if value > 0.25 else "#2c7fb8" for value in ordered]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ordered.index, ordered.to_numpy(), color=colours)
    ax.axhline(0.25, color="grey", linestyle="--", linewidth=1)
    ax.text(
        len(ordered) - 0.4,
        0.265,
        "25%: above this, much of the sample carries no information",
        ha="right",
        fontsize=8,
        color="grey",
    )
    ax.set_ylabel("Fraction of days with zero price change")
    ax.set_title("Policy-linked contracts barely move between FOMC meetings")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_spread_constructions(panel: pd.DataFrame, path: Path) -> None:
    """The three ZQ/SR3 spreads on one axis -- the trade-off, made visible.

    The 1:1 and OLS spreads sit flat around zero and revert within days. The
    DV01-neutral spread wanders across hundreds of basis points and does not come
    back. Same two contracts, different weights, entirely different object.
    """
    both = panel[["ZQ", "SR3"]].dropna()
    zq_bp = to_rate_space(both["ZQ"], "ZQ") * BP_PER_PRICE_POINT
    sr3_bp = to_rate_space(both["SR3"], "SR3") * BP_PER_PRICE_POINT

    ols_beta = float(sm.OLS(zq_bp, sm.add_constant(sr3_bp)).fit().params.iloc[1])
    dv01_beta = DV01_USD_PER_BP["SR3"] / DV01_USD_PER_BP["ZQ"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        both.index,
        (zq_bp - sr3_bp).to_numpy(),
        label="raw 1:1 -- stationary, NOT hedged",
        linewidth=1.0,
        color="#2c7fb8",
    )
    ax.plot(
        both.index,
        (zq_bp - ols_beta * sr3_bp).to_numpy(),
        label=f"OLS beta = {ols_beta:.2f} -- stationary, NOT hedged",
        linewidth=1.0,
        color="#31a354",
        linestyle="--",
    )
    ax.plot(
        both.index,
        (zq_bp - dv01_beta * sr3_bp).to_numpy(),
        label=f"DV01-neutral beta = {dv01_beta:.2f} -- hedged, NOT stationary",
        linewidth=1.2,
        color="#c0392b",
    )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("Spread (basis points)")
    ax.set_title("ZQ/SR3: the spread that mean-reverts is not the spread that is hedged")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_spread_with_zscore(
    spread: pd.Series,
    path: Path,
    units: str = "bp",
    title: str = "ZQ/SR3 spread (OLS residual, rate space) and rolling z-score",
) -> None:
    """A pair's spread and its z-score -- what issues #4 and #5 consume.

    The z-score uses a rolling window rather than the full-sample mean and
    standard deviation, because the spread's dispersion is not stable (see
    ``spread_volatility_by_year``). A full-sample z-score would be dominated by
    2022 and would essentially never fire in 2021.
    """
    window = ROLLING_WINDOW // 2
    z_score = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax_top.plot(spread.index, spread.to_numpy(), linewidth=0.9, color="#2c3e50")
    ax_top.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax_top.set_ylabel(f"Spread ({units})")
    ax_top.set_title(title)
    ax_top.grid(alpha=0.25)

    ax_bottom.plot(z_score.index, z_score.to_numpy(), linewidth=0.9, color="#2c7fb8")
    for level in (-2, 2):
        ax_bottom.axhline(level, color="#c0392b", linestyle="--", linewidth=1)
    ax_bottom.axhline(0, color="grey", linewidth=0.8)
    ax_bottom.set_ylabel(f"z-score ({window}d rolling)")
    ax_bottom.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_rolling_beta(panel: pd.DataFrame, pairs: list[tuple[str, str]], path: Path) -> None:
    """Rolling hedge ratio for every pair -- the stability check.

    A flat line supports a single static hedge ratio. CL/BZ's line is close to
    flat (0.74-1.14); ZQ/SR3's swings from -0.46 to +2.40, the argument for a
    time-varying model had that pair been chosen.

    CL/HO's beta is around 23 (crude in $/bbl against heating oil in $/gal) and
    would compress every other line to a flat streak, so the y-axis is clipped.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for leg_a, leg_b in pairs:
        both = panel[[leg_a, leg_b]].dropna()
        y = to_rate_space(both[leg_a], leg_a)
        x = to_rate_space(both[leg_b], leg_b)
        rolling = rolling_beta(y, x)
        if not rolling.empty:
            ax.plot(rolling.index, rolling.to_numpy(), label=f"{leg_a}/{leg_b}", linewidth=1.2)
    ax.set_ylim(-0.6, 2.6)  # clips CL/HO (beta ~ 23); see docstring
    ax.set_title(f"Rolling {ROLLING_WINDOW}-day OLS hedge ratio (y-axis clipped; CL/HO is off-scale)")
    ax.set_ylabel("beta")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    raw_panel = load_panel()
    panel = drop_sunday_session(raw_panel)
    print(
        f"Panel: {raw_panel.shape[0]} rows -> {panel.shape[0]} after dropping "
        f"{raw_panel.shape[0] - panel.shape[0]} Sunday-session rows"
    )

    coverage = coverage_report(panel)
    coverage.to_csv(TABLES_DIR / "coverage.csv")
    print("\n=== COVERAGE AND STALENESS ===")
    print(coverage.to_string(float_format=lambda v: f"{v:.3f}"))

    ranking = rank_pairs(panel)
    ranking.to_csv(TABLES_DIR / "pair_ranking.csv")
    print("\n=== PAIR RANKING ===")
    print(ranking.to_string(float_format=lambda v: f"{v:.4f}"))

    robustness = cointegration_robustness(panel)
    robustness.to_csv(TABLES_DIR / "cointegration_robustness.csv")
    print("\n=== ENGLE-GRANGER p-VALUE BY SUBSAMPLE START ===")
    print(robustness.to_string(float_format=lambda v: f"{v:.4f}"))
    print("  NOTE: CL/BZ is the only pair that stays under p = 0.05 on every window.")
    print("  ZQ/SR3 fails from 2023 (p = 0.23): its cointegration is a claim about")
    print("  2018-2021, not about today.")

    constructions = compare_spread_constructions(panel)
    constructions.to_csv(TABLES_DIR / "spread_constructions.csv")
    print("\n=== ZQ/SR3 SPREAD CONSTRUCTIONS ===")
    print(constructions.to_string(float_format=lambda v: f"{v:.4f}"))
    print("  The stationary spread is not hedged; the hedged spread is not stationary.")

    yearly = spread_volatility_by_year(panel)
    yearly.to_csv(TABLES_DIR / "spread_vol_by_year.csv")
    print("\n=== ZQ/SR3 SPREAD STANDARD DEVIATION BY YEAR (bp) ===")
    print(yearly.to_string(float_format=lambda v: f"{v:.2f}"))
    print("  ~1.5bp in 2021 against ~40bp in 2022: a static z-score threshold will not work.")

    yearly_cl_bz = spread_volatility_by_year(panel, "CL", "BZ", scale=1.0)
    yearly_cl_bz.to_csv(TABLES_DIR / "spread_vol_by_year_cl_bz.csv")
    print("\n=== CL/BZ SPREAD STANDARD DEVIATION BY YEAR ($/bbl) ===")
    print(yearly_cl_bz.to_string(float_format=lambda v: f"{v:.2f}"))

    plot_normalised_levels(panel, FIGURES_DIR / "01_levels_rebased.png")
    plot_change_correlation(panel, FIGURES_DIR / "02_change_correlation.png")
    plot_staleness(coverage, FIGURES_DIR / "03_staleness.png")
    plot_rolling_beta(panel, CANDIDATE_PAIRS, FIGURES_DIR / "04_rolling_beta.png")
    plot_spread_constructions(panel, FIGURES_DIR / "05_spread_constructions.png")

    _, spread = analyse_pair(panel, "ZQ", "SR3")
    plot_spread_with_zscore(spread * BP_PER_PRICE_POINT, FIGURES_DIR / "06_spread_zq_sr3.png")

    _, spread_cl_bz = analyse_pair(panel, "CL", "BZ")
    plot_spread_with_zscore(
        spread_cl_bz,
        FIGURES_DIR / "07_spread_cl_bz.png",
        units="$/bbl",
        title="CL/BZ spread (OLS residual, price space) and rolling z-score",
    )

    print(f"\nFigures -> {FIGURES_DIR}")
    print(f"Tables   -> {TABLES_DIR}")
    print("\nRECOMMENDATION: CL/BZ (WTI vs Brent) -- the only pair cointegrated on every")
    print("subsample window, shortest half-life (4.1d), beta ~ 1, no DV01 dilemma.")
    print("See docs/pair_selection.md for the full comparison and the ZQ/SR3 runner-up.")


if __name__ == "__main__":
    main()
