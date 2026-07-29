"""Tests for the pair-selection analysis (issue #3).

Every test here runs on **synthetic** data. Nothing touches Databento and
nothing needs an API key, so ``uv run pytest`` works on a fresh clone. This is
the pattern the assignment brief suggested for breaking the sequencing
dependency between issues.

The synthetic fixtures are built to have known properties, so the tests check
that the statistics actually detect what they claim to detect, rather than just
checking that the functions return without raising:

* ``cointegrated_pair`` -- two random walks sharing a common stochastic trend,
  plus a stationary noise term. Genuinely cointegrated by construction.
* ``independent_pair`` -- two unrelated random walks. Not cointegrated by
  construction, no matter how nice the level correlation happens to look on any
  given draw.
* ``stale_pair`` -- a cointegrated pair where one leg only updates every tenth
  day, mimicking a policy-linked contract between FOMC meetings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.exploratory_analysis import (
    RATE_LIKE_ROOTS,
    analyse_pair,
    cointegration_robustness,
    compare_spread_constructions,
    coverage_report,
    drop_sunday_session,
    half_life,
    rank_pairs,
    rolling_beta,
    spread_volatility_by_year,
    to_rate_space,
)

RNG_SEED = 20260712
N_DAYS = 1500


def _business_index(n: int = N_DAYS) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n, tz="UTC", name="date")


@pytest.fixture
def cointegrated_pair() -> pd.DataFrame:
    """Two series sharing a common trend: cointegrated by construction.

    ``b = 0.5 * a + stationary_noise``, so ``a - 2*b`` is stationary and the
    Engle-Granger test should reject the null of no cointegration.
    """
    rng = np.random.default_rng(RNG_SEED)
    common_trend = np.cumsum(rng.normal(0, 0.10, N_DAYS)) + 100.0
    # The noise has to be small relative to the trend. OLS of AA on BB puts the
    # noisy series on the right-hand side, so a large noise term induces
    # errors-in-variables attenuation and biases the estimated hedge ratio toward
    # zero -- with sd=0.25 the recovered beta is 1.73 rather than the true 2.0.
    # sd=0.05 keeps the attenuation under a percent while leaving the spread
    # comfortably stationary.
    noise = rng.normal(0, 0.05, N_DAYS)  # stationary, so the spread reverts
    return pd.DataFrame(
        {"AA": common_trend, "BB": 0.5 * common_trend + noise},
        index=_business_index(),
    )


@pytest.fixture
def independent_pair() -> pd.DataFrame:
    """Two unrelated random walks: not cointegrated, whatever the levels look like."""
    rng = np.random.default_rng(RNG_SEED + 1)
    return pd.DataFrame(
        {
            "AA": np.cumsum(rng.normal(0, 0.10, N_DAYS)) + 100.0,
            "BB": np.cumsum(rng.normal(0, 0.10, N_DAYS)) + 100.0,
        },
        index=_business_index(),
    )


@pytest.fixture
def stale_pair() -> pd.DataFrame:
    """A cointegrated pair where one leg only updates every tenth day.

    Mimics a policy-linked contract that sits unchanged between FOMC meetings.
    Used to check that ``coverage_report`` and ``analyse_pair`` actually surface
    the staleness rather than silently averaging over it.
    """
    rng = np.random.default_rng(RNG_SEED + 2)
    trend = np.cumsum(rng.normal(0, 0.10, N_DAYS)) + 100.0
    frame = pd.DataFrame(
        {"AA": trend, "BB": 0.5 * trend + rng.normal(0, 0.05, N_DAYS)},
        index=_business_index(),
    )
    # Hold BB flat except on every tenth day: ~90% of its daily changes are zero.
    frame["BB"] = frame["BB"].where(np.arange(N_DAYS) % 10 == 0).ffill().bfill()
    return frame


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


def test_drop_sunday_session_removes_only_sundays() -> None:
    """Sunday rows go; every other weekday survives untouched."""
    index = pd.date_range("2026-01-01", periods=28, freq="D", tz="UTC", name="date")
    panel = pd.DataFrame({"ZN": np.arange(28, dtype=float)}, index=index)

    cleaned = drop_sunday_session(panel)

    assert (cleaned.index.dayofweek != 6).all(), "a Sunday survived the filter"
    assert len(cleaned) == len(panel) - (panel.index.dayofweek == 6).sum()


def test_drop_sunday_session_is_idempotent(cointegrated_pair: pd.DataFrame) -> None:
    """Running it twice changes nothing the second time."""
    once = drop_sunday_session(cointegrated_pair)
    twice = drop_sunday_session(once)
    pd.testing.assert_frame_equal(once, twice)


def test_to_rate_space_inverts_rate_like_quotes() -> None:
    """ZQ/SR3 become rates; a Treasury price is passed through unchanged."""
    prices = pd.Series([99.50, 96.25, 95.00])

    rates = to_rate_space(prices, "ZQ")
    assert rates.tolist() == pytest.approx([0.50, 3.75, 5.00])

    # A root that is not rate-like must come back untouched, not 100 - price.
    assert "ZN" not in RATE_LIKE_ROOTS
    pd.testing.assert_series_equal(to_rate_space(prices, "ZN"), prices)


def test_to_rate_space_is_its_own_inverse() -> None:
    """Applying the transform twice returns the original quote."""
    prices = pd.Series([99.885, 96.375])
    pd.testing.assert_series_equal(
        to_rate_space(to_rate_space(prices, "ZQ"), "ZQ"), prices
    )


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_coverage_report_counts_missing_observations() -> None:
    """A NaN in the panel shows up as a missing observation, not a silent gap."""
    index = _business_index(10)
    panel = pd.DataFrame(
        {"AA": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=index
    )

    coverage = coverage_report(panel)

    assert coverage.loc["AA", "n_obs"] == 9
    assert coverage.loc["AA", "n_missing"] == 1
    assert coverage.loc["AA", "pct_missing"] == pytest.approx(0.1)


def test_coverage_report_detects_staleness(stale_pair: pd.DataFrame) -> None:
    """The stale leg is flagged; the active leg is not.

    This is the check that matters most: staleness is the property that
    disqualified ZQ/SR3 from being taken at face value, and nothing upstream
    reports it.
    """
    coverage = coverage_report(stale_pair)

    assert coverage.loc["BB", "stale_frac"] > 0.80, "failed to detect a mostly-flat series"
    assert coverage.loc["AA", "stale_frac"] < 0.05, "flagged an active series as stale"


# --------------------------------------------------------------------------
# Pair statistics
# --------------------------------------------------------------------------


def test_analyse_pair_detects_cointegration(cointegrated_pair: pd.DataFrame) -> None:
    """A pair that is cointegrated by construction is identified as such."""
    stats, spread = analyse_pair(cointegrated_pair, "AA", "BB")

    assert stats.eg_pvalue < 0.05, f"missed a cointegrated pair (EG p={stats.eg_pvalue:.4f})"
    assert stats.adf_pvalue < 0.05, f"spread was not stationary (ADF p={stats.adf_pvalue:.4f})"
    assert stats.beta == pytest.approx(2.0, rel=0.10), "hedge ratio is not the constructed 2.0"
    assert len(spread) == len(cointegrated_pair)


def test_analyse_pair_rejects_independent_series(independent_pair: pd.DataFrame) -> None:
    """Two unrelated random walks are not reported as cointegrated.

    The important half of the test. A screen that flags everything is worthless,
    so it has to be able to say no.
    """
    stats, _ = analyse_pair(independent_pair, "AA", "BB")
    assert stats.eg_pvalue > 0.05, f"false positive on independent walks (EG p={stats.eg_pvalue:.4f})"


def test_analyse_pair_drops_rows_missing_either_leg() -> None:
    """Only the overlapping window is used; a NaN in either leg drops the day.

    This is what makes ZQ/SR3's 1,645-day sample smaller than the panel's 2,986
    rows: SR3 simply does not exist before May 2018.
    """
    index = _business_index(200)
    rng = np.random.default_rng(RNG_SEED + 3)
    trend = np.cumsum(rng.normal(0, 0.1, 200)) + 100.0
    panel = pd.DataFrame(
        {"AA": trend, "BB": 0.5 * trend + rng.normal(0, 0.2, 200)}, index=index
    )
    panel.loc[panel.index[:50], "BB"] = np.nan  # BB "did not exist yet"

    stats, _ = analyse_pair(panel, "AA", "BB")

    assert stats.n_obs == 150
    assert stats.start == str(index[50].date())


def test_change_corr_differs_from_level_r_squared(stale_pair: pd.DataFrame) -> None:
    """The stale pair reproduces the ZQ/SR3 signature: high R^2, low change correlation.

    This is the diagnostic that exposed the problem in the real data, so it is
    worth pinning: a pair can fit almost perfectly on levels while its daily
    changes barely relate at all.
    """
    stats, _ = analyse_pair(stale_pair, "AA", "BB")

    assert stats.r_squared > 0.90, "expected the levels to track closely"
    assert stats.change_corr < 0.50, "expected the daily changes to decouple"


# --------------------------------------------------------------------------
# Half-life
# --------------------------------------------------------------------------


def test_half_life_of_mean_reverting_series() -> None:
    """A known AR(1) yields the half-life implied by its own coefficient.

    For ``s_t = phi * s_{t-1} + e_t``, the true half-life is ``-ln(2)/ln(phi)``.
    With phi = 0.90 that is about 6.6 days.
    """
    rng = np.random.default_rng(RNG_SEED + 4)
    phi = 0.90
    series = np.zeros(4000)
    for t in range(1, 4000):
        series[t] = phi * series[t - 1] + rng.normal(0, 1)

    expected = -np.log(2) / np.log(phi)
    assert half_life(pd.Series(series)) == pytest.approx(expected, rel=0.20)


def test_half_life_of_random_walk_is_effectively_infinite() -> None:
    """A random walk does not mean-revert, so its half-life must not look tradeable.

    Note what this does *not* assert. On a finite sample the OLS estimate of an
    AR(1) coefficient is downward-biased (the Dickey-Fuller result), so a random
    walk usually produces a slightly negative slope and therefore a finite
    half-life rather than NaN -- across 200 draws it comes back NaN only 8 times,
    with a median half-life of 438 days. Asserting NaN would be asserting a
    coin-flip.

    What matters for the screen is that the number is *large enough to be
    useless*: a spread with a 200+ day half-life is not something a trading
    strategy can hold for. So that is what we check.
    """
    rng = np.random.default_rng(RNG_SEED + 5)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 3000)))

    result = half_life(walk)
    assert np.isnan(result) or result > 200, (
        f"a random walk reported a tradeable half-life of {result:.1f} days"
    )


def test_half_life_handles_degenerate_input() -> None:
    """Too few points to fit: NaN rather than an exception."""
    assert np.isnan(half_life(pd.Series([1.0])))
    assert np.isnan(half_life(pd.Series([], dtype=float)))


# --------------------------------------------------------------------------
# Rolling beta
# --------------------------------------------------------------------------


def test_rolling_beta_recovers_a_constant_relationship() -> None:
    """When the true hedge ratio is fixed, the rolling estimate stays near it."""
    rng = np.random.default_rng(RNG_SEED + 6)
    index = _business_index(800)
    x = pd.Series(np.cumsum(rng.normal(0, 0.1, 800)) + 100.0, index=index)
    y = 1.5 * x + pd.Series(rng.normal(0, 0.05, 800), index=index)

    betas = rolling_beta(y, x, window=252)

    assert not betas.empty
    assert betas.mean() == pytest.approx(1.5, rel=0.05)
    assert betas.std() < 0.10, "a constant relationship should give a near-flat beta"


def test_rolling_beta_skips_degenerate_windows() -> None:
    """A window with no variation in x is skipped, not divided through by zero.

    In the real panel this happens during 2021, when the policy rate was pinned
    and the front-month contracts stopped moving entirely.
    """
    index = _business_index(400)
    x = pd.Series(np.r_[np.full(300, 99.99), np.linspace(99.99, 99.0, 100)], index=index)
    y = pd.Series(np.linspace(100, 101, 400), index=index)

    betas = rolling_beta(y, x, window=252)

    assert betas.notna().all(), "a degenerate window produced a NaN instead of being skipped"
    assert len(betas) < len(y) - 252 + 1, "expected at least one window to be skipped"


# --------------------------------------------------------------------------
# Ranking and robustness
# --------------------------------------------------------------------------


def test_rank_pairs_returns_one_row_per_pair(cointegrated_pair: pd.DataFrame) -> None:
    ranking = rank_pairs(cointegrated_pair, pairs=[("AA", "BB")])

    assert list(ranking.index) == ["AA/BB"]
    for column in ("eg_pvalue", "adf_pvalue", "half_life_days", "change_corr", "beta"):
        assert column in ranking.columns


def test_cointegration_robustness_reports_every_window(cointegrated_pair: pd.DataFrame) -> None:
    """A genuinely cointegrated pair stays cointegrated on later subsamples.

    This is the test that disqualified ZN/ZB in the real data: it passed on the
    full sample and failed on every window after 2018.
    """
    starts = ["2018-01-01", "2020-01-01"]
    robustness = cointegration_robustness(cointegrated_pair, pairs=[("AA", "BB")], starts=starts)

    assert list(robustness.columns) == ["eg_p_from_2018", "eg_p_from_2020"]

    # Both windows should still reject the null. The later window is shorter, so
    # the test has less power and the p-value rises -- which is the point of
    # running this at all. What disqualifies ZN/ZB in the real data is not a
    # p-value that drifts up a little, but one that goes from 0.054 to 0.54: an
    # order of magnitude, straight through every conventional threshold.
    assert (robustness.loc["AA/BB"] < 0.05).all(), "a real relationship should survive truncation"


def test_cointegration_robustness_handles_short_windows(cointegrated_pair: pd.DataFrame) -> None:
    """A subsample too short to test returns NaN rather than raising."""
    robustness = cointegration_robustness(
        cointegrated_pair, pairs=[("AA", "BB")], starts=["2099-01-01"]
    )
    assert np.isnan(robustness.loc["AA/BB", "eg_p_from_2099"])


# --------------------------------------------------------------------------
# Spread constructions -- the central finding
# --------------------------------------------------------------------------


def test_compare_spread_constructions_reports_all_three() -> None:
    """The three ZQ/SR3 weightings are each characterised, on synthetic input.

    Uses synthetic ZQ/SR3-shaped columns rather than the real panel, so this runs
    without Databento. The point of the check is that the function reports the
    DV01-neutral construction as a *distinct* object from the OLS one -- which is
    the whole finding: they are not the same spread and do not behave the same
    way.
    """
    rng = np.random.default_rng(RNG_SEED + 7)
    index = _business_index(1200)
    short_rate = np.cumsum(rng.normal(0, 0.01, 1200)) + 2.0  # a plausible policy path
    panel = pd.DataFrame(
        {
            "ZQ": 100.0 - short_rate,
            "SR3": 100.0 - (short_rate + rng.normal(0, 0.02, 1200)),
        },
        index=index,
    )

    constructions = compare_spread_constructions(panel)

    assert set(constructions.index) == {"raw_1to1", "ols_beta", "dv01_neutral"}
    assert constructions.loc["raw_1to1", "beta"] == pytest.approx(1.0)
    # 25.00 / 41.67
    assert constructions.loc["dv01_neutral", "beta"] == pytest.approx(0.60, abs=0.01)
    assert constructions.loc["dv01_neutral", "is_dv01_neutral"]
    assert not constructions.loc["raw_1to1", "is_dv01_neutral"]


def test_spread_volatility_by_year_respects_pair_and_scale(
    cointegrated_pair: pd.DataFrame,
) -> None:
    """The yearly-dispersion table works for an arbitrary pair, not just ZQ/SR3.

    With ``scale=1.0`` the values are the spread's own units (as for CL/BZ in
    $/bbl); the default basis-point scaling should be exactly 100x that.
    """
    yearly = spread_volatility_by_year(cointegrated_pair, "AA", "BB", scale=1.0)
    yearly_bp = spread_volatility_by_year(cointegrated_pair, "AA", "BB")

    assert (yearly.index == sorted(set(cointegrated_pair.index.year))).all()
    assert (yearly > 0).all()
    pd.testing.assert_series_equal(yearly_bp, yearly * 100.0)
