"""Tests for the spread model (issue #4).

Same pattern as ``test_exploratory_analysis``: everything runs on **synthetic**
data with known properties, so ``uv run pytest`` works on a fresh clone with no
Databento key. The fixtures construct a pair with a *known* alpha and beta, so
the tests check that the fit recovers the truth and that the saved artifacts
reproduce it exactly -- not merely that functions return without raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.spread import (
    SpreadModel,
    fit_spread_model,
    load_model,
    load_spread,
    save_model,
    save_spread,
    subsample_stability,
)

RNG_SEED = 20260721
N_DAYS = 1500

#: The relationship the cointegrated fixture is built around.
TRUE_ALPHA = 5.0
TRUE_BETA = 2.0


def _business_index(n: int = N_DAYS) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n, tz="UTC", name="date")


@pytest.fixture
def cointegrated_pair() -> pd.DataFrame:
    """``YY = 5 + 2 * XX + stationary_noise`` -- cointegrated by construction.

    The noise sits on the left-hand-side variable, where OLS assumes it lives,
    so the estimated alpha/beta are unbiased for the constructed values (unlike
    noise on the regressor, which attenuates beta -- see the EDA tests).
    """
    rng = np.random.default_rng(RNG_SEED)
    x = np.cumsum(rng.normal(0, 0.10, N_DAYS)) + 100.0
    noise = rng.normal(0, 0.05, N_DAYS)  # stationary, so the spread reverts
    return pd.DataFrame(
        {"YY": TRUE_ALPHA + TRUE_BETA * x + noise, "XX": x},
        index=_business_index(),
    )


@pytest.fixture
def independent_pair() -> pd.DataFrame:
    """Two unrelated random walks: not cointegrated, whatever the levels show."""
    rng = np.random.default_rng(RNG_SEED + 1)
    return pd.DataFrame(
        {
            "YY": np.cumsum(rng.normal(0, 0.10, N_DAYS)) + 100.0,
            "XX": np.cumsum(rng.normal(0, 0.10, N_DAYS)) + 100.0,
        },
        index=_business_index(),
    )


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def test_fit_recovers_known_parameters(cointegrated_pair: pd.DataFrame) -> None:
    """The fit finds the alpha and beta the fixture was built with."""
    model, _ = fit_spread_model(cointegrated_pair, "YY", "XX")

    assert model.beta == pytest.approx(TRUE_BETA, rel=0.02)
    assert model.alpha == pytest.approx(TRUE_ALPHA, abs=0.5)
    assert model.r_squared > 0.99
    assert model.n_obs == N_DAYS
    assert model.sample_start == str(cointegrated_pair.index.min().date())
    assert model.sample_end == str(cointegrated_pair.index.max().date())


def test_fit_detects_cointegration(cointegrated_pair: pd.DataFrame) -> None:
    """A pair cointegrated by construction is reported as such."""
    model, _ = fit_spread_model(cointegrated_pair, "YY", "XX")

    assert model.eg_pvalue < 0.05, f"missed cointegration (EG p={model.eg_pvalue:.4f})"
    assert model.adf_pvalue < 0.05, f"spread not stationary (ADF p={model.adf_pvalue:.4f})"
    assert model.half_life_days < 30, "a tight synthetic spread should revert quickly"


def test_fit_rejects_independent_series(independent_pair: pd.DataFrame) -> None:
    """Two unrelated random walks are not reported as cointegrated."""
    model, _ = fit_spread_model(independent_pair, "YY", "XX")
    assert model.eg_pvalue > 0.05, f"false positive (EG p={model.eg_pvalue:.4f})"


def test_spread_method_matches_frame_and_is_mean_zero(
    cointegrated_pair: pd.DataFrame,
) -> None:
    """``model.spread`` reproduces the frame's spread column exactly.

    This is the contract issue #5 relies on: applying the saved model to fresh
    prices must give the same series this stage computed. In-sample it is the
    OLS residual, so it is also mean-zero.
    """
    model, frame = fit_spread_model(cointegrated_pair, "YY", "XX")

    recomputed = model.spread(frame["yy"], frame["xx"])
    pd.testing.assert_series_equal(recomputed, frame["spread"], check_names=False)
    assert frame["spread"].mean() == pytest.approx(0.0, abs=1e-8)
    assert frame["spread"].std() == pytest.approx(model.spread_std)
    # The frame carries everything downstream stages need, keyed by date.
    assert list(frame.columns) == ["yy", "xx", "fitted", "spread"]


def test_fit_drops_rows_missing_either_leg(cointegrated_pair: pd.DataFrame) -> None:
    """A NaN in either leg drops the day; the sample metadata reflects it."""
    panel = cointegrated_pair.copy()
    panel.loc[panel.index[:50], "XX"] = np.nan  # XX "did not exist yet"
    panel.loc[panel.index[100], "YY"] = np.nan  # one missing settlement

    model, frame = fit_spread_model(panel, "YY", "XX")

    assert model.n_obs == N_DAYS - 51
    assert len(frame) == N_DAYS - 51
    assert model.sample_start == str(panel.index[50].date())


# --------------------------------------------------------------------------
# Subsample stability
# --------------------------------------------------------------------------


def test_subsample_stability_one_row_per_start(cointegrated_pair: pd.DataFrame) -> None:
    """Each start date gets a refit, and each refit finds the same relationship."""
    starts = ["2018-01-01", "2020-01-01", "2022-01-01"]
    stability = subsample_stability(cointegrated_pair, "YY", "XX", starts=starts)

    assert list(stability.index) == starts
    assert (stability["beta"] - TRUE_BETA).abs().max() < 0.05, (
        "a stable synthetic relationship should refit to the same beta everywhere"
    )
    assert (stability["eg_pvalue"] < 0.05).all()


def test_subsample_stability_handles_short_windows(
    cointegrated_pair: pd.DataFrame,
) -> None:
    """A start beyond (or too near) the end of the sample yields NaN, not a raise."""
    stability = subsample_stability(
        cointegrated_pair, "YY", "XX", starts=["2018-01-01", "2099-01-01"]
    )

    assert np.isnan(stability.loc["2099-01-01", "beta"])
    assert stability.loc["2099-01-01", "n_obs"] == 0
    assert not np.isnan(stability.loc["2018-01-01", "beta"])


# --------------------------------------------------------------------------
# Artifact round-trips
# --------------------------------------------------------------------------


def test_model_json_round_trip(cointegrated_pair: pd.DataFrame, tmp_path) -> None:
    """Save and reload reproduces the model field-for-field.

    Bit-exactness matters: issue #5's spread on live prices must equal this
    stage's spread on the same prices, which requires alpha/beta to survive
    serialisation without rounding.
    """
    model, _ = fit_spread_model(cointegrated_pair, "YY", "XX")
    path = tmp_path / "model.json"

    save_model(model, path)
    reloaded = load_model(path)

    assert reloaded == model  # dataclass equality: every field, exactly


def test_spread_parquet_round_trip(cointegrated_pair: pd.DataFrame, tmp_path) -> None:
    """The per-date artifact survives a parquet round-trip unchanged."""
    _, frame = fit_spread_model(cointegrated_pair, "YY", "XX")
    path = tmp_path / "spread.parquet"

    save_spread(frame, path)
    reloaded = load_spread(path)

    # check_freq=False: parquet stores dates, not the pandas-side freq attribute
    # the bdate_range fixture happens to carry; the real panel index has none.
    pd.testing.assert_frame_equal(reloaded, frame, check_freq=False)


def test_loaders_point_at_the_producing_command(tmp_path) -> None:
    """A missing artifact tells the caller which stage to run, like load_panel."""
    with pytest.raises(FileNotFoundError, match="src.models.spread"):
        load_model(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="src.models.spread"):
        load_spread(tmp_path / "missing.parquet")


def test_spread_model_dict_round_trip() -> None:
    """``to_dict``/``from_dict`` are exact inverses (what the JSON I/O rests on)."""
    model = SpreadModel(
        leg_y="CL",
        leg_x="BZ",
        alpha=1.2345678901234,
        beta=0.9753186420987,
        alpha_se=0.1,
        beta_se=0.002,
        r_squared=0.977,
        n_obs=2894,
        sample_start="2014-12-31",
        sample_end="2026-07-18",
        eg_pvalue=1e-05,
        adf_pvalue=2e-07,
        half_life_days=4.1,
        spread_std=2.5,
    )
    assert SpreadModel.from_dict(model.to_dict()) == model
