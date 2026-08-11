from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.seasonality import (
    MonthlySeasonalityModel,
    fit_monthly_seasonality,
    run_seasonal_backtest,
    seasonal_position_multiplier,
    walkforward_seasonality,
)
from src.models.spread import SpreadModel
from src.strategy.backtest import Fold, build_walkforward_folds, run_backtest


def dummy_model(beta: float = 0.975) -> SpreadModel:
    return SpreadModel(
        leg_y="CL",
        leg_x="BZ",
        alpha=-2.65,
        beta=beta,
        alpha_se=0.1,
        beta_se=0.01,
        r_squared=0.97,
        n_obs=10,
        sample_start="2020-01-01",
        sample_end="2020-01-10",
        eg_pvalue=0.001,
        adf_pvalue=0.001,
        half_life_days=4.0,
        spread_std=2.0,
    )


def test_fit_monthly_seasonality_recovers_known_injected_month_effect() -> None:
    idx = pd.bdate_range("2015-01-01", "2020-12-31", tz="UTC")
    rng = np.random.default_rng(11)
    changes = rng.normal(0, 0.1, len(idx))
    march_mask = idx.month == 3
    changes[march_mask] += 2.0
    spread = pd.Series(np.cumsum(changes), index=idx)

    model = fit_monthly_seasonality(spread)

    assert model.month_effect[3] == pytest.approx(2.0, abs=0.1)
    assert 3 in model.significant_months
    assert 6 not in model.significant_months


def test_fit_monthly_seasonality_respects_as_of_cutoff() -> None:
    idx = pd.bdate_range("2015-01-01", "2020-12-31", tz="UTC")
    rng = np.random.default_rng(2)
    changes = rng.normal(0, 0.1, len(idx))
    spread = pd.Series(np.cumsum(changes), index=idx)
    as_of = pd.Timestamp("2018-01-01", tz="UTC")

    model = fit_monthly_seasonality(spread, as_of=as_of)

    assert model.as_of == "2018-01-01"
    # every date used must be strictly before the cutoff
    changes = spread.diff().dropna()
    expected_n_obs = int((changes.index < as_of).sum())
    assert model.n_obs == expected_n_obs


def test_walkforward_seasonality_uses_only_pretest_years() -> None:
    idx = pd.bdate_range("2015-01-01", "2019-12-31", tz="UTC")
    rng = np.random.default_rng(5)
    changes = rng.normal(0, 0.1, len(idx))
    boundary = pd.Timestamp("2018-01-01", tz="UTC")
    march_mask = idx.month == 3
    before = march_mask & (idx < boundary)
    after = march_mask & (idx >= boundary)
    changes[before] += 2.0
    changes[after] -= 2.0
    spread = pd.Series(np.cumsum(changes), index=idx)

    folds = build_walkforward_folds(idx, min_train_sessions=504, min_test_sessions=60)
    fold0 = folds[0]
    assert fold0.test_start < boundary  # sanity: fold0 trains on pre-flip data only

    model = fit_monthly_seasonality(spread, as_of=fold0.test_start)
    assert model.month_effect[3] == pytest.approx(2.0, abs=0.3)

    table = walkforward_seasonality(spread, folds)
    assert table.loc[table["fold_id"] == fold0.fold_id, "month_3_effect"].iloc[0] == pytest.approx(2.0, abs=0.3)


def test_seasonal_position_multiplier_is_noop_when_effect_insignificant() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    position = pd.Series([1.0] * 10, index=idx)
    fold = Fold(
        fold_id=0,
        train_start=idx.min(),
        train_end=idx.min(),
        test_start=idx.min(),
        test_end=idx.max(),
        n_train_sessions=1,
        n_test_sessions=10,
    )
    model = MonthlySeasonalityModel(
        as_of=str(idx.min().date()),
        n_obs=100,
        month_effect={m: 5.0 for m in range(1, 13)},
        month_pvalue={m: 0.9 for m in range(1, 13)},
        overall_fpvalue=0.9,
        significant_months=[],
    )

    scale = seasonal_position_multiplier(idx, position, [(fold, model)])

    assert (scale == 1.0).all()


def test_seasonal_position_multiplier_tilts_when_significant() -> None:
    idx = pd.date_range("2024-01-05", periods=5, freq="D", tz="UTC")  # all in January
    position = pd.Series([1.0, -1.0, 1.0, -1.0, 0.0], index=idx)
    fold = Fold(
        fold_id=0,
        train_start=idx.min(),
        train_end=idx.min(),
        test_start=idx.min(),
        test_end=idx.max(),
        n_train_sessions=1,
        n_test_sessions=5,
    )
    model = MonthlySeasonalityModel(
        as_of=str(idx.min().date()),
        n_obs=100,
        month_effect={**{m: 0.0 for m in range(1, 13)}, 1: 2.0},  # positive January drift
        month_pvalue={**{m: 1.0 for m in range(1, 13)}, 1: 0.01},
        overall_fpvalue=0.01,
        significant_months=[1],
    )

    scale = seasonal_position_multiplier(idx, position, [(fold, model)])

    assert scale.iloc[0] > 1.0  # long position, positive drift -> tailwind
    assert scale.iloc[1] < 1.0  # short position, positive drift -> headwind
    assert scale.iloc[4] == pytest.approx(1.0)  # flat position -> no-op regardless


def test_run_seasonal_backtest_with_no_fold_models_matches_run_backtest() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    signals = pd.DataFrame(
        {
            "cl": [70.0] * 5,
            "bz": [70.0] * 5,
            "spread": [0.0, 1.0, 3.0, 2.0, 2.0],
            "next_session_position": [0, 1, 1, -1, 0],
        },
        index=idx,
    )
    model = dummy_model(beta=1.0)

    overlay = run_seasonal_backtest(signals, model, None)
    direct = run_backtest(signals, model)

    pd.testing.assert_frame_equal(overlay, direct)
