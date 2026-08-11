from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.spread import SpreadModel
from src.strategy.backtest import run_backtest
from src.strategy.regime import RegimeConfig, classify_vol_regime, regime_scale, run_regime_backtest


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


def test_classify_vol_regime_uses_expanding_quantile_not_full_sample() -> None:
    idx = pd.date_range("2015-01-01", periods=800, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    base = pd.Series(np.cumsum(rng.normal(0, 1, 800)), index=idx)
    perturbed = base.copy()
    perturbed.iloc[600:] = perturbed.iloc[600:] + rng.normal(0, 50, 200)

    config = RegimeConfig(vol_lookback=30)
    regime_base = classify_vol_regime(base, config)
    regime_pert = classify_vol_regime(perturbed, config)

    pd.testing.assert_series_equal(regime_base.iloc[:500], regime_pert.iloc[:500])


def test_regime_scale_reduces_exposure_in_high_vol_segment() -> None:
    idx = pd.date_range("2015-01-01", periods=100, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    calm = np.cumsum(rng.normal(0, 0.1, 60))
    stormy = calm[-1] + np.cumsum(rng.normal(0, 5.0, 40))
    spread = pd.Series(np.concatenate([calm, stormy]), index=idx)
    config = RegimeConfig(vol_lookback=20, vol_low_quantile=0.25, vol_high_quantile=0.75)

    scale = regime_scale(spread, config)

    stormy_scale_mean = scale.iloc[80:].mean()
    calm_scale_mean = scale.iloc[20:55].mean()
    assert stormy_scale_mean < calm_scale_mean


def test_trend_gate_disabled_by_default_leaves_scale_purely_vol_based() -> None:
    idx = pd.date_range("2015-01-01", periods=50, freq="D", tz="UTC")
    spread = pd.Series(np.linspace(0, 50, 50), index=idx)
    config_off = RegimeConfig(trend_gate_enabled=False)
    config_on = RegimeConfig(trend_gate_enabled=True, trend_lookback=10, trend_z_threshold=0.5, trend_scale=0.5)

    scale_off = regime_scale(spread, config_off)
    scale_on = regime_scale(spread, config_on)

    assert (scale_on <= scale_off + 1e-9).all()
    assert (scale_on < scale_off).any()


def test_run_regime_backtest_with_no_config_matches_run_backtest() -> None:
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

    overlay = run_regime_backtest(signals, model, None)
    direct = run_backtest(signals, model)

    pd.testing.assert_frame_equal(overlay, direct)
