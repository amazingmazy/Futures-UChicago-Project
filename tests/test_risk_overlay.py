from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.spread import SpreadModel
from src.strategy.backtest import run_backtest
from src.strategy.risk_overlay import (
    DrawdownGateConfig,
    VolTargetConfig,
    apply_drawdown_gate,
    run_overlay_backtest,
    spread_daily_dollar_vol,
    vol_target_scale,
)


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


# --------------------------------------------------------------------------
# Volatility targeting
# --------------------------------------------------------------------------


def test_spread_daily_dollar_vol_matches_hand_computed_std() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    spread = pd.Series([0.0, 1.0, 2.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0], index=idx)

    vol = spread_daily_dollar_vol(spread, lookback=3, contract_size=1.0)

    expected = spread.diff().shift(1).rolling(3).std()
    pd.testing.assert_series_equal(vol, expected, check_names=False)


def test_spread_daily_dollar_vol_is_causal() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    base = pd.Series(np.cumsum(rng.normal(0, 1, 20)), index=idx)
    perturbed = base.copy()
    perturbed.iloc[15:] = perturbed.iloc[15:] + 100.0

    vol_base = spread_daily_dollar_vol(base, lookback=5, contract_size=1.0)
    vol_pert = spread_daily_dollar_vol(perturbed, lookback=5, contract_size=1.0)

    pd.testing.assert_series_equal(vol_base.iloc[:15], vol_pert.iloc[:15])


def test_vol_target_scale_clips_to_bounds() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    small = np.tile([0.0, 0.001], 5)
    big = np.tile([0.0, 1000.0], 5)
    spread = pd.Series(np.concatenate([small, big]), index=idx)
    config = VolTargetConfig(target_daily_vol_usd=1.0, lookback=3, min_scale=0.25, max_scale=2.0)

    scale = vol_target_scale(spread, config, contract_size=1.0)

    assert scale.iloc[5] == pytest.approx(2.0)  # tiny trailing vol -> clipped to max_scale
    assert scale.iloc[-1] == pytest.approx(0.25)  # huge trailing vol -> clipped to min_scale


def test_vol_target_scale_neutral_during_warmup() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    spread = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], index=idx)
    config = VolTargetConfig(target_daily_vol_usd=1.0, lookback=10)

    scale = vol_target_scale(spread, config)

    assert (scale == 1.0).all()


# --------------------------------------------------------------------------
# Drawdown gate
# --------------------------------------------------------------------------


def test_apply_drawdown_gate_triggers_and_recovers_with_hysteresis() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    unit_pnl = pd.Series([0.0, -20.0, -20.0, -20.0, 0.0, 0.0, 30.0, 30.0, 10.0, 10.0], index=idx)
    position = pd.Series([1.0] * 10, index=idx)
    config = DrawdownGateConfig(trigger_pct=-0.15, recovery_pct=-0.05, de_risked_scale=0.5)

    gated = apply_drawdown_gate(unit_pnl, position, reference_notional=100.0, config=config)

    expected = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
    assert gated.tolist() == pytest.approx(expected)


def test_apply_drawdown_gate_is_causal_wrt_todays_pnl() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    # Day 1's own move alone breaches the -0.15 * 100 = -15 trigger.
    unit_pnl = pd.Series([0.0, -50.0, -50.0], index=idx)
    position = pd.Series([1.0, 1.0, 1.0], index=idx)
    config = DrawdownGateConfig(trigger_pct=-0.15, recovery_pct=-0.05, de_risked_scale=0.5)

    gated = apply_drawdown_gate(unit_pnl, position, reference_notional=100.0, config=config)

    assert gated.iloc[1] == pytest.approx(1.0)  # not reduced by its own move
    assert gated.iloc[2] == pytest.approx(0.5)  # reduced the session after


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def test_run_overlay_backtest_with_no_configs_matches_run_backtest() -> None:
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

    overlay = run_overlay_backtest(signals, model)
    direct = run_backtest(signals, model)

    pd.testing.assert_frame_equal(overlay, direct)
