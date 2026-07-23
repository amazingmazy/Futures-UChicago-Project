from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.spread import SpreadModel
from src.strategy.signals import (
    SignalConfig,
    generate_positions,
    generate_signals,
    rolling_zscore,
    summarize_signals,
    validate_config,
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


def test_validate_config_rejects_bad_threshold_order() -> None:
    with pytest.raises(ValueError, match="exit_z must be smaller"):
        validate_config(SignalConfig(entry_z=1.0, exit_z=1.0))


def test_validate_config_rejects_too_little_history() -> None:
    with pytest.raises(ValueError, match="min_periods"):
        validate_config(SignalConfig(window=5, min_periods=1))


def test_rolling_zscore_uses_prior_window_by_default() -> None:
    spread = pd.Series(np.arange(6, dtype=float))
    config = SignalConfig(window=3, min_periods=3, lag_rolling_stats=True)

    zscore = rolling_zscore(spread, config)

    assert zscore.iloc[:3].isna().all()
    expected_std = math.sqrt(2.0 / 3.0)  # population std of [0, 1, 2]
    assert zscore.iloc[3] == pytest.approx((3.0 - 1.0) / expected_std)


def test_generate_positions_enters_exits_and_does_not_same_day_reverse() -> None:
    zscore = pd.Series([np.nan, -1.6, -1.0, -0.4, 1.8, 0.4, -2.2])
    config = SignalConfig(entry_z=1.5, exit_z=0.5)

    positions = generate_positions(zscore, config)

    assert positions.tolist() == [0, 1, 1, 0, -1, 0, 1]


def test_generate_signals_builds_hedged_contract_targets() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    spread_frame = pd.DataFrame(
        {
            "cl": [70, 69, 68, 67, 75, 74, 73, 72],
            "bz": [72, 72, 72, 72, 72, 72, 72, 72],
            "spread": [0.0, 0.1, -0.1, -2.0, -1.0, -0.2, 2.0, 0.1],
        },
        index=idx,
    )
    model = dummy_model(beta=0.8)
    config = SignalConfig(window=3, min_periods=3, entry_z=1.0, exit_z=0.25)

    signals = generate_signals(spread_frame, model, config)

    assert {"zscore", "target_position", "next_session_position", "signal"}.issubset(signals.columns)
    assert signals.loc[idx[3], "signal"] == "enter_long_spread"
    assert signals.loc[idx[3], "cl_contracts"] == pytest.approx(1.0)
    assert signals.loc[idx[3], "bz_contracts"] == pytest.approx(-0.8)
    assert signals.loc[idx[4], "next_session_position"] == 1
    assert signals.loc[idx[6], "signal"] == "enter_short_spread"
    assert signals.loc[idx[6], "cl_contracts"] == pytest.approx(-1.0)
    assert signals.loc[idx[6], "bz_contracts"] == pytest.approx(0.8)


def test_generate_signals_requires_spread_column() -> None:
    with pytest.raises(KeyError, match="spread"):
        generate_signals(pd.DataFrame({"cl": [1.0]}), dummy_model())


def test_summarize_signals_counts_entries_and_exits() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    signals = pd.DataFrame(
        {
            "zscore": [np.nan, -2.0, -1.0, -0.1, 2.0, 0.1],
            "target_position": [0, 1, 1, 0, -1, 0],
            "entry": [False, True, False, False, True, False],
            "exit": [False, False, False, True, False, True],
        },
        index=idx,
    )

    summary = summarize_signals(signals, SignalConfig())

    assert summary["n_sessions"] == 6
    assert summary["n_entries"] == 2
    assert summary["n_long_entries"] == 1
    assert summary["n_short_entries"] == 1
    assert summary["n_exits"] == 2
    assert summary["final_target_position"] == 0
