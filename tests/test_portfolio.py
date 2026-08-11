from __future__ import annotations

import math
import statistics

import pandas as pd
import pytest

from src.strategy.portfolio import combine_strategies, portfolio_diagnostics, summarize_pnl_series


def test_combine_strategies_weights_inversely_proportional_to_std() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    a = pd.Series([10.0, -10.0, 10.0, -10.0, 10.0, -10.0], index=idx)
    b = pd.Series([20.0, -20.0, 20.0, -20.0, 20.0, -20.0], index=idx)  # std_b == 2 * std_a exactly

    combined, weights = combine_strategies({"a": a, "b": b})

    assert weights["a"] == pytest.approx(1.0)
    assert weights["b"] == pytest.approx(0.5, rel=1e-9)
    assert list(combined.index) == list(idx)


def test_combine_strategies_restricts_to_intersection_not_zero_fill() -> None:
    idx_a = pd.date_range("2015-01-01", periods=10, freq="D", tz="UTC")
    idx_b = pd.date_range("2015-01-05", periods=10, freq="D", tz="UTC")
    a = pd.Series(1.0, index=idx_a)
    b = pd.Series(1.0, index=idx_b)

    combined, _ = combine_strategies({"a": a, "b": b})

    expected_common = idx_a.intersection(idx_b)
    assert list(combined.index) == list(expected_common)
    assert len(combined) < len(idx_a)
    assert len(combined) < len(idx_b)


def test_combine_strategies_requires_at_least_two_series() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    with pytest.raises(ValueError, match="at least two"):
        combine_strategies({"a": pd.Series([1.0, 2.0, 3.0], index=idx)})


def test_portfolio_diagnostics_correlation_and_sharpe() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    a = pd.Series([1.0, -1.0, 2.0, -2.0, 3.0, -3.0], index=idx)
    b = a.copy()  # identical series -> correlation exactly 1.0

    combined, _ = combine_strategies({"a": a, "b": b})
    diag = portfolio_diagnostics(a, b, combined)

    assert diag["n_common_sessions"] == 6
    assert diag["pnl_correlation"] == pytest.approx(1.0)

    negatively = portfolio_diagnostics(a, -a, combine_strategies({"a": a, "neg_a": -a})[0])
    assert negatively["pnl_correlation"] == pytest.approx(-1.0)


def test_summarize_pnl_series_matches_hand_computed_values() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    pnl = pd.Series([100.0, -50.0, 200.0, -300.0, 50.0], index=idx)

    summary = summarize_pnl_series(pnl)

    assert summary["n_sessions"] == 5
    assert summary["total_pnl_usd"] == pytest.approx(0.0)
    # cumsum = [100, 50, 250, -50, 0]; running max = [100,100,250,250,250]
    # drawdown = cumsum - running_max = [0,-50,0,-300,-250] -> min -300
    assert summary["max_drawdown_usd"] == pytest.approx(-300.0)
    expected_sharpe = statistics.mean(pnl) / statistics.stdev(pnl) * math.sqrt(252)
    assert summary["sharpe"] == pytest.approx(expected_sharpe)


def test_summarize_pnl_series_empty_series_returns_nan_not_error() -> None:
    empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    summary = summarize_pnl_series(empty)
    assert summary["n_sessions"] == 0
    assert math.isnan(summary["max_drawdown_usd"])
