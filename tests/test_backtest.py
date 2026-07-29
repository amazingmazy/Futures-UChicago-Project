from __future__ import annotations

import math
import statistics

import numpy as np
import pandas as pd
import pytest

from src.models.spread import SpreadModel
from src.strategy.backtest import (
    CONTRACT_SIZE_BBL,
    CostConfig,
    apply_daily_loss_cap,
    build_grid_frames,
    build_walkforward_folds,
    contract_notional_usd,
    identify_trades,
    load_backtest,
    max_drawdown,
    run_backtest,
    run_walkforward,
    save_backtest,
    sharpe_ratio,
    summarize_backtest,
    summarize_grid,
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


def make_signals_frame(
    cl: list[float], bz: list[float], spread: list[float], next_session_position: list[int]
) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(cl), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "cl": cl,
            "bz": bz,
            "spread": spread,
            "next_session_position": next_session_position,
        },
        index=idx,
    )


# --------------------------------------------------------------------------
# PnL and cost engine
# --------------------------------------------------------------------------


def test_contract_notional_uses_absolute_price() -> None:
    cl = pd.Series([-37.63, 40.0])
    bz = pd.Series([25.0, 41.0])
    notional = contract_notional_usd(cl, bz, beta=0.5)
    expected = [(37.63 + 0.5 * 25.0) * 1000.0, (40.0 + 0.5 * 41.0) * 1000.0]
    assert notional.tolist() == pytest.approx(expected)


def test_gross_pnl_matches_hand_computation() -> None:
    signals = make_signals_frame(
        cl=[70.0] * 5,
        bz=[70.0] * 5,
        spread=[0.0, 1.0, 3.0, 2.0, 2.0],
        next_session_position=[0, 1, 1, -1, 0],
    )
    result = run_backtest(signals, dummy_model(beta=1.0), CostConfig(apply_costs=False))
    expected = [0.0, 1000.0, 2000.0, 1000.0, 0.0]
    assert result["gross_pnl_usd"].tolist() == pytest.approx(expected)


def test_transaction_cost_zero_on_hold_days_and_matches_formula_on_turnover() -> None:
    signals = make_signals_frame(
        cl=[70.0] * 5,
        bz=[70.0] * 5,
        spread=[0.0, 1.0, 3.0, 2.0, 2.0],
        next_session_position=[0, 1, 1, -1, 0],
    )
    cost_config = CostConfig(transaction_cost_bps=10.0, financing_bps_per_day=0.0)
    result = run_backtest(signals, dummy_model(beta=1.0), cost_config)

    notional = (70.0 + 1.0 * 70.0) * 1000.0
    turnover = [0.0, 1.0, 0.0, 2.0, 1.0]
    expected_cost = [t * 10.0 / 1e4 * notional for t in turnover]
    assert result["turnover"].tolist() == pytest.approx(turnover)
    assert result["transaction_cost_usd"].tolist() == pytest.approx(expected_cost)
    # hold days (no turnover) cost exactly zero
    assert result["transaction_cost_usd"].iloc[2] == pytest.approx(0.0)
    assert result["net_pnl_usd"].tolist() == pytest.approx(
        (result["gross_pnl_usd"] - result["transaction_cost_usd"]).tolist()
    )


def test_financing_cost_accrues_while_held_not_when_flat() -> None:
    signals = make_signals_frame(
        cl=[70.0] * 5,
        bz=[70.0] * 5,
        spread=[0.0, 0.0, 0.0, 0.0, 0.0],
        next_session_position=[0, 1, 1, 1, 0],
    )
    cost_config = CostConfig(transaction_cost_bps=0.0, financing_bps_per_day=5.0)
    result = run_backtest(signals, dummy_model(beta=1.0), cost_config)

    notional = (70.0 + 1.0 * 70.0) * 1000.0
    expected = [0.0, 5.0 / 1e4 * notional, 5.0 / 1e4 * notional, 5.0 / 1e4 * notional, 0.0]
    assert result["financing_cost_usd"].tolist() == pytest.approx(expected)
    assert result["financing_cost_usd"].iloc[0] == pytest.approx(0.0)
    assert result["financing_cost_usd"].iloc[4] == pytest.approx(0.0)


def test_run_backtest_zero_cost_config_equals_raw_pnl() -> None:
    signals = make_signals_frame(
        cl=[70.0, 71.0, 69.0, 72.0],
        bz=[70.0, 70.5, 70.0, 71.0],
        spread=[0.0, 1.0, -1.0, 2.0],
        next_session_position=[0, 1, -1, 1],
    )
    result = run_backtest(signals, dummy_model(beta=0.9), CostConfig(apply_costs=False))
    assert result["net_pnl_usd"].tolist() == pytest.approx(result["gross_pnl_usd"].tolist())
    assert (result["transaction_cost_usd"] == 0.0).all()
    assert (result["financing_cost_usd"] == 0.0).all()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_sharpe_ratio_matches_hand_computed_value() -> None:
    pnl = pd.Series([2.0, -1.0, 3.0, 0.0])
    expected = statistics.mean(pnl) / statistics.stdev(pnl) * math.sqrt(252)
    assert sharpe_ratio(pnl) == pytest.approx(expected)


def test_sharpe_ratio_is_nan_when_flat() -> None:
    assert math.isnan(sharpe_ratio(pd.Series([0.0, 0.0, 0.0])))


def test_max_drawdown_matches_hand_computed_value() -> None:
    equity = pd.Series([100.0, 110.0, 90.0, 95.0, 80.0, 120.0])
    expected = 80.0 / 110.0 - 1.0
    assert max_drawdown(equity) == pytest.approx(expected)


def test_identify_trades_labels_contiguous_runs() -> None:
    position = pd.Series([0, 1, 1, 0, -1, -1, 0, 1])
    trade_id = identify_trades(position)
    expected = [np.nan, 1, 1, np.nan, 2, 2, np.nan, 3]
    for got, want in zip(trade_id.tolist(), expected):
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == want


def test_hit_rate_and_trade_count_on_synthetic_round_trips() -> None:
    signals = make_signals_frame(
        cl=[70.0] * 7,
        bz=[70.0] * 7,
        spread=[0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0],
        next_session_position=[0, 1, 1, 0, -1, -1, 0],
    )
    result = run_backtest(signals, dummy_model(beta=0.0), CostConfig(apply_costs=False))
    summary = summarize_backtest(result, reference_notional=70_000.0)
    assert summary["n_round_trip_trades"] == 2
    # trade 1 (long, spread flat->1->1): gains; trade 2 (short, spread 0->-1->-1): gains too
    assert summary["hit_rate"] == pytest.approx(1.0)
    assert summary["avg_holding_sessions"] == pytest.approx(2.0)


def test_apply_daily_loss_cap_only_clips_losses_beyond_threshold() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    frame = pd.DataFrame({"net_pnl_usd": [100.0, -50.0, -500.0, 200.0, -10.0]}, index=idx)
    capped = apply_daily_loss_cap(frame, cap_usd=100.0)
    expected = [100.0, -50.0, -100.0, 200.0, -10.0]
    assert capped["net_pnl_usd"].tolist() == pytest.approx(expected)
    assert capped["net_cum_pnl_usd"].tolist() == pytest.approx(list(pd.Series(expected).cumsum()))


def test_summarize_backtest_matches_hand_computed_fields() -> None:
    signals = make_signals_frame(
        cl=[70.0] * 5,
        bz=[70.0] * 5,
        spread=[0.0, 1.0, 3.0, 2.0, 2.0],
        next_session_position=[0, 1, 1, -1, 0],
    )
    result = run_backtest(signals, dummy_model(beta=1.0), CostConfig(apply_costs=False))
    summary = summarize_backtest(result, reference_notional=140_000.0)
    assert summary["n_sessions"] == 5
    assert summary["total_net_pnl_usd"] == pytest.approx(4000.0)
    assert summary["total_gross_pnl_usd"] == pytest.approx(summary["total_net_pnl_usd"])
    assert not math.isnan(summary["sharpe_net"])


# --------------------------------------------------------------------------
# Threshold grid
# --------------------------------------------------------------------------


def test_build_grid_frames_skips_invalid_exit_greater_than_entry_combos() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    cl = 70.0 + np.cumsum(rng.normal(0, 0.3, size=20))
    bz = 70.0 + np.cumsum(rng.normal(0, 0.3, size=20))
    spread_frame = pd.DataFrame({"cl": cl, "bz": bz, "spread": cl - bz}, index=idx)

    grid_frames = build_grid_frames(
        spread_frame,
        dummy_model(beta=1.0),
        entry_grid=(1.0, 2.0),
        exit_grid=(0.5, 1.0, 3.0),
        window=3,
    )

    assert set(grid_frames.keys()) == {(1.0, 0.5), (2.0, 0.5), (2.0, 1.0)}
    for frame in grid_frames.values():
        assert len(frame) == len(spread_frame)


def test_summarize_grid_selects_highest_net_sharpe_config() -> None:
    def frame_with_pnl(pnl_values: list[float]) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
        frame = pd.DataFrame(
            {
                "next_session_position": [0, 1, 1, 0],
                "gross_pnl_usd": pnl_values,
                "net_pnl_usd": pnl_values,
                "turnover": [0.0, 1.0, 0.0, 1.0],
            },
            index=idx,
        )
        frame["gross_cum_pnl_usd"] = frame["gross_pnl_usd"].cumsum()
        frame["net_cum_pnl_usd"] = frame["net_pnl_usd"].cumsum()
        return frame

    grid_frames = {
        (1.0, 0.5): frame_with_pnl([0.0, 100.0, 100.0, -20.0]),  # clear positive -> best
        (3.0, 0.5): frame_with_pnl([0.0, 0.0, 0.0, 0.0]),  # never trades -> NaN sharpe
        (1.5, 0.5): frame_with_pnl([0.0, -50.0, 30.0, -80.0]),  # negative/choppy -> worse
    }
    grid_summary = summarize_grid(grid_frames, reference_notional=140_000.0)
    best = grid_summary.sort_values("sharpe_net", ascending=False).iloc[0]
    assert (float(best["entry_z"]), float(best["exit_z"])) == (1.0, 0.5)


# --------------------------------------------------------------------------
# Walk-forward folds and leakage
# --------------------------------------------------------------------------


def test_build_walkforward_folds_are_contiguous_and_cover_full_range() -> None:
    index = pd.bdate_range("2015-01-01", "2020-12-31", tz="UTC")
    folds = build_walkforward_folds(index, min_train_sessions=504, min_test_sessions=60)

    assert len(folds) > 1
    assert folds[0].n_train_sessions >= 504
    assert folds[-1].test_end == index.max()
    for i in range(len(folds) - 1):
        assert folds[i].test_end < folds[i + 1].test_start
        assert folds[i].fold_id == i

    covered = index[index >= folds[0].test_start]
    assert sum(f.n_test_sessions for f in folds) == len(covered)


def test_build_walkforward_folds_raises_when_insufficient_history() -> None:
    index = pd.bdate_range("2024-01-01", "2024-06-30", tz="UTC")
    with pytest.raises(ValueError, match="Not enough history"):
        build_walkforward_folds(index, min_train_sessions=504, min_test_sessions=60)


def _synthetic_spread_frame(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spread = np.cumsum(rng.normal(0, 1.0, size=len(index)))
    # mean-revert it slightly so the rolling z-score actually crosses thresholds
    spread = spread - pd.Series(spread).rolling(60, min_periods=1).mean().to_numpy() * 0.5
    return pd.DataFrame({"cl": 70.0, "bz": 70.0, "spread": spread}, index=index)


def test_walkforward_fold_selection_and_oos_pnl_are_invariant_to_future_data() -> None:
    index = pd.bdate_range("2015-01-01", "2019-12-31", tz="UTC")
    base = _synthetic_spread_frame(index, seed=11)

    folds = build_walkforward_folds(index, min_train_sessions=504, min_test_sessions=60)
    fold0_test_end = folds[0].test_end

    perturbed = base.copy()
    future_mask = perturbed.index > fold0_test_end
    rng = np.random.default_rng(999)
    perturbed.loc[future_mask, "spread"] = rng.normal(0, 5.0, size=int(future_mask.sum()))

    model = dummy_model(beta=0.9)
    small_grid = dict(entry_grid=(1.0, 1.5), exit_grid=(0.25, 0.5), window=20)

    table_a, oos_a = run_walkforward(base, model, **small_grid)
    table_b, oos_b = run_walkforward(perturbed, model, **small_grid)

    assert table_a.iloc[0]["selected_entry_z"] == table_b.iloc[0]["selected_entry_z"]
    assert table_a.iloc[0]["selected_exit_z"] == table_b.iloc[0]["selected_exit_z"]

    fold0_mask_a = oos_a.index <= fold0_test_end
    fold0_mask_b = oos_b.index <= fold0_test_end
    pd.testing.assert_series_equal(
        oos_a.loc[fold0_mask_a, "net_pnl_usd"],
        oos_b.loc[fold0_mask_b, "net_pnl_usd"],
        check_names=False,
    )


def test_concatenated_oos_pnl_has_no_duplicate_or_missing_dates() -> None:
    index = pd.bdate_range("2015-01-01", "2019-12-31", tz="UTC")
    spread_frame = _synthetic_spread_frame(index, seed=5)
    model = dummy_model(beta=0.9)

    walkforward_table, oos_frame = run_walkforward(
        spread_frame, model, entry_grid=(1.0, 1.5), exit_grid=(0.25, 0.5), window=20
    )

    assert oos_frame.index.is_monotonic_increasing
    assert not oos_frame.index.has_duplicates

    folds = build_walkforward_folds(index, min_train_sessions=504, min_test_sessions=60)
    expected_index = index[index >= folds[0].test_start]
    assert list(oos_frame.index) == list(expected_index)
    assert len(walkforward_table) == len(folds)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def test_save_load_backtest_round_trip(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame({"net_pnl_usd": [1.0, 2.0, 3.0]}, index=idx)
    path = tmp_path / "backtest_cl_bz.parquet"

    save_backtest(frame, path)
    loaded = load_backtest(path)

    pd.testing.assert_frame_equal(loaded, frame, check_freq=False)


def test_load_backtest_points_at_producing_command(tmp_path) -> None:
    missing_path = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError, match="src.strategy.backtest"):
        load_backtest(missing_path)
