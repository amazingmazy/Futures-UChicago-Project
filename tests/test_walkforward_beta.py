from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.backtest import Fold, build_walkforward_folds
from src.strategy.walkforward_beta import build_fold_spread_frame, run_walkforward_beta_refit


def _synthetic_panel(idx: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """A mean-reverting Y/X pair with a stable ~1:1 true relationship, used
    where the walk-forward machinery matters more than the economics."""
    rng = np.random.default_rng(seed)
    x = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, size=len(idx))), index=idx)
    noise = np.cumsum(rng.normal(0, 1.0, size=len(idx)))
    noise = noise - pd.Series(noise).rolling(60, min_periods=1).mean().to_numpy() * 0.5
    y = x + 5.0 + noise
    return pd.DataFrame({"Y": y, "X": x}, index=idx)


def test_build_fold_spread_frame_beta_matches_manual_ols_on_train_only() -> None:
    idx = pd.bdate_range("2015-01-01", "2016-12-31", tz="UTC")
    boundary = pd.Timestamp("2016-01-01", tz="UTC")
    rng = np.random.default_rng(42)
    x = pd.Series(rng.normal(100, 5, size=len(idx)), index=idx)
    before = idx < boundary
    true_beta = np.where(before, 2.0, -1.0)
    true_alpha = np.where(before, 5.0, 50.0)
    # Noise is required: with an exactly colinear y, the OLS residuals are
    # identically zero and fit_spread_model's ADF diagnostic raises
    # "Invalid input, x is constant".
    y = pd.Series(
        true_alpha + true_beta * x.to_numpy() + rng.normal(0, 0.05, size=len(idx)),
        index=idx,
    )
    panel = pd.DataFrame({"Y": y, "X": x})

    fold = Fold(
        fold_id=0,
        train_start=idx.min(),
        train_end=boundary,
        test_start=boundary,
        test_end=idx.max(),
        n_train_sessions=int(before.sum()),
        n_test_sessions=int((~before).sum()),
    )

    model, frame = build_fold_spread_frame(panel, "Y", "X", fold)

    expected_beta = np.polyfit(x[before], y[before], 1)[0]
    assert model.beta == pytest.approx(expected_beta, rel=1e-6)
    assert model.beta == pytest.approx(2.0, rel=1e-3)
    assert model.beta != pytest.approx(-1.0, abs=0.5)
    assert set(frame.columns) >= {"y", "x", "fitted", "spread"}
    assert len(frame) == len(idx)


def test_beta_refit_walkforward_fold0_invariant_to_future_panel_data() -> None:
    idx = pd.bdate_range("2015-01-01", "2019-12-31", tz="UTC")
    base = _synthetic_panel(idx, seed=11)

    folds = build_walkforward_folds(idx, min_train_sessions=504, min_test_sessions=60)
    fold0_test_end = folds[0].test_end

    perturbed = base.copy()
    future_mask = perturbed.index > fold0_test_end
    rng = np.random.default_rng(999)
    perturbed.loc[future_mask, "X"] = 100.0 + rng.normal(0, 20, size=int(future_mask.sum()))
    perturbed.loc[future_mask, "Y"] = (
        perturbed.loc[future_mask, "X"] * 3.0 + rng.normal(0, 5, size=int(future_mask.sum()))
    )

    small_grid = dict(entry_grid=(1.0, 1.5), exit_grid=(0.25, 0.5), window=20, reference_notional=100_000.0)

    table_a, oos_a, _ = run_walkforward_beta_refit(base, "Y", "X", **small_grid)
    table_b, oos_b, _ = run_walkforward_beta_refit(perturbed, "Y", "X", **small_grid)

    assert table_a.iloc[0]["fold_alpha"] == pytest.approx(table_b.iloc[0]["fold_alpha"])
    assert table_a.iloc[0]["fold_beta"] == pytest.approx(table_b.iloc[0]["fold_beta"])
    assert table_a.iloc[0]["selected_entry_z"] == table_b.iloc[0]["selected_entry_z"]
    assert table_a.iloc[0]["selected_exit_z"] == table_b.iloc[0]["selected_exit_z"]

    fold0_mask_a = oos_a.index <= fold0_test_end
    fold0_mask_b = oos_b.index <= fold0_test_end
    pd.testing.assert_series_equal(
        oos_a.loc[fold0_mask_a, "net_pnl_usd"],
        oos_b.loc[fold0_mask_b, "net_pnl_usd"],
        check_names=False,
    )


def test_beta_refit_walkforward_close_to_stable_beta_case() -> None:
    idx = pd.bdate_range("2015-01-01", "2019-12-31", tz="UTC")
    rng = np.random.default_rng(3)
    x = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, size=len(idx))), index=idx)
    true_beta = 1.5
    y = x * true_beta + 10.0 + rng.normal(0, 0.5, size=len(idx))
    panel = pd.DataFrame({"Y": y, "X": x}, index=idx)

    table, _, _ = run_walkforward_beta_refit(
        panel, "Y", "X", entry_grid=(1.0, 1.5), exit_grid=(0.25, 0.5), window=20, reference_notional=100_000.0
    )

    assert table["fold_beta"].sub(true_beta).abs().max() < 0.05


def test_run_walkforward_beta_refit_folds_match_build_walkforward_folds() -> None:
    idx = pd.bdate_range("2015-01-01", "2019-12-31", tz="UTC")
    panel = _synthetic_panel(idx, seed=5)
    expected_folds = build_walkforward_folds(idx)

    table, _, beta_folds = run_walkforward_beta_refit(
        panel, "Y", "X", entry_grid=(1.0, 1.5), exit_grid=(0.25, 0.5), window=20, reference_notional=100_000.0
    )

    assert len(table) == len(expected_folds)
    assert list(table["fold_id"]) == [f.fold_id for f in expected_folds]
    assert list(table["test_start"]) == [str(f.test_start.date()) for f in expected_folds]
    assert list(table["test_end"]) == [str(f.test_end.date()) for f in expected_folds]
    assert [bf.fold.fold_id for bf in beta_folds] == [f.fold_id for f in expected_folds]
