"""Shared access to the processed settlement-price panel.

Every stage downstream of ingestion (EDA, spread modeling, signals, backtest)
reads the same wide panel written by ``src.data.ingest`` and writes its outputs
under the same ``outputs/`` tree. This module owns those paths and the loader,
so no stage has to import them from another stage's analysis script.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"

PANEL_PATH = PROCESSED_DIR / "continuous_settlement_prices.parquet"


def load_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    """Load the wide settlement-price panel written by ``src.data.ingest``.

    Raises a pointed error rather than a bare ``FileNotFoundError``, so a
    teammate who clones the repo and runs this first is told what to do next.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No price panel at {path}.\n"
            "Run the ingestion step first:\n"
            "    uv run python -m src.data.ingest"
        )
    return pd.read_parquet(path)


def drop_sunday_session(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop any Sunday-evening Globex session rows.

    CME Globex reopens Sunday evening US time, so bar-style schemas emit rows
    dated Sunday, populated inconsistently across roots, which injects spurious
    NaNs when the panel aligns roots on the union of their indices. The
    settlement panel keys rows by settlement session date and contains no
    Sundays, so this is a no-op there -- kept as a cheap guard in case the
    panel is ever rebuilt from a bar schema again.
    """
    return panel.loc[panel.index.dayofweek != 6]
