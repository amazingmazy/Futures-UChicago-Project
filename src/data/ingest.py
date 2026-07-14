"""Pull continuous daily futures settlement prices from Databento into data/raw and data/processed.

Uses Databento's continuous symbology with the calendar roll rule (``<root>.c.0``),
which rolls to the next contract on a fixed calendar schedule rather than by volume
or open interest. This keeps roll dates deterministic and reproducible across runs,
at the cost of occasionally rolling before/after the more liquid contract takes over.

Prices come from the ``statistics`` schema rather than ``ohlcv-1d``: the daily bar
close is just the last trade, while CME's official settlement price (what margining
and most academic futures studies use) is published as a statistics record.
"""

import time
from pathlib import Path

import databento as db
import pandas as pd
import requests
from dotenv import load_dotenv

# Errors worth retrying: Databento's own streaming errors, plus the raw
# requests/urllib3 connection errors (including transient DNS resolution
# failures) that were observed escaping past BentoError on this network.
RETRYABLE_ERRORS = (db.common.error.BentoError, requests.exceptions.RequestException)

DATASET = "GLBX.MDP3"  # CME Globex MDP 3.0 - covers all the futures roots below
SCHEMA = "statistics"  # carries CME's official settlement prices (ohlcv-1d close is only the last trade)

# Candidate futures roots from issue #3's pair-selection shortlist:
# ZQ/SR3 = Fed Funds vs SOFR, ZT/ZF/ZN/ZB = Treasury curve, CL/HO = crude vs heating oil.
ROOTS = ["ZQ", "SR3", "ZT", "ZF", "ZN", "ZB", "CL", "HO"]

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def get_client() -> db.Historical:
    """Build the Databento client, reading DATABENTO_API_KEY from .env.

    python-dotenv doesn't load .env into the environment automatically, so we
    call load_dotenv() first; db.Historical() then picks the key up from env.
    """
    load_dotenv()
    return db.Historical()


def _year_chunks(start: str, end: str):
    """Split a [start, end) date range into consecutive (year_start, year_end) string pairs.

    e.g. ("2015-01-01", "2017-06-01") -> [("2015-01-01","2016-01-01"),
    ("2016-01-01","2017-01-01"), ("2017-01-01","2017-06-01")]
    """
    years = pd.date_range(start, end, freq="YS").tolist()
    if not years or years[0] > pd.Timestamp(start):
        years.insert(0, pd.Timestamp(start))
    bounds = years + [pd.Timestamp(end)]
    for chunk_start, chunk_end in zip(bounds[:-1], bounds[1:]):
        yield chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")


def fetch_continuous_daily(
    client: db.Historical, root: str, start: str, end: str, retries: int = 5
) -> pd.DataFrame:
    """Fetch a continuous daily series for one futures root, one year at a time.

    Requesting the full multi-year range in a single call was observed to hang
    or drop the connection on this network. Chunking by calendar year keeps
    each request small enough to complete reliably; failed chunks (connection
    resets, transient DNS lookup failures, etc.) are retried with linear
    backoff before giving up.
    """
    parts = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        for attempt in range(retries):
            try:
                data = client.timeseries.get_range(
                    dataset=DATASET,
                    symbols=[f"{root}.c.0"],
                    stype_in="continuous",
                    schema=SCHEMA,
                    start=chunk_start,
                    end=chunk_end,
                )
                parts.append(data.to_df())
                break  # chunk succeeded, move on to the next one
            except RETRYABLE_ERRORS:
                if attempt == retries - 1:
                    raise  # out of retries, let the caller see the failure
                time.sleep(2 * (attempt + 1))  # 2s, 4s, 6s, ... before retrying

    # Concatenate the per-year chunks back into one continuous frame. Chunk
    # boundaries can overlap by a day; distinct statistics records can share a
    # receive timestamp, so drop fully duplicated rows rather than index dupes.
    df = pd.concat(parts).sort_index()
    return df[~df.reset_index().duplicated().to_numpy()]


def settlement_series(stats: pd.DataFrame) -> pd.Series:
    """Extract the daily settlement price series from a raw statistics frame.

    The statistics feed carries many stat types (open interest, session
    high/low, ...); settlements are stat_type == SETTLEMENT_PRICE. CME
    publishes preliminary and final settlements under the same type, so keep
    the last record per session date (ts_ref), which is the final value.
    """
    settle = stats[stats["stat_type"] == db.StatType.SETTLEMENT_PRICE]
    session_date = settle["ts_ref"].dt.normalize()
    return settle.groupby(session_date)["price"].last().rename_axis("date")


def build_settlement_panel(raw_by_root: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine each root's raw statistics frame into one wide panel of settlement prices.

    Output is date-indexed with one column per root (e.g. "ZN", "ZB"), which is
    the shape issue #3 (EDA/pair selection) and #4 (spread modeling) need.
    """
    settlements = {root: settlement_series(df) for root, df in raw_by_root.items()}
    return pd.DataFrame(settlements).sort_index()


def main(start: str = "2015-01-01", end: str = "2026-07-01") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    client = get_client()
    raw_by_root = {}
    for root in ROOTS:
        print(f"Fetching {root}.c.0 ({start} to {end})...")
        df = fetch_continuous_daily(client, root, start, end)
        # Raw per-symbol pull, unmodified aside from concatenation - keeps the
        # exact statistics records (all stat types) Databento returns.
        df.to_parquet(RAW_DIR / f"{root}.c.0.parquet")
        raw_by_root[root] = df

    # Processed settlement-price panel: the single file downstream EDA/modeling
    # steps should read from, instead of re-assembling it from the raw files.
    panel = build_settlement_panel(raw_by_root)
    panel.to_parquet(PROCESSED_DIR / "continuous_settlement_prices.parquet")
    print(f"Saved {len(ROOTS)} raw series to {RAW_DIR} and settlement panel to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
