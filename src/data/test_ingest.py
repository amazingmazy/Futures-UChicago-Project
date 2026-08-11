"""Ad-hoc smoke test: pull a small ZQ sample from Databento and print it.

Not part of the ingest pipeline - just a quick way to eyeball what
fetch_continuous_daily() returns before trusting it in main().
"""

import pandas as pd

from src.data.ingest import fetch_continuous_daily, get_client

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

if __name__ == "__main__":
    client = get_client()
    df = fetch_continuous_daily(client, "ZQ", "2024-01-01", "2024-01-15")

    print(df.dtypes)
    print()
    print(f"{len(df)} rows")
    print()
    print(df)
