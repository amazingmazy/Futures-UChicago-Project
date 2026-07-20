# Pair Selection

*Issue #3 — Exploratory Data Analysis and Pair Selection*

## Recommendation

**Trade ZQ/SR3 (Fed Funds vs SOFR)** — the pair the README already assumed.

That sounds like a null result. It isn't. The job of #3 was to *test* the
assumption rather than inherit it, and the testing surfaced four things that
change what #4 should do.

All numbers below are computed on **official CME settlement prices** (the
`statistics` schema), not last-trade closes — settlements are what margining
uses and they are noticeably cleaner for the policy-linked contracts.

Reproduce with:

```bash
uv run python -m src.analysis.exploratory_analysis
```

---

## The ranking

| Pair             | n     | Start      | β    | R²   | Δ-corr        | EG p             | ADF p            | Half-life       | Stale         |
| ---------------- | ----- | ---------- | ----- | ----- | -------------- | ---------------- | ---------------- | --------------- | ------------- |
| **ZQ/SR3** | 2,052 | 2018-05-04 | 0.986 | 0.989 | **0.01** | **0.0003** | **0.0000** | **12.4d** | **48%** |
| ZT/ZF            | 2,895 | 2014-12-31 | 0.458 | 0.966 | 0.909          | 0.395            | 0.186            | 144.6d          | 4%            |
| ZF/ZN            | 2,895 | 2014-12-31 | 0.683 | 0.984 | 0.939          | 0.359            | 0.163            | 94.5d           | 2%            |
| ZN/ZB            | 2,895 | 2014-12-31 | 0.451 | 0.958 | 0.866          | 0.049            | 0.013            | 85.5d           | 2%            |
| CL/HO            | 2,894 | 2014-12-31 | 23.0  | 0.883 | 0.526          | 0.016            | 0.004            | 14.2d           | 1%            |

`Δ-corr` is the correlation of daily **changes**, not levels. Every R² in that
table is above 0.89 — level correlation on trending series flags everything as a
good pair, so it can't be used as a screen.

**The Treasury curve pairs fail cointegration** (p ≈ 0.36–0.40) despite their
excellent level fits. So "2Y vs 5Y" and "10Y vs 30Y" from the #3 shortlist are out.

**ZQ/SR3 wins, and not narrowly** — EG p = 0.0003, ADF 0.0000, half-life an order
of magnitude shorter than any curve pair's.

But its other two columns are strange: **Δ-corr of 0.01 alongside R² of 0.99, and
48% staleness.** The four caveats below are what that turned out to mean.

---

## Caveat 1 — the sample starts May 2018, not 2015

SOFR futures didn't exist before then. `ingest.py` requests 2015 onward, gets
empty responses for three years, and appends the empty frames without comment.
The resulting NaNs are indistinguishable from missing data.

**Usable ZQ/SR3 overlap: 2,052 days, against 2,895 for ZN/ZB.** About 70% of the
sample, and nothing in the pipeline says so.

## Caveat 2 — both legs are stale roughly half the time

| Root | Days with zero settlement change |
| ---- | -------------------------------- |
| ZQ   | **56.8%**                  |
| SR3  | **37.1%**                  |
| ZT   | 4.5%                             |
| ZN   | 1.9%                             |
| ZB   | 1.3%                             |

Front-month short-rate contracts are pinned to a policy rate that only moves at
eight scheduled FOMC meetings a year. Between meetings they sit still.

**This explains the R²=0.99 / Δ-corr=0.01 signature.** The levels track because
both legs are anchored to the same policy rate. The daily changes don't, because
on most days at least one leg didn't move.

→ `outputs/figures/03_staleness.png`

## Caveat 3 — no pair survives 2023, including this one

Engle-Granger p, re-run on later subsamples:

| Pair             | from 2015        | from 2018        | from 2022 | from 2023 |
| ---------------- | ---------------- | ---------------- | --------- | --------- |
| **ZQ/SR3** | **0.0003** | **0.0003** | **0.0131** | 0.2299    |
| ZT/ZF            | 0.3808           | 0.5035           | 0.7350    | 0.6993    |
| ZF/ZN            | 0.3602           | 0.5747           | 0.5826    | 0.6791    |
| ZN/ZB            | 0.0463           | 0.5030           | 0.3151    | 0.4880    |

Two different things here.

**ZN/ZB never had a relationship.** Its 0.046 is carried entirely by the quiet
2015–17 window and collapses to 0.32–0.50 once that window is dropped. The p-value
is an artifact.

**ZQ/SR3 had one, and it has weakened** — it still passes from 2022 (p = 0.013)
but fails from 2023 (p = 0.23). Expected, after the 2022–23 hiking cycle repriced
the front end.

The rolling hedge ratio agrees: **β swings −0.46 → +2.40** on 252-day windows.

**So "ZQ/SR3 is cointegrated" is a claim about 2018–2021, not about today.**

## Caveat 4 — the stationary spread is the *unhedged* one

ZQ and SR3 have different DV01s:

- **ZQ** — 30-day, $5m notional → **$41.67/bp**
- **SR3** — 3-month, $1m notional → **$25.00/bp**

A 1:1 spread is therefore **not market-neutral** — it carries net exposure to the
*level* of rates, exactly the directional risk the strategy is meant to remove.
DV01-flat requires β = 25.00/41.67 = **0.60**.

Test all three constructions:

| Construction | β    | σ (bp) | ADF p            | Half-life        | Hedged? |
| ------------ | ----- | ------- | ---------------- | ---------------- | ------- |
| Raw 1:1      | 1.00  | 20.5    | **0.0000** | **12.4d**  | No      |
| OLS β       | 0.986 | 20.3    | **0.0000** | **12.4d**  | No      |
| DV01-neutral | 0.60  | 78.2    | **0.7497** | **302.4d** | Yes     |

**The spread that mean-reverts is not hedged. The spread that is hedged does not
mean-revert.**

The OLS regression recovers β ≈ 0.99 — essentially 1:1 — *because that's the
combination that happens to be stationary*. It finds the unhedged ratio and calls
it a hedge ratio.

→ `outputs/figures/05_spread_constructions.png`

There's no clean answer. #4 has to pick one and be explicit about the trade-off.

---

## Handoff

**Issue #4 — model the relationship**

1. Use the **2018-05 → present** sample.
2. Work in **rate space** (`100 - price`), so β reads bp-per-bp and the spread is
   in basis points.
3. **Decide the DV01 question** (Caveat 4) and write the decision down.
4. **The time-varying / Bayesian model the README lists as optional isn't optional
   for this pair.** β moves −0.46 to +2.40 and cointegration decays from 2023. A
   static hedge ratio fits the average of a relationship that doesn't hold still.

**Issue #5 — signal generation**

Spread σ by year: **2021 = 1.47bp. 2022 = 40.50bp.** A 27× swing.

A `|z| > 2` rule on full-sample statistics will never fire in 2021 and will fire
constantly in 2022. **Use a rolling window.** `plot_spread_with_zscore` already
does (126-day).

With 48% staleness, consider gating entries on days where at least one leg moved.

---

## Data cleaning

**The settlement panel needs almost none.** Rows are keyed by settlement session
date, so the Sunday-evening Globex bars that plagued the old close-price panel
(592 spurious rows, hundreds of injected NaNs) simply don't exist here — the
Sunday-drop step in the code is now a no-op kept as a guard. Outside SR3's
pre-2018 gap, every root is missing at most 2 of 2,896 dates.

**One quirk left alone:**

- **CL settled at −$37.63 on 2020-04-20.** Real (the negative-WTI episode), not a
  bug. It makes `np.log(CL)` silently NaN. Doesn't affect our pair, but worth
  knowing.

---

## Reproducing

```bash
uv run python -m src.data.ingest              # 1-5h depending on connection, needs DATABENTO_API_KEY; resumable if interrupted
uv run python -m src.analysis.exploratory_analysis   # ~30s, reads the saved panel
uv run pytest tests/ -q                       # 19 tests, no API key needed
```

Tests run on synthetic fixtures with known properties, so they work on a fresh
clone without Databento access.
