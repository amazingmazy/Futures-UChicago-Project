# Pair Selection

*Issue #3 — Exploratory Data Analysis and Pair Selection*

## Recommendation

**Trade ZQ/SR3 (Fed Funds vs SOFR)** — the pair the README already assumed.

That sounds like a null result. It isn't. The job of #3 was to *test* the
assumption rather than inherit it, and the testing surfaced four things that
change what #4 should do.

Reproduce with:

```bash
uv run python -m src.analysis.exploratory_analysis
```

---

## The ranking

| Pair | n | Start | β | R² | Δ-corr | EG p | ADF p | Half-life | Stale |
|---|---|---|---|---|---|---|---|---|---|
| **ZQ/SR3** | 1,645 | 2018-05-07 | 0.986 | 0.990 | **0.10** | **0.0000** | **0.0000** | **10.0d** | **42%** |
| ZT/ZF | 2,854 | 2015-01-01 | 0.459 | 0.965 | 0.878 | 0.363 | 0.165 | 110.6d | 4% |
| ZF/ZN | 2,927 | 2015-01-01 | 0.683 | 0.984 | 0.896 | 0.373 | 0.172 | 57.7d | 2% |
| ZN/ZB | 2,983 | 2015-01-01 | 0.451 | 0.958 | 0.836 | 0.054 | 0.015 | 76.4d | 2% |
| CL/HO | 2,980 | 2015-01-01 | 22.9 | 0.890 | 0.677 | 0.014 | 0.003 | 29.1d | 0% |

`Δ-corr` is the correlation of daily **changes**, not levels. Every R² in that
table is above 0.89 — level correlation on trending series flags everything as a
good pair, so it can't be used as a screen.

**The Treasury curve pairs fail cointegration** (p ≈ 0.36) despite their excellent
level fits. So "2Y vs 5Y" and "10Y vs 30Y" from the #3 shortlist are out.

**ZQ/SR3 wins, and not narrowly** — EG and ADF both 0.0000, half-life six times
shorter than anything else.

But its other two columns are strange: **Δ-corr of 0.10 alongside R² of 0.99, and
42% staleness.** The four caveats below are what that turned out to mean.

---

## Caveat 1 — the sample starts May 2018, not 2015

SOFR futures didn't exist before then. `ingest.py` requests 2015 onward, gets
`symbol did not resolve: SR3.c.0` for three years, and appends the empty frames
without comment (`ingest.py:80-81`). The resulting NaNs are indistinguishable from
missing data.

**Usable ZQ/SR3 overlap: 1,645 days, against 2,983 for ZN/ZB.** Half the sample,
and nothing in the pipeline says so.

## Caveat 2 — both legs are stale ~40% of the time

| Root | Days with zero price change |
|---|---|
| ZQ | **46.0%** |
| SR3 | **37.5%** |
| ZT | 4.4% |
| ZN | 1.6% |
| ZB | 1.7% |

Front-month short-rate contracts are pinned to a policy rate that only moves at
eight scheduled FOMC meetings a year. Between meetings they sit still.

**This explains the R²=0.99 / Δ-corr=0.10 signature.** The levels track because
both legs are anchored to the same policy rate. The daily changes don't, because
on most days at least one leg didn't move.

→ `outputs/figures/03_staleness.png`

## Caveat 3 — no pair survives 2022, including this one

Engle-Granger p, re-run on later subsamples:

| Pair | from 2015 | from 2018 | from 2022 | from 2023 |
|---|---|---|---|---|
| **ZQ/SR3** | **0.0000** | **0.0000** | 0.0618 | 0.2799 |
| ZT/ZF | 0.3634 | 0.5012 | 0.7219 | 0.7634 |
| ZF/ZN | 0.3732 | 0.5509 | 0.3166 | 0.5676 |
| ZN/ZB | 0.0543 | 0.5368 | 0.3062 | 0.4601 |

Two different things here.

**ZN/ZB never had a relationship.** Its 0.054 is carried entirely by the quiet
2015–17 window and collapses to 0.31–0.54 once that window is dropped. The p-value
is an artifact.

**ZQ/SR3 had one, and it has weakened** — expected, after the 2022–23 hiking cycle
repriced the front end.

The rolling hedge ratio agrees: **β swings 0.27 → 1.36** on 252-day windows.

**So "ZQ/SR3 is cointegrated" is a claim about 2018–2021, not about today.**

## Caveat 4 — the stationary spread is the *unhedged* one

ZQ and SR3 have different DV01s:

- **ZQ** — 30-day, $5m notional → **$41.67/bp**
- **SR3** — 3-month, $1m notional → **$25.00/bp**

A 1:1 spread is therefore **not market-neutral** — it carries net exposure to the
*level* of rates, exactly the directional risk the strategy is meant to remove.
DV01-flat requires β = 25.00/41.67 = **0.60**.

Test all three constructions:

| Construction | β | σ (bp) | ADF p | Half-life | Hedged? |
|---|---|---|---|---|---|
| Raw 1:1 | 1.00 | 19.8 | **0.0000** | **10.1d** | No |
| OLS β | 0.986 | 19.6 | **0.0000** | **10.0d** | No |
| DV01-neutral | 0.60 | 79.0 | **0.7410** | **276.8d** | Yes |

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
   for this pair.** β moves 0.27–1.36 and cointegration decays post-2022. A static
   hedge ratio fits the average of a relationship that doesn't hold still.

**Issue #5 — signal generation**

Spread σ by year: **2021 = 1.45bp. 2022 = 39.34bp.** A 27× swing.

A `|z| > 2` rule on full-sample statistics will never fire in 2021 and will fire
constantly in 2022. **Use a rolling window.** `plot_spread_with_zscore` already
does (126-day).

With 42% staleness, consider gating entries on days where at least one leg moved.

---

## Data cleaning

**Sunday sessions dropped (592 rows).** Globex reopens Sunday evening, so
`ohlcv-1d` emits a Sunday bar. They're real but thin, and populated inconsistently
across roots — ZQ has no Sunday close on 55% of Sundays, ZN on 6%. Since
`build_close_panel` aligns on the union of indices, keeping them injects spurious
NaNs:

| Root | NaN before | NaN after |
|---|---|---|
| ZQ | 345 | 20 |
| ZT | 242 | 99 |
| ZN | 36 | 2 |
| ZB | 51 | 2 |

Every "missing" date in the Feb 2026 ingestion warnings is a Sunday.

**Two quirks left alone:**

- The `degraded` warnings are feed-wide GLBX quality flags, not per-contract — the
  identical dates appear on every root. 8 days out of ~2,900.
- **CL closed at −$2.67 on 2020-04-20.** Real (WTI went negative), not a bug. It
  makes `np.log(CL)` silently NaN. Doesn't affect our pair, but worth knowing.

---

## Reproducing

```bash
uv run python -m src.data.ingest              # ~1 hour, needs DATABENTO_API_KEY
uv run python -m src.analysis.exploratory_analysis   # ~30s, reads the saved panel
uv run pytest tests/ -q                       # 19 tests, no API key needed
```

Tests run on synthetic fixtures with known properties, so they work on a fresh
clone without Databento access.
