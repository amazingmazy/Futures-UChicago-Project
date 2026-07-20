# Pair Selection

*Issue #3 — Exploratory Data Analysis and Pair Selection*

## Recommendation

**Trade CL/BZ (WTI vs Brent crude).**

The question this analysis answers: *which two futures contracts should we
model and potentially trade against each other?* Nine CME roots were ingested
from Databento's `GLBX.MDP3` dataset — short-rate futures (Fed Funds `ZQ`,
SOFR `SR3`), the Treasury curve (`ZT`, `ZF`, `ZN`, `ZB`), and energy (`CL`
WTI, `HO` heating oil, `BZ` Brent) — and six candidate pairs were run through
the same statistical battery on **official CME settlement prices**:
Engle-Granger and ADF cointegration tests, subsample robustness,
mean-reversion half-life, daily-change correlation against level R², rolling
hedge-ratio stability, and staleness.

CL/BZ dominates every candidate on every dimension that matters, and it is
the only pair whose cointegration survives outside any single regime. ZQ/SR3
— the natural first candidate for a rates project, and the pair the project
brief originally assumed — is the runner-up; the tables below document
exactly why it lost.

BZ here is CME's Brent Last Day Financial contract, cash-settled against the
ICE Brent Index — which makes a WTI/Brent pair testable from the same
Databento dataset without licensing a second data feed.

Reproduce with:

```bash
uv run python -m src.analysis.exploratory_analysis
```

---

## The ranking (settlement prices)

| Pair            | n     | Start      | β              | R²   | Δ-corr        | EG p             | ADF p            | Half-life      | Stale          |
| --------------- | ----- | ---------- | --------------- | ----- | -------------- | ---------------- | ---------------- | -------------- | -------------- |
| **CL/BZ** | 2,894 | 2014-12-31 | **0.975** | 0.977 | **0.70** | **0.0000** | **0.0000** | **4.1d** | **0.6%** |
| ZQ/SR3          | 2,052 | 2018-05-04 | 0.986           | 0.989 | 0.01           | 0.0003           | 0.0000           | 12.4d          | 48%            |
| ZT/ZF           | 2,895 | 2014-12-31 | 0.458           | 0.966 | 0.909          | 0.395            | 0.186            | 144.6d         | 4%             |
| ZF/ZN           | 2,895 | 2014-12-31 | 0.683           | 0.984 | 0.939          | 0.359            | 0.163            | 94.5d          | 2%             |
| ZN/ZB           | 2,895 | 2014-12-31 | 0.451           | 0.958 | 0.866          | 0.049            | 0.013            | 85.5d          | 2%             |
| CL/HO           | 2,894 | 2014-12-31 | 23.0            | 0.883 | 0.526          | 0.016            | 0.004            | 14.2d          | 1%             |

`Δ-corr` is the correlation of daily **changes**, not levels — every R² in the
table is above 0.88, so level correlation cannot be used as a screen.

CL/BZ is the only pair that is strong on *both* measures at once: the levels
are cointegrated (EG p = 0.0000) **and** the legs genuinely co-move day to day
(Δ-corr 0.70). ZQ/SR3 has the level fit but no daily co-movement (0.01); the
Treasury pairs have the daily co-movement but fail cointegration (p ≈ 0.36–0.40).

## The decider — cointegration robustness by subsample

Engle-Granger p-value, re-run on progressively later windows:

| Pair            | from 2015        | from 2018        | from 2022        | from 2023        |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| **CL/BZ** | **0.0000** | **0.0003** | **0.0087** | **0.0192** |
| ZQ/SR3          | 0.0003           | 0.0003           | 0.0131           | 0.2299           |
| ZN/ZB           | 0.0463           | 0.5030           | 0.3151           | 0.4880           |
| ZT/ZF           | 0.3808           | 0.5035           | 0.7350           | 0.6993           |
| CL/HO           | 0.0153           | 0.0074           | 0.1945           | 0.4800           |

**CL/BZ is the only pair that stays under 0.05 on every window.** Later windows
are shorter and have less power, so some upward drift is mechanical — what
matters is staying on the right side of the threshold, and only CL/BZ does.

The economics explain why. ZQ/SR3's relationship was underwritten by a policy
regime that the 2022–23 hiking cycle disturbed. CL/BZ's is underwritten by a
physical arbitrage — WTI and Brent are two grades of light sweet crude, and if
the spread widens past freight economics, shipping crude across the Atlantic
pulls it back. Regimes change; tankers keep sailing.

## Why CL/BZ is also the *practical* choice

1. **β ≈ 1 with identical contract specs.** Both legs are 1,000 barrels quoted
   in $/bbl, so the ~1:1 spread that is stationary is also (approximately) the
   dollar-neutral one. The DV01 dilemma that made ZQ/SR3 painful — the
   stationary spread is unhedged, the hedged spread isn't stationary — simply
   does not exist for this pair.
2. **The hedge ratio holds still.** Rolling 252-day β stays in 0.74–1.14.
   ZQ/SR3's swings from −0.46 to +2.40, which would have made the
   time-varying/Bayesian extension mandatory. For CL/BZ a static ratio from
   plain OLS is defensible, which keeps issue #4 simple.
3. **No staleness.** Both legs settle to a new price on >99% of days. Nearly
   every observation carries information, and no signal-gating logic is needed.
4. **Fastest mean reversion in the table**: half-life 4.1 days, so capital is
   in a trade for days, not months.
5. **Position sizing is trivial**: 1 lot vs 1 lot.
6. **The story fits in one sentence** — useful for the final presentation. The
   spread even has a household name ("the WTI–Brent spread") and is quoted in
   $/bbl on financial TV.

→ `outputs/figures/07_spread_cl_bz.png`

## Honest caveats on CL/BZ

- **April 2020.** WTI settled at **−$37.63** in the front-month expiry squeeze
  at Cushing (the settlement panel carries it on the 2020-04-17 and 2020-04-20
  sessions) while Brent stayed positive around $26–28; the 1:1 spread bottomed
  at **−$62**. It is real, not a data error, and it single-handedly explains
  the 2020 row in the volatility table. Issue #5's z-score should use rolling
  statistics partly for this reason; issue #6 should look at how the strategy
  behaves through that week. It also makes `np.log(CL)` silently NaN — model
  in price space, not log space.
- **Spread dispersion still varies by year** — σ ranges from $0.74 (2024) to
  $5.73 (2020, the COVID year), with 2026 elevated at $3.08. A 4–8× swing,
  far milder than ZQ/SR3's 27× but still an argument for a rolling z-score
  rather than a full-sample one.
- **The relationship has broken before, outside our sample.** During the
  2011–2015 US crude-export ban the spread blew out to over $25 because the
  arbitrage physically could not operate. The ban was lifted in December 2015;
  our sample is almost entirely post-ban. Worth one slide of humility: the
  leash is physical, but policy can cut it.
- **BZ is a proxy for ICE Brent.** It is cash-settled against the ICE Brent
  Index, so it tracks the benchmark closely (and avoids licensing a second
  data feed), but it is less liquid than ICE BRN. CL is physically delivered,
  BZ cash-settled — irrelevant for daily settlement-based modeling, worth
  knowing at expiry.

---

## The runner-up: ZQ/SR3, and why it lost

Fed Funds vs SOFR looks perfect on paper: both legs are anchored to the same
policy rate, the levels track almost exactly (R² = 0.99), and the full-sample
cointegration tests pass. The details are what disqualify it — CL/BZ beats it
on every dimension:

1. **Cointegration decays out of sample**: EG p = 0.0003 full-sample →
   0.2299 from 2023. It is a claim about 2018–2021, not about today.
2. **Short sample**: SOFR futures only exist from May 2018 — 2,052 usable
   days against CL/BZ's 2,894.
3. **Staleness**: ZQ's settlement is unchanged on 57% of days and SR3's on
   37%, because front-month short-rate contracts are pinned to a policy rate
   that only moves at eight scheduled FOMC meetings a year. The
   daily-change correlation is 0.01 against a level R² of 0.99: the legs
   almost never actually move together on a given day.
4. **The DV01 dilemma is unresolved and unresolvable**: the stationary ~1:1
   spread is not rate-neutral (ZQ DV01 $41.67/bp vs SR3 $25.00/bp; neutral
   requires β = 0.60), and the DV01-neutral spread is not stationary
   (ADF p = 0.75, half-life 302 days). Whichever construction issue #4 picked,
   something important would have been given up.
5. **Unstable hedge ratio**: rolling β from −0.46 to +2.40, making the
   optional time-varying model mandatory.

The spread-constructions table that documents point 4 (`spread_constructions.csv`,
`outputs/figures/05_spread_constructions.png`) is kept in the codebase as the
record of why a rates pair that looks perfect on levels was passed over.

**The Treasury curve pairs** (ZT/ZF, ZF/ZN, ZN/ZB) fail Engle-Granger outright
despite beautiful level fits and high Δ-corr: the curve's slope trends through
policy regimes rather than reverting to a fixed level. ZN/ZB's marginal
full-sample p = 0.046 collapses to 0.31–0.50 the moment the quiet 2015–17
window is dropped — an artifact, not a relationship.

---

## Why settlement prices, not closes

The panel is built from Databento's `statistics` schema, which carries CME's
official daily settlement prices — the prices margining actually uses. The
alternative, the `ohlcv-1d` bar close, is just the last trade of the session,
which for thin policy-linked contracts can be hours old. The choice is not
cosmetic; the same battery run on closes gives materially different numbers:

| Statistic                 | Last-trade closes | Official settlements |
| ------------------------- | -------------- | --------------- |
| ZQ/SR3 Δ-corr            | 0.10           | 0.01            |
| ZQ stale fraction         | 46%            | 57%             |
| ZQ/SR3 rolling β range   | 0.27 → 1.36   | −0.46 → +2.40 |
| ZQ/SR3 EG p (full sample) | 0.0000         | 0.0003          |
| ZQ/SR3 half-life          | 10.0d          | 12.4d           |

Close prices flatter the policy-linked contracts by mistaking last-trade
noise for daily movement; settlements show them as they are — *stickier*.

Two structural bonuses: the settlement panel has **no Sunday rows**
(settlements are keyed by session date, whereas daily bars emit a thin
Sunday-evening Globex bar that injects spurious NaNs), and outside SR3's
pre-2018 gap every root is missing at most 2 of 2,896 sessions, starting from
the 2014-12-31 session.

---

## Handoff

**Issue #4 — model the relationship**

1. Model **CL/BZ in price space** ($/bbl). No rate conversion, no DV01
   arithmetic — and no logs (April 2020 is negative).
2. A **static OLS hedge ratio (β ≈ 0.97)** is a defensible baseline; rolling β
   stays in 0.74–1.14. The time-varying/Bayesian model is a genuine optional
   extension, not a requirement.
3. Fit on the full 2015→present sample; check sensitivity by re-fitting from
   2022 (the relationship holds there too: EG p = 0.0087).

**Issue #5 — signal generation**

1. Spread σ by year runs $0.74–$5.73 — use a **rolling mean/σ for the z-score**
   (the 126-day window in `plot_spread_with_zscore` is a reasonable start).
2. With a 4.1-day half-life, expect short holding periods; entry/exit
   thresholds can be tested tighter than the |z| > 2 folklore.
3. Decide explicitly how the April 2020 week is handled (trade through it,
   filter it, or cap position size) and write the decision down.

---

## Reproducing

```bash
uv run python -m src.data.ingest                      # statistics schema, 9 roots; 1-5h depending on connection, resumable
uv run python -m src.analysis.exploratory_analysis    # ~40s, reads the saved settlement panel
uv run pytest tests/ -q                               # 20 tests, no API key needed
```

Ingestion needs `DATABENTO_API_KEY` in `.env`; already-downloaded roots are
skipped on re-runs, so adding a root only fetches the new one. Tests run on
synthetic fixtures with known properties, so they work on a fresh clone
without Databento access.
