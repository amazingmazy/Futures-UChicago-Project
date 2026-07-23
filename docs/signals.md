# Signal Generation and Strategy Logic

*Issue #5 — convert the modeled CL/BZ spread into a reproducible market-neutral trading rule.*

This stage consumes the spread-model artifact from issue #4 and writes a clean signal table for the backtest stage. It does **not** compute performance; issue #6 should take these positions, apply execution/cost assumptions, and evaluate PnL.

Reproduce after the spread model has been run:

```bash
uv run python -m src.models.spread
uv run python -m src.strategy.signals
```

## Baseline rule

The model spread is

\[
\text{spread}_t = \text{CL}_t - \alpha - \beta \, \text{BZ}_t.
\]

The signal step converts that spread into a rolling z-score:

\[
z_t = \frac{\text{spread}_t - \mu_{t-1,126}}{\sigma_{t-1,126}}
\]

where the rolling mean and standard deviation use the previous 126 sessions only. That one-session shift is important: today's settlement can trigger a signal, but it is not allowed to help define the benchmark used to judge itself.

Default thresholds in `src/strategy/signals.py`:

| Parameter | Default | Meaning |
|---|---:|---|
| `window` | 126 sessions | about six trading months of recent spread behavior |
| `entry_z` | 1.5 | enter when the spread is at least 1.5 rolling standard deviations from normal |
| `exit_z` | 0.5 | exit once the spread has reverted close to the rolling mean |

The thresholds are intentionally configurable. The spread-model document reports a 4.1-day full-sample half-life and an 8–9 day recent half-life, so a first backtest should not depend only on rare two-sigma events before it can learn anything about the trade.

## Position convention

The signal file uses a simple spread-position convention:

| `target_position` | Interpretation | CL contracts | BZ contracts |
|---:|---|---:|---:|
| `+1` | long spread | `+1` | `-beta` |
| `-1` | short spread | `-1` | `+beta` |
| `0` | flat | `0` | `0` |

A long spread means WTI/CL is cheap versus Brent/BZ under the fitted relationship. A short spread means WTI/CL is rich versus Brent/BZ. The hedge ratio is read from `spread_model_cl_bz.json`, so this stage stays tied to the model actually fitted in issue #4.

## Output contract for the backtest

`data/processed/signals_cl_bz.parquet` contains one row per spread-model session:

| Column | Meaning |
|---|---|
| `cl`, `bz`, `spread` | copied from the issue #4 spread artifact when available |
| `zscore` | rolling shifted z-score used for signals |
| `target_position` | position decided after observing that row's settlement |
| `next_session_position` | previous row's target, for a backtest that trades one session after the signal |
| `position_change` | change in target position on that row |
| `entry`, `exit` | booleans marking transitions in/out of trades |
| `signal` | one of `enter_long_spread`, `enter_short_spread`, `exit`, or `hold` |
| `cl_contracts`, `bz_contracts` | hedge-leg targets implied by the fitted beta |

The `next_session_position` column is the safest input for issue #6 if the team assumes signals are observed after settlement and executed on the next tradable session. Using same-day settlement-to-settlement returns would leak information.

## Risk handoff

This PR leaves three choices explicit for the performance stage:

1. Execution timing: use `next_session_position` unless the backtest has a defensible same-day execution assumption.
2. Transaction costs: each nonzero `position_change` implies leg turnover in CL and BZ.
3. April 2020: the negative-WTI week should be handled deliberately in the backtest, either by accepting it as a stress event, filtering it, or adding a risk cap.

The tests cover the z-score lag, entry/exit state machine, hedge-leg construction, and summary output on synthetic data, so this stage can be reviewed without a Databento key.
