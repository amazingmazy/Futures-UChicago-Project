# Relative-Value Trading in CME Interest-Rate Futures Using Databento Data

## Project Overview

This project studies whether economically related CME interest-rate futures exhibit stable relative-value relationships that can be modeled and potentially traded. The initial focus is on futures contracts connected to short-term interest-rate expectations, especially Fed Funds futures and SOFR futures. Treasury futures may also be explored as a comparison or extension.

The central research question is the following:
Can we identify a stable relationship between two CME interest-rate futures contracts, model deviations from that relationship, and use those deviations to construct a market-neutral trading strategy?

## Motivation

Interest-rate futures are linked because they all reflect market expectations about interest rates, monetary policy, and future Federal Reserve decisions. Since these contracts are economically related, they should not move independently. If two contracts usually move together but temporarily diverge, that divergence may represent a relative-value opportunity.

This idea is similar to pairs trading. In a pairs strategy, the goal is not simply to predict whether the whole market will rise or fall. Instead, the goal is to trade the relationship between two related instruments. If one contract appears expensive relative to another, the strategy can short the expensive contract and go long the cheap contract. If the relationship later returns to normal, the trade may be profitable.

## Related Work

This project is inspired by two main ideas from the reference materials in the course Canvas.

1. The Bayesian pairs-trading paper motivates the use of cointegration and spread modeling. Correlation only measures short-term co-movement, while cointegration asks whether two price series share a longer-term equilibrium relationship. If two futures contracts are cointegrated, then the spread between them may be mean-reverting. This makes the pair a possible candidate for a relative-value strategy.

2. The Treasury futures roll paper provides useful background on futures-specific issues, especially contract rolls and calendar spreads. Futures contracts expire, so any futures strategy must carefully handle rolling from one contract to the next. For this project, roll timing and calendar-spread behavior are not the main focus, but they may become an extension if they appear important during the data exploration stage.

## Data

The project will use CME futures data from Databento. The initial data work will focus on identifying which contracts and date ranges are available.

Candidate products include:

* Fed Funds futures
* SOFR futures
* Treasury futures

The first stage of the project will determine whether we have enough data history for a longer-horizon cointegration analysis. If the available data is shorter or more intraday focused, the project may emphasize shorter horizon spread behavior instead.

## Methodology

The project will proceed in six main stages.

### 1. Project Setup

We will create a shared Python project structure with separate folders for source code, notebooks, data, documentation, and outputs. The repository should be easy for all team members to clone, run, and contribute to.

### 2. Data Ingestion from Databento

We will build a reproducible pipeline to pull CME futures data from Databento and store cleaned outputs locally. This step includes connecting with a Databento API key, selecting the relevant futures contracts, downloading historical data, and deciding how to handle contract rolls.

### 3. Exploratory Data Analysis and Pair Selection

We will explore candidate futures contracts and decide which pair is most appropriate for the strategy. This step includes plotting prices, computing returns, comparing correlations, building spreads, and testing whether candidate pairs show stable relationships.

The main goal of this stage is to answer: **Which two futures contracts should we model and potentially trade against each other?**

The initial candidate pair is Fed Funds futures versus SOFR futures because both are closely tied to short-term interest rate expectations.

Assess the pairs rigorously using the following:

- Engle–Granger cointegration test
- Augmented Dickey-Fuller test
- Analysis of half-life
- Analysis of long-term two-way regression
- Analysis of return correlation
- Staleness

These considerations will guide our selection of a pair candidate.

### 4. Model the Normal Relationship Between the Selected Futures Pair

After selecting a pair, we will estimate how the two contracts normally move together. A simple baseline model may use regression or cointegration methods to estimate a hedge ratio and spread. The hedge ratio tells us how much of one contract should be held against the other to create a balanced relative-value trade. The spread measures how far the pair is from its normal relationship. We may extend the baseline model to a Bayesian or time-varying model that allows the relationship between the contracts to change over time, the latter will be useful in the case that there is varying term structures of the contracts.

### 5. Signal Generation and Strategy Logic

Once the spread is constructed, we will convert it into trading signals. A simple strategy may use a z-score of the spread:

* Enter a trade when the spread moves unusually far from its historical mean.
* Go long the relatively cheap contract and short the relatively expensive contract.
* Exit when the spread returns closer to normal.

The goal is to create a market-neutral strategy that focuses on relative mispricing rather than the overall direction of interest rates. We may explore other signal constructions in order to achieve this goal, a z-score based signal is often used because of it's statistical informed foundation, however alternative signals may capture better.

### 6. Performance Evaluation

The final stage will evaluate whether the strategy performs well after accounting for risk and realistic trading assumptions. Rigorously backtest the strategy against not only historical data, and consider some use of synthetic or resampling of data. With the goal of being market neutral, analyze performance across regimes.

## Issue Roadmap

The open GitHub issues define the project roadmap:

1. **Set up the project layout**
   Create the shared repository structure, dependency management, `.gitignore`, `.env.example`, and initial project skeleton.

2. **Data ingestion from Databento**
   Pull CME futures data from Databento, store it locally, and prepare clean data for analysis.

3. **Exploratory Data Analysis and Pair Selection**
   Compare candidate futures contracts, plot data, compute correlations and spreads, and select the pair most suitable for modeling.

4. **Model the Normal Relationship Between the Selected Futures Pair**
   Estimate the hedge ratio and spread using regression, cointegration, or a more advanced model if appropriate.

5. **Signal Generation and Strategy Logic**
   Convert spread deviations into entry and exit signals for a market-neutral trading strategy.

6. **Performance Evaluation**
   Backtest the strategy and evaluate returns and risk.

## Repository Structure

The repository structure is:

```text
Futures-UChicago-Project/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/            # ingest.py (Databento pull), panel.py (shared paths/loader)
│   ├── analysis/        # exploratory_analysis.py (EDA, pair selection)
│   ├── models/          # spread.py, seasonality.py
│   └── strategy/        # signals.py, backtest.py, walkforward_beta.py,
│                        # run_zq_sr3.py, portfolio.py, risk_overlay.py, regime.py
│
├── tests/               # fully synthetic pytest suite
│
├── outputs/
│   ├── figures/         # 26 numbered PNGs, committed
│   └── tables/          # 23 CSVs, committed
│
├── references/
│
└── docs/                # per-stage writeups: pair_selection, spread_model,
                         # signals, backtest
```

The exact structure may change as the project develops, but the goal is to keep the analysis modular rather than placing everything in one notebook.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/amazingmazy/Futures-UChicago-Project.git
cd Futures-UChicago-Project
```

### 2. Install dependencies

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management. If you do not have it installed, follow the instructions in the `uv` documentation.

The required Python version is pinned in `.python-version` (3.14), and `uv` will automatically provision a matching interpreter. Create the virtual environment and install all dependencies from `pyproject.toml` / `uv.lock` with:

```bash
uv sync
```

### 3. Set up environment variables

Create a local `.env` file:

```bash
cp .env.example .env
```

Then add your Databento API key:

```text
DATABENTO_API_KEY=your_api_key_here
```

Do not commit `.env` to GitHub.

### 4. Pull the data

```bash
uv run python -m src.data.ingest
```

This pulls continuous daily settlement prices for all nine futures roots from Databento into `data/raw/` and writes the processed price panel to `data/processed/continuous_settlement_prices.parquet`. The `data/` folder is gitignored, so this step must be run once locally before any analysis stage. The pull is resumable: roots whose raw files already exist are skipped.

### 5. Run the analysis

Run the stages in this order. Each stage reads the saved outputs of earlier stages, not the Databento API.

```bash
uv run python -m src.analysis.exploratory_analysis   # EDA and pair selection tables/figures
uv run python -m src.models.spread                   # CL/BZ hedge ratio and spread artifacts
uv run python -m src.strategy.signals                # z-score entry/exit signals
uv run python -m src.strategy.backtest               # PnL, costs, threshold sweep, walk-forward
uv run python -m src.strategy.walkforward_beta       # per-fold hedge-ratio refit
uv run python -m src.strategy.run_zq_sr3             # ZQ/SR3 second-pair comparison
uv run python -m src.strategy.portfolio              # combines the two pairs (needs the two lines above)
uv run python -m src.strategy.risk_overlay           # vol-target and drawdown-gate overlays
uv run python -m src.strategy.regime                 # vol-regime position sizing
uv run python -m src.models.seasonality              # walk-forward monthly seasonality
```

The full downstream run takes under a minute on the daily panel.

### 6. Run the tests

```bash
uv run pytest
```

The test suite is fully synthetic: it needs no Databento key and no downloaded data, so it can be run on a fresh clone before anything else.

### 7. Review outputs

Final figures and tables should be saved in:

```text
outputs/figures/
outputs/tables/
```

The final project should include a short written summary explaining the data, methods, results, and limitations.

## Expected Final Deliverable

The final deliverable will be a runnable GitHub repository that allows another user to reproduce the analysis. The repository should include:

* A clear README
* Modular Python code
* Notebooks for exploration and presentation
* Databento data-ingestion instructions
* Backtest results
* Performance metrics
* A concise explanation of the strategy and findings
