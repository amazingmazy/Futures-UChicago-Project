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

The intended/draft repository structure is:

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
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pair_selection.ipynb
│   └── 03_strategy_backtest.ipynb
│
├── src/
│   ├── data/
│   │   └── pull_databento.py
│   ├── analysis/
│   │   └── exploratory_analysis.py
│   ├── models/
│   │   └── spread_model.py
│   ├── strategy/
│   │   └── backtest.py
│   └── utils/
│       └── config.py
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── references/
│
└── docs/
```

The exact structure may change as the project develops, but the goal is to keep the analysis modular rather than placing everything in one notebook.

## How to Run the Project

These instructions are aspirational for the initial project setup and will be updated as the repository develops.

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

### 4. Pull or prepare the data

The final command may change, but the intended workflow is:

```bash
uv run python -m src.data.pull_databento
```

This should download or update the relevant futures data and store it in the local `data/` folder.

### 5. Run the analysis

The intended workflow is:

```bash
uv run python -m src.analysis.exploratory_analysis
uv run python -m src.models.spread_model
uv run python -m src.strategy.backtest
```

The notebooks in `notebooks/` may also be used to reproduce exploratory plots and intermediate findings.

### 6. Review outputs

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
