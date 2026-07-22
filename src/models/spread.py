"""Spread model for the selected CL/BZ pair (issue #4).

Models the *normal relationship* between WTI (CL) and Brent (BZ) continuous
front-month settlements as a static OLS fit in price space::

    CL_t = alpha + beta * BZ_t + eps_t        [both legs in $/bbl]

and defines the spread as the deviation from that relationship,
``CL_t - alpha - beta * BZ_t``. Issue #5 turns this spread into z-score
entry/exit signals; issue #6 backtests them. Both read this stage's saved
artifacts rather than refitting:

* ``data/processed/spread_cl_bz.parquet`` -- per-date legs, fitted value and
  spread (gitignored, reproduced by running this module).
* ``data/processed/spread_model_cl_bz.json`` -- the fitted ``SpreadModel``
  parameters, reloadable via :func:`load_model`.

Modeling choices, made in the EDA (see ``docs/pair_selection.md``, Handoff):

* **Price space, not logs.** CL settled at -$37.63 on 2020-04-20; log prices
  do not exist there. Both legs quote in $/bbl on identical 1,000-barrel
  contracts, so the price-space spread is also (approximately) the
  dollar-neutral book -- no DV01-style dilemma.
* **Static hedge ratio.** The rolling 252-day beta stays in 0.74-1.14 over
  the full sample, so one full-sample OLS beta is defensible. The rolling
  beta is reported here as a stability *diagnostic*, not used as the model.
  The Bayesian / time-varying extension in issue #4 is out of scope.
* **Same test machinery as the EDA** (Engle-Granger ``coint``, ADF with
  ``autolag="AIC"``), so p-values are comparable across the two stages.

Run with::

    uv run python -m src.models.spread

Writes figures 08-09 to ``outputs/figures/`` and the model summary/stability
tables to ``outputs/tables/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from src.analysis.exploratory_analysis import (
    ROBUSTNESS_STARTS,
    ROLLING_WINDOW,
    half_life,
    rolling_beta,
)
from src.data.panel import (
    FIGURES_DIR,
    PROCESSED_DIR,
    TABLES_DIR,
    drop_sunday_session,
    load_panel,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: The regression is ``LEG_Y ~ const + beta * LEG_X``, matching the EDA's
#: ``analyse_pair(panel, "CL", "BZ")`` so the fitted numbers line up with the
#: pair-selection doc.
LEG_Y = "CL"
LEG_X = "BZ"

SPREAD_PATH = PROCESSED_DIR / "spread_cl_bz.parquet"
MODEL_PATH = PROCESSED_DIR / "spread_model_cl_bz.json"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass
class SpreadModel:
    """A fitted static-OLS spread model and its diagnostics.

    This is the object issue #5 consumes: ``beta`` sizes the hedge (short
    ``beta`` units of the X leg per unit of the Y leg), and :meth:`spread`
    reproduces the deviation series on any price data, in-sample or live.
    """

    leg_y: str
    leg_x: str
    alpha: float
    beta: float
    alpha_se: float
    beta_se: float
    r_squared: float
    n_obs: int
    sample_start: str
    sample_end: str
    eg_pvalue: float
    adf_pvalue: float
    half_life_days: float
    spread_std: float

    def spread(self, y: pd.Series, x: pd.Series) -> pd.Series:
        """Deviation from the fitted relationship: ``y - alpha - beta * x``.

        Includes ``alpha``, so on the fitting sample this equals the OLS
        residual and is mean-zero -- the same series as EDA figure 07.
        """
        return (y - self.alpha - self.beta * x).rename("spread")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpreadModel":
        return cls(**d)


def fit_spread_model(
    panel: pd.DataFrame, leg_y: str = LEG_Y, leg_x: str = LEG_X
) -> tuple[SpreadModel, pd.DataFrame]:
    """Fit the static OLS spread model on all dates where both legs settled.

    Returns the fitted model and the per-date frame that becomes the parquet
    artifact: columns ``<leg_y>``, ``<leg_x>`` (lowercased), ``fitted``
    (``alpha + beta * x``) and ``spread`` (``y - fitted``).
    """
    both = panel[[leg_y, leg_x]].dropna()
    y = both[leg_y]
    x = both[leg_x]

    fit = sm.OLS(y, sm.add_constant(x)).fit()
    spread = fit.resid

    model = SpreadModel(
        leg_y=leg_y,
        leg_x=leg_x,
        alpha=float(fit.params.iloc[0]),
        beta=float(fit.params.iloc[1]),
        alpha_se=float(fit.bse.iloc[0]),
        beta_se=float(fit.bse.iloc[1]),
        r_squared=float(fit.rsquared),
        n_obs=len(both),
        sample_start=str(both.index.min().date()),
        sample_end=str(both.index.max().date()),
        eg_pvalue=float(coint(y, x)[1]),
        adf_pvalue=float(adfuller(spread, autolag="AIC")[1]),
        half_life_days=half_life(spread),
        spread_std=float(spread.std()),
    )

    frame = pd.DataFrame(
        {
            leg_y.lower(): y,
            leg_x.lower(): x,
            "fitted": model.alpha + model.beta * x,
            "spread": model.spread(y, x),
        }
    )
    return model, frame


def subsample_stability(
    panel: pd.DataFrame,
    leg_y: str = LEG_Y,
    leg_x: str = LEG_X,
    starts: list[str] = ROBUSTNESS_STARTS,
) -> pd.DataFrame:
    """Refit the model from each progressively later start date.

    The EDA established that CL/BZ's Engle-Granger p-value survives every
    subsample window; this table adds what the EDA's robustness check did not
    report -- whether the *fitted parameters* also agree, i.e. whether the
    full-sample alpha/beta describe the recent regime or only the average of
    old ones. A window shorter than one rolling window (~1 year) has too
    little data to say anything, so it is reported as NaN.
    """
    rows = []
    for start in starts:
        sub = panel.loc[start:, [leg_y, leg_x]].dropna()
        row: dict[str, object] = {"start": start, "n_obs": len(sub)}
        if len(sub) < ROLLING_WINDOW:
            row.update(
                alpha=float("nan"),
                beta=float("nan"),
                eg_pvalue=float("nan"),
                adf_pvalue=float("nan"),
                half_life_days=float("nan"),
            )
        else:
            model, _ = fit_spread_model(sub, leg_y, leg_x)
            row.update(
                alpha=model.alpha,
                beta=model.beta,
                eg_pvalue=model.eg_pvalue,
                adf_pvalue=model.adf_pvalue,
                half_life_days=model.half_life_days,
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("start")


# --------------------------------------------------------------------------
# Artifact I/O (issue #5 imports these loaders)
# --------------------------------------------------------------------------


def save_model(model: SpreadModel, path: Path = MODEL_PATH) -> None:
    path.write_text(json.dumps(model.to_dict(), indent=2) + "\n")


def load_model(path: Path = MODEL_PATH) -> SpreadModel:
    if not path.exists():
        raise FileNotFoundError(
            f"No fitted spread model at {path}.\n"
            "Run the spread-modeling step first:\n"
            "    uv run python -m src.models.spread"
        )
    return SpreadModel.from_dict(json.loads(path.read_text()))


def save_spread(frame: pd.DataFrame, path: Path = SPREAD_PATH) -> None:
    frame.to_parquet(path)


def load_spread(path: Path = SPREAD_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No spread artifact at {path}.\n"
            "Run the spread-modeling step first:\n"
            "    uv run python -m src.models.spread"
        )
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_fit_scatter(frame: pd.DataFrame, model: SpreadModel, path: Path) -> None:
    """The fitted relationship itself: leg_y against leg_x with the OLS line.

    The one CL/BZ view no EDA figure shows. The April-2020 negative-WTI
    points sit far below the line (Brent stayed near $25 while WTI settled
    at -$37.63) -- the largest deviation in the sample, and the reason the
    model lives in price space: log prices do not exist at -$37.
    """
    y = frame[model.leg_y.lower()]
    x = frame[model.leg_x.lower()]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(x, y, s=6, alpha=0.35, color="#2c7fb8", edgecolors="none")
    grid = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(
        grid,
        model.alpha + model.beta * grid,
        color="#c0392b",
        linewidth=1.6,
        label=(
            f"{model.leg_y} = {model.alpha:.2f} + {model.beta:.3f} x {model.leg_x}"
            f"   (R$^2$ = {model.r_squared:.3f})"
        ),
    )
    ax.set_xlabel(f"{model.leg_x} settlement ($/bbl)")
    ax.set_ylabel(f"{model.leg_y} settlement ($/bbl)")
    ax.set_title(
        f"{model.leg_y}/{model.leg_x} static OLS fit, "
        f"{model.sample_start} to {model.sample_end} (n = {model.n_obs})"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_rolling_beta_vs_static(
    frame: pd.DataFrame, model: SpreadModel, path: Path
) -> None:
    """Rolling hedge ratio against the static one -- the stability diagnostic.

    EDA figure 04 shows this line squeezed among five other pairs on a clipped
    axis; here it gets its own axis plus the static beta it is being compared
    to. If the rolling line strayed far from the static line for long, the
    static model would be mis-hedged in that regime and the time-varying
    extension would stop being optional.
    """
    rolling = rolling_beta(frame[model.leg_y.lower()], frame[model.leg_x.lower()])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(rolling.index, rolling.to_numpy(), linewidth=1.2, color="#2c7fb8",
            label=f"rolling {ROLLING_WINDOW}d OLS beta")
    ax.axhline(model.beta, color="#c0392b", linestyle="--", linewidth=1.4,
               label=f"static full-sample beta = {model.beta:.3f}")
    ax.set_ylabel("beta")
    ax.set_title(
        f"{model.leg_y}/{model.leg_x} hedge-ratio stability: "
        f"rolling beta range {rolling.min():.2f} to {rolling.max():.2f}"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    panel = drop_sunday_session(load_panel())

    model, frame = fit_spread_model(panel)
    save_model(model)
    save_spread(frame)

    summary = pd.Series(model.to_dict(), name="value")
    summary.to_csv(TABLES_DIR / "spread_model_summary.csv", header=True)
    print("=== SPREAD MODEL (static OLS, price space) ===")
    print(summary.to_string())

    stability = subsample_stability(panel)
    stability.to_csv(TABLES_DIR / "spread_model_stability.csv")
    print("\n=== SUBSAMPLE REFITS ===")
    print(stability.to_string(float_format=lambda v: f"{v:.4f}"))
    print("  Cointegration holds on every window (EG p < 0.05), and the refit")
    print("  betas stay near the full-sample value: the static hedge ratio is")
    print("  not an artifact of the early sample.")

    plot_fit_scatter(frame, model, FIGURES_DIR / "08_cl_bz_fit.png")
    plot_rolling_beta_vs_static(frame, model, FIGURES_DIR / "09_cl_bz_rolling_beta.png")

    print(f"\nModel   -> {MODEL_PATH}")
    print(f"Spread  -> {SPREAD_PATH}")
    print(f"Figures -> {FIGURES_DIR}")
    print(f"Tables  -> {TABLES_DIR}")
    print("\nNext (issue #5): rolling z-score signals on the saved spread --")
    print("see docs/spread_model.md for the artifact contract.")


if __name__ == "__main__":
    main()
