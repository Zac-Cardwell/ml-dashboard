"""Demo system: a synthetic loan-default problem with controllable drift.

The same generator produces the training dataset (a normal dataset plugin) and the live traffic of the
simulator, so drift scenarios have a known ground truth:

  none          stationary traffic
  downturn      gradual economic downturn: incomes and credit scores fall, defaults rise (covariate + label shift)
  new_segment   sudden influx of young, thin-file applicants from a new marketing channel (covariate shift)
  broken_feature  an upstream pipeline bug starts sending income in thousands (data-quality incident)
  concept       the relationship changes: same applicants, but defaults rise (concept drift — only labels reveal it)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..plugins.base import DatasetPlugin, Param
from ..plugins.registry import register

SCENARIOS = ["none", "downturn", "new_segment", "broken_feature", "concept"]
PURPOSES = ["car", "home_improvement", "education", "business", "debt_consolidation", "medical"]
PURPOSE_RISK = {"car": -0.2, "home_improvement": -0.1, "education": 0.15, "business": 0.45, "debt_consolidation": 0.25, "medical": 0.3}
HOME = ["RENT", "MORTGAGE", "OWN"]
REGIONS = ["north", "south", "east", "west", "central"]


def drift_params(scenario: str, intensity: float) -> dict:
    t = float(np.clip(intensity, 0, 1))
    if scenario == "downturn":
        return {"income_mult": 1 - 0.35 * t, "credit_shift": -70 * t, "dti_shift": 0.12 * t, "concept": 0.5 * t}
    if scenario == "new_segment":
        return {"young_fraction": 0.65 * t}
    if scenario == "broken_feature":
        return {"income_scale": 1 / 1000 if t > 0 else 1.0}
    if scenario == "concept":
        return {"concept": 1.6 * t}
    return {}


def generate(n: int, rng: np.random.Generator, drift: dict | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    d = drift or {}
    young = rng.random(n) < d.get("young_fraction", 0.0)
    age = np.where(young, rng.normal(23, 2.5, n), rng.normal(42, 11, n)).clip(18, 80).round()
    income = np.exp(rng.normal(np.log(58000) + 0.012 * (age - 40) - np.where(young, 0.45, 0), 0.42)) * d.get("income_mult", 1.0)
    credit = (rng.normal(695, 55, n) + 1.1 * (age - 40) - np.where(young, 45, 0) + d.get("credit_shift", 0)).clip(300, 850).round()
    emp = (age - 22 - rng.exponential(4, n)).clip(0, 45).round(1)
    purpose = rng.choice(PURPOSES, n, p=[0.22, 0.14, 0.12, 0.1, 0.32, 0.1])
    loan = np.exp(rng.normal(np.log(14000), 0.55, n)) * np.where(purpose == "home_improvement", 1.6, 1.0)
    dti = (rng.beta(2, 5, n) * 0.7 + 0.25 * loan / income + d.get("dti_shift", 0)).clip(0, 1.5)
    lines = rng.poisson(np.where(young, 2, 6))
    p_own = np.clip((age - 25) / 60, 0.02, 0.6)
    home = np.where(rng.random(n) < p_own, "OWN", np.where(rng.random(n) < 0.55, "MORTGAGE", "RENT"))
    home = np.where(young & (rng.random(n) < 0.8), "RENT", home)
    region = rng.choice(REGIONS, n)

    logit = (-1.45 - 0.011 * (credit - 690) + 2.6 * (dti - 0.32) - 0.55 * np.log(income / 58000) + 0.35 * np.log(loan / 14000)
             + np.vectorize(PURPOSE_RISK.get)(purpose) + np.where(home == "RENT", 0.3, 0) - 0.025 * emp
             - 0.04 * (lines - 5) + d.get("concept", 0.0))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "age": age, "annual_income": (income * d.get("income_scale", 1.0)).round(0), "credit_score": credit,
        "employment_years": emp, "loan_amount": loan.round(-1), "debt_to_income": dti.round(3),
        "num_credit_lines": lines, "home_ownership": home, "loan_purpose": purpose, "region": region,
    })
    miss = rng.random(n) < 0.03                       # realistic missingness
    df.loc[miss, "employment_years"] = np.nan
    return df, y


@register
class LoanDefault(DatasetPlugin):
    id = "demo.loan_default"
    name = "Loan default (demo)"
    description = "Synthetic credit-risk data with numeric + categorical features; can generate drifted versions"
    task = "classification"
    modality = "tabular"

    def params(self):
        from ..builtin.datasets import TABULAR_PARAMS
        return [Param("n_samples", "int", 20000, min=500, group="Generator"),
                Param("scenario", "choice", "none", "Generate data as it looks under a drift scenario", choices=SCENARIOS, group="Generator"),
                Param("intensity", "float", 0.0, "How far into the scenario (0 = none, 1 = full)", 0.0, 1.0, group="Generator"),
                Param("data_seed", "int", 0, "Seed of the generated population", group="Generator")] + TABULAR_PARAMS

    def load(self, params, seed):
        from ..builtin.datasets import tabular_bundle
        rng = np.random.default_rng(params["data_seed"])
        df, y = generate(params["n_samples"], rng, drift_params(params["scenario"], params["intensity"]))
        return tabular_bundle(df, y, params, seed, "classification", class_names=["repaid", "default"])
