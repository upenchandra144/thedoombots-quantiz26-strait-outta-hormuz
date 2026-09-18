from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .metrics import counterfactual_metrics, portfolio_metrics


@dataclass(frozen=True)
class ScenarioInputs:
    freight_adjustment_pct: float = 0.0
    fuel_adjustment_pct: float = 0.0
    insurance_adjustment_pct: float = 0.0
    penalty_adjustment_pct: float = 0.0
    customer_pass_through_pct: float = 0.0
    held_release_days_earlier: int = 0
    pipeline_to_cape: bool = False
    pipeline_to_benchmark: bool = False


def _safe_rate(series: pd.Series, weight: pd.Series) -> float:
    total_weight = float(weight.sum())
    if total_weight <= 0:
        return 0.0
    return float(series.sum()) / total_weight


def _route_product_rates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    weight = work["Cargo_Weight_Tons"].replace(0, np.nan)
    for col in ["Freight_Cost_USD", "Fuel_Cost_USD", "Insurance_Cost_USD", "Penalty_Cost_USD"]:
        work[f"{col}_per_ton"] = work[col] / weight
    return (
        work.groupby(["Product_Category", "Route_Canonical"], dropna=False)
        .agg(
            Freight_Cost_USD_per_ton=("Freight_Cost_USD_per_ton", "median"),
            Fuel_Cost_USD_per_ton=("Fuel_Cost_USD_per_ton", "median"),
            Insurance_Cost_USD_per_ton=("Insurance_Cost_USD_per_ton", "median"),
            Penalty_Cost_USD_per_ton=("Penalty_Cost_USD_per_ton", "median"),
            DIFOT=("DIFOT_Flag", "mean"),
        )
        .reset_index()
    )


def _apply_route_intervention(work: pd.DataFrame, base: pd.DataFrame, scenario: ScenarioInputs) -> pd.DataFrame:
    if not (scenario.pipeline_to_cape or scenario.pipeline_to_benchmark):
        work["Scenario_Route"] = work["Route_Canonical"]
        return work

    work["Scenario_Route"] = work["Route_Canonical"]
    mask = work["Route_Canonical"].eq("Pipeline")
    if not mask.any():
        return work

    if scenario.pipeline_to_cape:
        work.loc[mask, "Scenario_Route"] = "Cape"
        return work

    # Best observed service-compliant benchmark by product.
    bench = base.groupby(["Product_Category", "Route_Canonical"], dropna=False).agg(
        Cost_per_Ton_USD=("Cost_per_Ton_USD", "median"),
        DIFOT=("DIFOT_Flag", "mean"),
    ).reset_index()
    feasible = bench[bench["DIFOT"] >= 0.80]
    if feasible.empty:
        return work
    best_idx = feasible.groupby("Product_Category")["Cost_per_Ton_USD"].idxmin()
    best = feasible.loc[best_idx].set_index("Product_Category")["Route_Canonical"]
    work.loc[mask, "Scenario_Route"] = work.loc[mask, "Product_Category"].map(best).fillna("Pipeline")
    return work


def simulate_scenario(df: pd.DataFrame, inputs: ScenarioInputs) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Run a scenario on a copy of the source dataframe.

    Route intervention uses observed product×route median cost components. Other
    adjustments are explicit user assumptions. The source dataframe is never mutated.
    """
    base = df.copy(deep=True)

    # Exact control case: when no scenario lever is active, preserve the source
    # dataframe and source financial totals exactly. This avoids surfacing tiny
    # arithmetic differences that exist between the workbook's Total_Cost field
    # and the sum of its component cost fields.
    no_change = (
        inputs.freight_adjustment_pct == 0
        and inputs.fuel_adjustment_pct == 0
        and inputs.insurance_adjustment_pct == 0
        and inputs.penalty_adjustment_pct == 0
        and inputs.customer_pass_through_pct == 0
        and inputs.held_release_days_earlier == 0
        and not inputs.pipeline_to_cape
        and not inputs.pipeline_to_benchmark
    )
    if no_change:
        baseline = portfolio_metrics(base)
        baseline_cf, _ = counterfactual_metrics(base)
        baseline_avoidable = float(baseline_cf["Avoidable_Exposure_USD"].sum())
        baseline_held = float(base.loc[base["Is_Held"], "Contracted_Freight_Revenue_USD"].sum())
        summary = {
            "baseline_gross_margin": baseline["gross_margin"],
            "scenario_gross_margin": baseline["gross_margin"],
            "baseline_margin_pct": baseline["gross_margin_pct_of_contracted"],
            "scenario_margin_pct": baseline["gross_margin_pct_of_contracted"],
            "baseline_total_cost": baseline["total_cost"],
            "scenario_total_cost": baseline["total_cost"],
            "baseline_held_exposure": baseline_held,
            "scenario_held_exposure": baseline_held,
            "baseline_avoidable_exposure": baseline_avoidable,
            "scenario_avoidable_exposure": baseline_avoidable,
            "gross_margin_change": 0.0,
            "total_cost_change": 0.0,
            "margin_pct_change": 0.0,
            "held_exposure_change": 0.0,
            "avoidable_exposure_change": 0.0,
        }
        control = base.copy(deep=True)
        control["Scenario_Route"] = control["Route_Canonical"]
        return control, summary

    work = base.copy(deep=True)
    work = _apply_route_intervention(work, base, inputs)
    rates = _route_product_rates(base)

    # Replace route economics only for shipments affected by an explicit route intervention.
    changed = work["Scenario_Route"].ne(work["Route_Canonical"])
    if changed.any():
        merge_rates = rates.rename(columns={"Route_Canonical": "Scenario_Route"})
        work = work.merge(
            merge_rates,
            on=["Product_Category", "Scenario_Route"],
            how="left",
            suffixes=("", "_scenario"),
        )
        wt = work["Cargo_Weight_Tons"]
        work.loc[changed, "Freight_Cost_USD"] = work.loc[changed, "Freight_Cost_USD_per_ton"] * wt.loc[changed]
        work.loc[changed, "Fuel_Cost_USD"] = work.loc[changed, "Fuel_Cost_USD_per_ton"] * wt.loc[changed]
        work.loc[changed, "Insurance_Cost_USD"] = work.loc[changed, "Insurance_Cost_USD_per_ton"] * wt.loc[changed]
        work.loc[changed, "Penalty_Cost_USD"] = work.loc[changed, "Penalty_Cost_USD_per_ton"] * wt.loc[changed]
        route_difot = work.loc[changed, "DIFOT"]
        work.loc[changed, "Revenue_Recognized_USD"] = (
            work.loc[changed, "Contracted_Freight_Revenue_USD"] * route_difot
        )
        work.drop(columns=[
            "Freight_Cost_USD_per_ton", "Fuel_Cost_USD_per_ton",
            "Insurance_Cost_USD_per_ton", "Penalty_Cost_USD_per_ton", "DIFOT",
        ], inplace=True)

    # Cost adjustments.
    work["Freight_Cost_USD"] *= 1 + inputs.freight_adjustment_pct
    work["Fuel_Cost_USD"] *= 1 + inputs.fuel_adjustment_pct
    work["Insurance_Cost_USD"] *= 1 + inputs.insurance_adjustment_pct
    work["Penalty_Cost_USD"] *= 1 + inputs.penalty_adjustment_pct

    # Customer pass-through is modeled as additional contracted revenue and
    # corresponding recognised revenue for currently recognised shipments.
    pass_through_factor = 1 + inputs.customer_pass_through_pct
    work["Contracted_Freight_Revenue_USD"] *= pass_through_factor
    work["Revenue_Recognized_USD"] *= pass_through_factor

    # Optional held-cargo acceleration: a 30-day planning window is used as a
    # transparent scenario assumption, not a forecast.
    release_fraction = min(max(inputs.held_release_days_earlier, 0) / 30.0, 1.0)
    held_mask = work["Is_Held"]
    if release_fraction > 0 and held_mask.any():
        original_unrecognized = (work.loc[held_mask, "Contracted_Freight_Revenue_USD"] - work.loc[held_mask, "Revenue_Recognized_USD"]).clip(lower=0)
        work.loc[held_mask, "Revenue_Recognized_USD"] += original_unrecognized * release_fraction
        work.loc[held_mask, "Penalty_Cost_USD"] *= 1 - release_fraction
        work.loc[held_mask, "Insurance_Cost_USD"] *= 1 - release_fraction

    work["Total_Cost_to_Serve_USD"] = (
        work["Freight_Cost_USD"]
        + work["Fuel_Cost_USD"]
        + work["Insurance_Cost_USD"]
        + work["Penalty_Cost_USD"]
    )
    work["Gross_Margin_USD"] = work["Revenue_Recognized_USD"] - work["Total_Cost_to_Serve_USD"]
    work["Gross_Margin_Pct"] = np.divide(
        work["Gross_Margin_USD"],
        work["Contracted_Freight_Revenue_USD"].replace(0, np.nan),
    )
    work["Cost_per_Ton_USD"] = np.divide(
        work["Total_Cost_to_Serve_USD"],
        work["Cargo_Weight_Tons"].replace(0, np.nan),
    )
    work["Revenue_per_Ton_USD"] = np.divide(
        work["Contracted_Freight_Revenue_USD"],
        work["Cargo_Weight_Tons"].replace(0, np.nan),
    )

    baseline = portfolio_metrics(base)
    scenario = portfolio_metrics(work)
    _, cf_route = counterfactual_metrics(work)

    baseline_held = float(base.loc[base["Is_Held"], "Contracted_Freight_Revenue_USD"].sum())
    scenario_held = baseline_held * (1 - release_fraction)

    baseline_cf, _ = counterfactual_metrics(base)
    scenario_cf, _ = counterfactual_metrics(work)
    baseline_avoidable = float(baseline_cf["Avoidable_Exposure_USD"].sum())
    scenario_avoidable = float(scenario_cf["Avoidable_Exposure_USD"].sum())

    summary = {
        "baseline_gross_margin": baseline["gross_margin"],
        "scenario_gross_margin": scenario["gross_margin"],
        "baseline_margin_pct": baseline["gross_margin_pct_of_contracted"],
        "scenario_margin_pct": scenario["gross_margin_pct_of_contracted"],
        "baseline_total_cost": baseline["total_cost"],
        "scenario_total_cost": scenario["total_cost"],
        "baseline_held_exposure": baseline_held,
        "scenario_held_exposure": scenario_held,
        "baseline_avoidable_exposure": baseline_avoidable,
        "scenario_avoidable_exposure": scenario_avoidable,
        "gross_margin_change": scenario["gross_margin"] - baseline["gross_margin"],
        "total_cost_change": scenario["total_cost"] - baseline["total_cost"],
        "margin_pct_change": scenario["gross_margin_pct_of_contracted"] - baseline["gross_margin_pct_of_contracted"],
        "held_exposure_change": scenario_held - baseline_held,
        "avoidable_exposure_change": scenario_avoidable - baseline_avoidable,
    }
    return work, summary
