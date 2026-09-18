from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_SERVICE_THRESHOLD = 0.80


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator in (0, None) or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def _loss_from_margin(margin: pd.Series) -> pd.Series:
    """Convert negative gross margin into positive loss exposure."""
    return margin.clip(upper=0).abs()


def portfolio_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Core portfolio KPIs used across the War Room."""
    contracted_revenue = float(df["Contracted_Freight_Revenue_USD"].sum())
    recognized_revenue = float(df["Revenue_Recognized_USD"].sum())
    total_cost = float(df["Total_Cost_to_Serve_USD"].sum())
    gross_margin = float(df["Gross_Margin_USD"].sum())
    cargo_value = float(df["Cargo_Value_USD"].sum())
    immobilised_cargo_value = float(df.loc[df["Is_Held"], "Cargo_Value_USD"].sum()) if "Is_Held" in df.columns else 0.0
    gross_margin_pct = _safe_div(gross_margin, contracted_revenue)

    return {
        "shipment_count": int(len(df)),
        "contracted_revenue": contracted_revenue,
        "recognized_revenue": recognized_revenue,
        "unrecognized_revenue": contracted_revenue - recognized_revenue,
        "total_cost": total_cost,
        "gross_margin": gross_margin,
        "gross_margin_pct_of_contracted": gross_margin_pct,
        "total_cargo_value": cargo_value,
        "immobilised_cargo_value": immobilised_cargo_value,
        "difot_pct": float(df["DIFOT_Flag"].mean()) if len(df) else 0.0,
        "held_shipments": int(df["Is_Held"].sum()),
    }


def customer_metrics(df: pd.DataFrame, customer: str | None = None) -> pd.DataFrame | Dict[str, float]:
    """Customer-level exposure table or one customer's summary."""
    grouped = (
        df.groupby("Customer_Name", dropna=False)
        .agg(
            Shipments=("Shipment_ID", "nunique"),
            Contracted_Revenue_USD=("Contracted_Freight_Revenue_USD", "sum"),
            Recognized_Revenue_USD=("Revenue_Recognized_USD", "sum"),
            Gross_Margin_USD=("Gross_Margin_USD", "sum"),
            DIFOT=("DIFOT_Flag", "mean"),
            Held_Shipments=("Is_Held", "sum"),
            Cargo_Value_USD=("Cargo_Value_USD", "sum"),
        )
        .reset_index()
    )

    portfolio_revenue = float(df["Contracted_Freight_Revenue_USD"].sum())
    # For customer loss share, use the portfolio net margin loss basis used in the case:
    # absolute value of total portfolio gross margin, not the sum of only negative rows.
    portfolio_loss = abs(float(df["Gross_Margin_USD"].sum()))

    grouped["Revenue_Share"] = grouped["Contracted_Revenue_USD"].map(
        lambda x: _safe_div(x, portfolio_revenue)
    )
    grouped["Margin_Loss_USD"] = _loss_from_margin(grouped["Gross_Margin_USD"])
    grouped["Margin_Loss_Share"] = grouped["Margin_Loss_USD"].map(
        lambda x: _safe_div(x, portfolio_loss)
    )
    grouped["DIFOT_Pct"] = grouped["DIFOT"] * 100.0

    grouped = grouped.sort_values("Contracted_Revenue_USD", ascending=False).reset_index(drop=True)

    if customer is None:
        return grouped

    match = grouped[grouped["Customer_Name"] == customer]
    if match.empty:
        raise KeyError(f"Customer not found: {customer}")

    row = match.iloc[0]
    return {
        "customer": str(row["Customer_Name"]),
        "shipments": int(row["Shipments"]),
        "contracted_revenue": float(row["Contracted_Revenue_USD"]),
        "recognized_revenue": float(row["Recognized_Revenue_USD"]),
        "revenue_share": float(row["Revenue_Share"]),
        "gross_margin": float(row["Gross_Margin_USD"]),
        "margin_loss": float(row["Margin_Loss_USD"]),
        "margin_loss_share": float(row["Margin_Loss_Share"]),
        "difot_pct": float(row["DIFOT_Pct"]),
        "held_shipments": int(row["Held_Shipments"]),
        "cargo_value": float(row["Cargo_Value_USD"]),
    }


def route_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Route-level operational and financial exposure."""
    grouped = (
        df.groupby("Route_Canonical", dropna=False)
        .agg(
            Shipments=("Shipment_ID", "nunique"),
            Contracted_Revenue_USD=("Contracted_Freight_Revenue_USD", "sum"),
            Recognized_Revenue_USD=("Revenue_Recognized_USD", "sum"),
            Gross_Margin_USD=("Gross_Margin_USD", "sum"),
            DIFOT=("DIFOT_Flag", "mean"),
            Cargo_Value_USD=("Cargo_Value_USD", "sum"),
            Cargo_Weight_Tons=("Cargo_Weight_Tons", "sum"),
        )
        .reset_index()
    )
    grouped["Margin_Loss_USD"] = _loss_from_margin(grouped["Gross_Margin_USD"])
    grouped["DIFOT_Pct"] = grouped["DIFOT"] * 100.0
    grouped["Revenue_Share"] = grouped["Contracted_Revenue_USD"] / max(
        float(df["Contracted_Freight_Revenue_USD"].sum()), 1.0
    )
    grouped["Cost_per_Ton_USD"] = grouped["Route_Canonical"].map(
        df.groupby("Route_Canonical")["Cost_per_Ton_USD"].median()
    )
    return grouped.sort_values("Margin_Loss_USD", ascending=False).reset_index(drop=True)


def product_route_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Observed economics for Product × Route combinations."""
    grouped = (
        df.groupby(["Product_Category", "Route_Canonical"], dropna=False)
        .agg(
            Shipments=("Shipment_ID", "nunique"),
            Median_Cost_per_Ton_USD=("Cost_per_Ton_USD", "median"),
            Mean_Cost_per_Ton_USD=("Cost_per_Ton_USD", "mean"),
            Median_Margin_per_Ton_USD=("Margin_per_Ton_USD", "median"),
            Mean_Margin_per_Ton_USD=("Margin_per_Ton_USD", "mean"),
            DIFOT=("DIFOT_Flag", "mean"),
            Contracted_Revenue_USD=("Contracted_Freight_Revenue_USD", "sum"),
            Gross_Margin_USD=("Gross_Margin_USD", "sum"),
        )
        .reset_index()
    )
    grouped["DIFOT_Pct"] = grouped["DIFOT"] * 100.0
    grouped["Margin_Loss_USD"] = _loss_from_margin(grouped["Gross_Margin_USD"])
    grouped["Service_Compliant"] = grouped["DIFOT"] >= DEFAULT_SERVICE_THRESHOLD
    return grouped.sort_values(
        ["Product_Category", "Median_Cost_per_Ton_USD"]
    ).reset_index(drop=True)


def customer_route_metrics(df: pd.DataFrame, customer: str | None = None) -> pd.DataFrame:
    """Customer × route exposure table, optionally filtered to one customer."""
    work = df if customer is None else df[df["Customer_Name"].eq(customer)]
    grouped = (
        work.groupby(["Customer_Name", "Route_Canonical"], dropna=False)
        .agg(
            Shipments=("Shipment_ID", "nunique"),
            Contracted_Revenue_USD=("Contracted_Freight_Revenue_USD", "sum"),
            Recognized_Revenue_USD=("Revenue_Recognized_USD", "sum"),
            Gross_Margin_USD=("Gross_Margin_USD", "sum"),
            DIFOT=("DIFOT_Flag", "mean"),
            Held_Shipments=("Is_Held", "sum"),
            Cargo_Value_USD=("Cargo_Value_USD", "sum"),
        )
        .reset_index()
    )
    grouped["Margin_Loss_USD"] = _loss_from_margin(grouped["Gross_Margin_USD"])
    grouped["DIFOT_Pct"] = grouped["DIFOT"] * 100.0
    return grouped.sort_values("Margin_Loss_USD", ascending=False).reset_index(drop=True)


def customer_product_metrics(df: pd.DataFrame, customer: str) -> pd.DataFrame:
    """Customer × product exposure for the command panel."""
    work = df[df["Customer_Name"].eq(customer)]
    grouped = (
        work.groupby("Product_Category", dropna=False)
        .agg(
            Shipments=("Shipment_ID", "nunique"),
            Contracted_Revenue_USD=("Contracted_Freight_Revenue_USD", "sum"),
            Gross_Margin_USD=("Gross_Margin_USD", "sum"),
            DIFOT=("DIFOT_Flag", "mean"),
            Held_Shipments=("Is_Held", "sum"),
        )
        .reset_index()
    )
    grouped["DIFOT_Pct"] = grouped["DIFOT"] * 100.0
    grouped["Margin_Loss_USD"] = _loss_from_margin(grouped["Gross_Margin_USD"])

    # Share of this customer's shipment count concentrated on its dominant route.
    route_concentration = (
        work.groupby(["Product_Category", "Route_Canonical"])["Shipment_ID"]
        .nunique()
        .reset_index(name="Shipments")
    )
    if not route_concentration.empty:
        dominant = (
            route_concentration.groupby("Product_Category")["Shipments"].max()
            / route_concentration.groupby("Product_Category")["Shipments"].sum()
        ).rename("Route_Concentration")
        grouped = grouped.merge(dominant, on="Product_Category", how="left")
    else:
        grouped["Route_Concentration"] = 0.0

    return grouped.sort_values("Margin_Loss_USD", ascending=False).reset_index(drop=True)


def held_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Held/trapped cargo exposure metrics."""
    held = df[df["Is_Held"]].copy()
    return {
        "held_shipments": int(len(held)),
        "trapped_contracted_revenue": float(held["Contracted_Freight_Revenue_USD"].sum()),
        "immobilised_cargo_value": float(held["Cargo_Value_USD"].sum()),
        "held_penalty_cost": float(held["Penalty_Cost_USD"].sum()),
        "held_insurance_cost": float(held["Insurance_Cost_USD"].sum()),
        "held_total_cost": float(held["Total_Cost_to_Serve_USD"].sum()),
        "held_gross_margin": float(held["Gross_Margin_USD"].sum()),
    }


def concentration_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Revenue concentration and HHI across customers."""
    customer_rev = df.groupby("Customer_Name")["Contracted_Freight_Revenue_USD"].sum().sort_values(ascending=False)
    total = float(customer_rev.sum())
    shares = customer_rev / total if total else customer_rev * 0

    def top_share(n: int) -> float:
        return float(shares.head(n).sum())

    return {
        "top_1_revenue_share": top_share(1),
        "top_3_revenue_share": top_share(3),
        "top_5_revenue_share": top_share(5),
        # HHI reported in the standard 0–10,000 scale.
        "customer_revenue_hhi": float(((shares * 100.0) ** 2).sum()),
    }


def loss_concentration(df: pd.DataFrame, threshold: float = 0.80) -> Dict[str, float]:
    """Find how many shipments account for a chosen share of portfolio loss."""
    work = df[["Shipment_ID", "Gross_Margin_USD"]].copy()
    work["Loss_USD"] = _loss_from_margin(work["Gross_Margin_USD"])
    work = work.sort_values("Loss_USD", ascending=False).reset_index(drop=True)
    total_loss = float(work["Loss_USD"].sum())
    if total_loss <= 0:
        return {"total_loss": 0.0, "shipments_to_threshold": 0, "threshold_share": 0.0}

    work["Cumulative_Loss_USD"] = work["Loss_USD"].cumsum()
    work["Cumulative_Share"] = work["Cumulative_Loss_USD"] / total_loss
    reached = work[work["Cumulative_Share"] >= threshold]

    if reached.empty:
        n = len(work)
        share = 1.0
    else:
        first = reached.iloc[0]
        n = int(reached.index[0] + 1)
        share = float(first["Cumulative_Share"])

    return {
        "total_loss": total_loss,
        "shipments_to_threshold": n,
        "threshold_share": share,
        "shipment_share_of_book": _safe_div(n, len(work)),
    }


def _product_route_benchmarks(
    df: pd.DataFrame,
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD,
) -> pd.DataFrame:
    """Create observed product × route benchmarks used by the counterfactual."""
    bench = (
        df.groupby(["Product_Category", "Route_Canonical"], dropna=False)
        .agg(
            Observed_Cost_per_Ton_USD=("Cost_per_Ton_USD", "median"),
            Observed_Margin_per_Ton_USD=("Margin_per_Ton_USD", "median"),
            Observed_DIFOT=("DIFOT_Flag", "mean"),
            Available_Shipments=("Shipment_ID", "nunique"),
        )
        .reset_index()
    )
    bench["Service_Compliant"] = bench["Observed_DIFOT"] >= service_threshold
    return bench


def counterfactual_metrics(
    df: pd.DataFrame,
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the 80% DIFOT benchmark shipment-by-shipment.

    Returns
    -------
    shipment_cf:
        One row per shipment with benchmark route, modeled cost and exposure.
    route_summary:
        Aggregated avoidable exposure by current route.
    """
    work = df.copy()
    bench = _product_route_benchmarks(work, service_threshold)
    feasible = bench[bench["Service_Compliant"]].copy()

    if feasible.empty:
        work["Best_Service_Compliant_Route"] = pd.NA
        work["Best_Service_Compliant_Cost_per_Ton_USD"] = np.nan
    else:
        best_idx = feasible.groupby("Product_Category")["Observed_Cost_per_Ton_USD"].idxmin()
        best = feasible.loc[best_idx, [
            "Product_Category",
            "Route_Canonical",
            "Observed_Cost_per_Ton_USD",
            "Observed_DIFOT",
        ]].rename(columns={
            "Route_Canonical": "Best_Service_Compliant_Route",
            "Observed_Cost_per_Ton_USD": "Best_Service_Compliant_Cost_per_Ton_USD",
            "Observed_DIFOT": "Best_Service_Compliant_DIFOT",
        })
        work = work.merge(best, on="Product_Category", how="left")

    if "Best_Service_Compliant_DIFOT" not in work.columns:
        work["Best_Service_Compliant_DIFOT"] = np.nan

    work["Current_Cost_USD"] = work["Cost_per_Ton_USD"] * work["Cargo_Weight_Tons"]
    work["Counterfactual_Cost_USD"] = (
        work["Best_Service_Compliant_Cost_per_Ton_USD"] * work["Cargo_Weight_Tons"]
    )
    work["Counterfactual_Margin_USD"] = (
        work["Contracted_Freight_Revenue_USD"] - work["Counterfactual_Cost_USD"]
    )
    work["Avoidable_Exposure_USD"] = (
        work["Counterfactual_Margin_USD"] - work["Gross_Margin_USD"]
    ).clip(lower=0)
    work["Residual_Exposure_USD"] = (-work["Counterfactual_Margin_USD"]).clip(lower=0)
    work["Cost_Gap_USD"] = work["Current_Cost_USD"] - work["Counterfactual_Cost_USD"]
    work["Service_Compliant_Benchmark_Available"] = work[
        "Best_Service_Compliant_Cost_per_Ton_USD"
    ].notna()

    route_summary = (
        work.groupby("Route_Canonical", dropna=False)
        .agg(
            Shipments=("Shipment_ID", "nunique"),
            Avoidable_Exposure_USD=("Avoidable_Exposure_USD", "sum"),
            Residual_Exposure_USD=("Residual_Exposure_USD", "sum"),
            Cost_Gap_USD=("Cost_Gap_USD", "sum"),
            Contracted_Revenue_USD=("Contracted_Freight_Revenue_USD", "sum"),
        )
        .reset_index()
        .sort_values("Avoidable_Exposure_USD", ascending=False)
        .reset_index(drop=True)
    )

    return work, route_summary
