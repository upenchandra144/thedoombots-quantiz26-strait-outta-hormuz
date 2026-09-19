from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# ============================================================
# GLOBAL DECISION / EVIDENCE PARAMETERS
# ============================================================

DEFAULT_SERVICE_THRESHOLD = 0.80

# Direct is retained as a historical/pre-blockade reference,
# but is never an actionable disruption-era reroute.
ACTIONABLE_EXCLUDED_ROUTES = {"Direct", "Held"}

# Held is never treated as an observed transport alternative.
HISTORICAL_EXCLUDED_ROUTES = {"Held"}

# We do not allow a route-product combination with fewer than
# five observations to become the PRIMARY actionable benchmark.
# Such combinations may still appear as historical references
# or conditional fallbacks.
MIN_ACTIONABLE_BENCHMARK_SHIPMENTS = 5

# Evidence labels describe sample support, not route quality.
EVIDENCE_STRONG_N = 10
EVIDENCE_MODERATE_N = 5

BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 42

ROUTE_MAP = {
    "Direct (Pre-Blockade)": "Direct",
    "Cape of Good Hope": "Cape",
    "Pipeline Bypass": "Pipeline",
    "Overland Truck": "Overland",
    "Air Bridge": "Air",
    "Held in Gulf": "Held",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def _safe_div(numerator: float, denominator: float) -> float:
    if denominator in (0, None) or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def _loss_from_margin(margin: pd.Series) -> pd.Series:
    """Convert negative gross margin into positive loss exposure."""
    return margin.clip(upper=0).abs()


def _threshold_mask(
    series: pd.Series,
    threshold: float,
    tolerance: float = 1e-10,
) -> pd.Series:
    """Inclusive threshold comparison with floating-point tolerance."""
    return series >= (float(threshold) - tolerance)


def _evidence_level(n: int) -> str:
    """Sample-size evidence label.

    This is an evidence-availability label, not a quality/ranking score.
    """
    n = int(n)

    if n >= EVIDENCE_STRONG_N:
        return "Strong"
    if n >= EVIDENCE_MODERATE_N:
        return "Moderate"
    return "Limited"


def _ensure_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the minimum analysis columns guaranteed.

    The normal application path already creates these fields in
    data_loader.py. This helper keeps metrics.py robust when called
    independently in analysis scripts or tests.

    The input dataframe is never modified in-place.
    """
    work = df.copy()

    if "Route_Canonical" not in work.columns:
        if "Route_Type" not in work.columns:
            raise KeyError(
                "Expected 'Route_Canonical' or 'Route_Type' in dataframe."
            )
        work["Route_Canonical"] = (
            work["Route_Type"]
            .map(ROUTE_MAP)
            .fillna(work["Route_Type"])
        )

    if "DIFOT_Flag" not in work.columns:
        if "DIFOT_Met" in work.columns:
            work["DIFOT_Flag"] = (
                work["DIFOT_Met"]
                .astype(str)
                .str.strip()
                .str.upper()
                .eq("Y")
            )
        else:
            raise KeyError(
                "Expected 'DIFOT_Flag' or 'DIFOT_Met' in dataframe."
            )

    if "Is_Held" not in work.columns:
        work["Is_Held"] = work["Route_Canonical"].eq("Held")

    if "Cost_per_Ton_USD" not in work.columns:
        if {
            "Total_Cost_to_Serve_USD",
            "Cargo_Weight_Tons",
        }.issubset(work.columns):
            work["Cost_per_Ton_USD"] = (
                work["Total_Cost_to_Serve_USD"]
                / work["Cargo_Weight_Tons"].replace(0, np.nan)
            )
        else:
            raise KeyError(
                "Expected 'Cost_per_Ton_USD' or the fields required to derive it."
            )

    if "Margin_per_Ton_USD" not in work.columns:
        if {
            "Gross_Margin_USD",
            "Cargo_Weight_Tons",
        }.issubset(work.columns):
            work["Margin_per_Ton_USD"] = (
                work["Gross_Margin_USD"]
                / work["Cargo_Weight_Tons"].replace(0, np.nan)
            )
        else:
            raise KeyError(
                "Expected 'Margin_per_Ton_USD' or the fields required to derive it."
            )

    return work


# ============================================================
# BINOMIAL UNCERTAINTY
# ============================================================

def _wilson_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    n = int(n)
    successes = int(successes)

    if n <= 0:
        return np.nan, np.nan

    # Standard normal critical values. 1.959963984540054 ~= 95%.
    z_lookup = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    z = z_lookup.get(
        round(float(confidence), 2),
        1.959963984540054,
    )

    phat = successes / n
    denominator = 1.0 + (z**2 / n)
    centre = (
        phat
        + (z**2 / (2.0 * n))
    ) / denominator
    half_width = (
        z
        * np.sqrt(
            (
                phat * (1.0 - phat) / n
            )
            + (
                z**2 / (4.0 * n**2)
            )
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


def _leave_one_out_difot(
    group: pd.DataFrame,
    service_threshold: float,
) -> Dict[str, float]:
    """Leave-one-out DIFOT sensitivity for a single Product × Route."""
    n = len(group)

    if n <= 1:
        return {
            "LOO_Min_DIFOT": np.nan,
            "LOO_Max_DIFOT": np.nan,
            "LOO_Mean_DIFOT": np.nan,
            "LOO_Below_Threshold_Count": 0,
            "LOO_At_Least_Threshold_Count": 0,
            "LOO_Sample_Count": 0,
            "Borderline_At_Threshold": False,
        }

    values = group["DIFOT_Flag"].astype(bool).to_numpy()

    loo_rates = []

    for i in range(n):
        reduced = np.delete(values, i)
        loo_rates.append(
            float(reduced.mean())
        )

    loo = np.asarray(loo_rates, dtype=float)

    below = int(
        np.sum(
            loo < (service_threshold - 1e-10)
        )
    )

    at_least = int(
        np.sum(
            loo >= (service_threshold - 1e-10)
        )
    )

    observed = float(values.mean())

    borderline = (
        observed >= (service_threshold - 1e-10)
        and below > 0
    )

    return {
        "LOO_Min_DIFOT": float(np.min(loo)),
        "LOO_Max_DIFOT": float(np.max(loo)),
        "LOO_Mean_DIFOT": float(np.mean(loo)),
        "LOO_Below_Threshold_Count": below,
        "LOO_At_Least_Threshold_Count": at_least,
        "LOO_Sample_Count": int(len(loo)),
        "Borderline_At_Threshold": bool(borderline),
    }


# ============================================================
# COST UNCERTAINTY / EFFECT SIZE
# ============================================================

def _bootstrap_median_difference(
    x: Sequence[float],
    y: Sequence[float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for median(x) - median(y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan

    observed = float(
        np.median(x) - np.median(y)
    )

    rng = np.random.default_rng(seed)

    x_boot = rng.choice(
        x,
        size=(iterations, len(x)),
        replace=True,
    )
    y_boot = rng.choice(
        y,
        size=(iterations, len(y)),
        replace=True,
    )

    boot_diffs = (
        np.median(x_boot, axis=1)
        - np.median(y_boot, axis=1)
    )

    lower, upper = np.percentile(
        boot_diffs,
        [2.5, 97.5],
    )

    return (
        observed,
        float(lower),
        float(upper),
    )


def _cliffs_delta(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    """Cliff's delta for x versus y.

    +1 means every x observation exceeds every y observation.
    -1 means every x observation is below every y observation.
    0 means no net dominance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    differences = x[:, None] - y[None, :]

    greater = np.sum(differences > 0)
    less = np.sum(differences < 0)

    return float(
        (greater - less)
        / (len(x) * len(y))
    )


# ============================================================
# PORTFOLIO
# ============================================================

def portfolio_metrics(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Core portfolio KPIs used across the War Room."""
    work = _ensure_analysis_columns(df)

    contracted_revenue = float(
        work["Contracted_Freight_Revenue_USD"].sum()
    )
    recognized_revenue = float(
        work["Revenue_Recognized_USD"].sum()
    )
    total_cost = float(
        work["Total_Cost_to_Serve_USD"].sum()
    )
    gross_margin = float(
        work["Gross_Margin_USD"].sum()
    )
    cargo_value = float(
        work["Cargo_Value_USD"].sum()
    )
    immobilised_cargo_value = float(
        work.loc[
            work["Is_Held"],
            "Cargo_Value_USD",
        ].sum()
    )

    metrics = {
        "shipment_count": int(len(work)),
        "contracted_revenue": contracted_revenue,
        "recognized_revenue": recognized_revenue,
        "unrecognized_revenue": (
            contracted_revenue
            - recognized_revenue
        ),
        "total_cost": total_cost,
        "gross_margin": gross_margin,
        "gross_margin_pct_of_contracted": _safe_div(
            gross_margin,
            contracted_revenue,
        ),
        "total_cargo_value": cargo_value,
        "immobilised_cargo_value": immobilised_cargo_value,
        "difot_pct": (
            float(work["DIFOT_Flag"].mean())
            if len(work)
            else 0.0
        ),
        "held_shipments": int(
            work["Is_Held"].sum()
        ),
        "service_threshold": DEFAULT_SERVICE_THRESHOLD,
    }

    # If counterfactual columns have already been attached,
    # expose the portfolio-level actionable/historical results.
    if "Historical_Avoidable_Exposure_USD" in work.columns:
        metrics["historical_avoidable_exposure"] = float(
            work[
                "Historical_Avoidable_Exposure_USD"
            ].sum(skipna=True)
        )
        metrics["historical_residual_exposure"] = float(
            work[
                "Historical_Residual_Exposure_USD"
            ].sum(skipna=True)
        )

    if "Avoidable_Exposure_USD" in work.columns:
        metrics["actionable_avoidable_exposure"] = float(
            work[
                "Avoidable_Exposure_USD"
            ].sum(skipna=True)
        )
        metrics["actionable_residual_exposure"] = float(
            work[
                "Residual_Exposure_USD"
            ].sum(skipna=True)
        )

    return metrics


# ============================================================
# CUSTOMER METRICS
# ============================================================

def customer_metrics(
    df: pd.DataFrame,
    customer: str | None = None,
) -> pd.DataFrame | Dict[str, float]:
    """Customer-level exposure table or one customer's summary."""
    work = _ensure_analysis_columns(df)

    grouped = (
        work.groupby(
            "Customer_Name",
            dropna=False,
        )
        .agg(
            Shipments=(
                "Shipment_ID",
                "nunique",
            ),
            Contracted_Revenue_USD=(
                "Contracted_Freight_Revenue_USD",
                "sum",
            ),
            Recognized_Revenue_USD=(
                "Revenue_Recognized_USD",
                "sum",
            ),
            Gross_Margin_USD=(
                "Gross_Margin_USD",
                "sum",
            ),
            DIFOT=(
                "DIFOT_Flag",
                "mean",
            ),
            Held_Shipments=(
                "Is_Held",
                "sum",
            ),
            Cargo_Value_USD=(
                "Cargo_Value_USD",
                "sum",
            ),
        )
        .reset_index()
    )

    portfolio_revenue = float(
        work[
            "Contracted_Freight_Revenue_USD"
        ].sum()
    )

    portfolio_loss = abs(
        float(
            work[
                "Gross_Margin_USD"
            ].sum()
        )
    )

    grouped["Revenue_Share"] = grouped[
        "Contracted_Revenue_USD"
    ].map(
        lambda x: _safe_div(
            x,
            portfolio_revenue,
        )
    )

    grouped["Margin_Loss_USD"] = _loss_from_margin(
        grouped["Gross_Margin_USD"]
    )

    grouped["Margin_Loss_Share"] = grouped[
        "Margin_Loss_USD"
    ].map(
        lambda x: _safe_div(
            x,
            portfolio_loss,
        )
    )

    grouped["DIFOT_Pct"] = (
        grouped["DIFOT"] * 100.0
    )

    grouped = grouped.sort_values(
        "Contracted_Revenue_USD",
        ascending=False,
    ).reset_index(drop=True)

    if customer is None:
        return grouped

    match = grouped[
        grouped["Customer_Name"]
        == customer
    ]

    if match.empty:
        raise KeyError(
            f"Customer not found: {customer}"
        )

    row = match.iloc[0]

    return {
        "customer": str(
            row["Customer_Name"]
        ),
        "shipments": int(
            row["Shipments"]
        ),
        "contracted_revenue": float(
            row["Contracted_Revenue_USD"]
        ),
        "recognized_revenue": float(
            row["Recognized_Revenue_USD"]
        ),
        "revenue_share": float(
            row["Revenue_Share"]
        ),
        "gross_margin": float(
            row["Gross_Margin_USD"]
        ),
        "margin_loss": float(
            row["Margin_Loss_USD"]
        ),
        "margin_loss_share": float(
            row["Margin_Loss_Share"]
        ),
        "difot_pct": float(
            row["DIFOT_Pct"]
        ),
        "held_shipments": int(
            row["Held_Shipments"]
        ),
        "cargo_value": float(
            row["Cargo_Value_USD"]
        ),
    }


# ============================================================
# ROUTE METRICS
# ============================================================

def route_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Route-level operational and financial exposure."""
    work = _ensure_analysis_columns(df)

    grouped = (
        work.groupby(
            "Route_Canonical",
            dropna=False,
        )
        .agg(
            Shipments=(
                "Shipment_ID",
                "nunique",
            ),
            Contracted_Revenue_USD=(
                "Contracted_Freight_Revenue_USD",
                "sum",
            ),
            Recognized_Revenue_USD=(
                "Revenue_Recognized_USD",
                "sum",
            ),
            Gross_Margin_USD=(
                "Gross_Margin_USD",
                "sum",
            ),
            DIFOT=(
                "DIFOT_Flag",
                "mean",
            ),
            Cargo_Value_USD=(
                "Cargo_Value_USD",
                "sum",
            ),
            Cargo_Weight_Tons=(
                "Cargo_Weight_Tons",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["Margin_Loss_USD"] = _loss_from_margin(
        grouped["Gross_Margin_USD"]
    )

    grouped["DIFOT_Pct"] = (
        grouped["DIFOT"] * 100.0
    )

    grouped["Revenue_Share"] = (
        grouped["Contracted_Revenue_USD"]
        / max(
            float(
                work[
                    "Contracted_Freight_Revenue_USD"
                ].sum()
            ),
            1.0,
        )
    )

    route_cost_medians = (
        work.groupby(
            "Route_Canonical"
        )[
            "Cost_per_Ton_USD"
        ]
        .median()
    )

    grouped["Cost_per_Ton_USD"] = grouped[
        "Route_Canonical"
    ].map(route_cost_medians)

    grouped["Is_Actionable_Route"] = ~grouped[
        "Route_Canonical"
    ].isin(ACTIONABLE_EXCLUDED_ROUTES)

    grouped["Is_Historical_Route"] = ~grouped[
        "Route_Canonical"
    ].isin(HISTORICAL_EXCLUDED_ROUTES)

    grouped["Service_Compliant"] = _threshold_mask(
        grouped["DIFOT"],
        DEFAULT_SERVICE_THRESHOLD,
    )

    return grouped.sort_values(
        "Margin_Loss_USD",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# PRODUCT × ROUTE
# ============================================================

def product_route_metrics(
    df: pd.DataFrame,
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD,
) -> pd.DataFrame:
    """Observed economics and evidence for Product × Route combinations."""
    work = _ensure_analysis_columns(df)

    grouped = (
        work.groupby(
            [
                "Product_Category",
                "Route_Canonical",
            ],
            dropna=False,
        )
        .agg(
            Shipments=(
                "Shipment_ID",
                "nunique",
            ),
            Median_Cost_per_Ton_USD=(
                "Cost_per_Ton_USD",
                "median",
            ),
            Mean_Cost_per_Ton_USD=(
                "Cost_per_Ton_USD",
                "mean",
            ),
            Median_Margin_per_Ton_USD=(
                "Margin_per_Ton_USD",
                "median",
            ),
            Mean_Margin_per_Ton_USD=(
                "Margin_per_Ton_USD",
                "mean",
            ),
            DIFOT=(
                "DIFOT_Flag",
                "mean",
            ),
            Contracted_Revenue_USD=(
                "Contracted_Freight_Revenue_USD",
                "sum",
            ),
            Gross_Margin_USD=(
                "Gross_Margin_USD",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["DIFOT_Pct"] = (
        grouped["DIFOT"] * 100.0
    )

    grouped["Margin_Loss_USD"] = _loss_from_margin(
        grouped["Gross_Margin_USD"]
    )

    grouped["Service_Compliant"] = _threshold_mask(
        grouped["DIFOT"],
        service_threshold,
    )

    grouped["Is_Actionable_Route"] = ~grouped[
        "Route_Canonical"
    ].isin(ACTIONABLE_EXCLUDED_ROUTES)

    grouped["Is_Historical_Route"] = ~grouped[
        "Route_Canonical"
    ].isin(HISTORICAL_EXCLUDED_ROUTES)

    # Wilson interval + leave-one-out evidence.
    evidence_rows = []

    for (product, route), group in work.groupby(
        [
            "Product_Category",
            "Route_Canonical",
        ],
        dropna=False,
    ):
        n = len(group)
        successes = int(
            group["DIFOT_Flag"].sum()
        )

        lower, upper = _wilson_interval(
            successes,
            n,
            confidence=0.95,
        )

        loo = _leave_one_out_difot(
            group,
            service_threshold,
        )

        evidence_rows.append(
            {
                "Product_Category": product,
                "Route_Canonical": route,
                "Wilson_DIFOT_Lower": lower,
                "Wilson_DIFOT_Upper": upper,
                "Wilson_DIFOT_Lower_Pct": lower * 100.0,
                "Wilson_DIFOT_Upper_Pct": upper * 100.0,
                "Evidence_Level": _evidence_level(n),
                **loo,
            }
        )

    evidence = pd.DataFrame(
        evidence_rows
    )

    grouped = grouped.merge(
        evidence,
        on=[
            "Product_Category",
            "Route_Canonical",
        ],
        how="left",
    )

    # The route is "borderline" when the observed estimate meets
    # the threshold but at least one leave-one-out observation
    # causes it to fall below the threshold.
    grouped["Borderline_Service_Evidence"] = (
        grouped["Borderline_At_Threshold"]
        .fillna(False)
        .astype(bool)
    )

    return grouped.sort_values(
        [
            "Product_Category",
            "Median_Cost_per_Ton_USD",
        ]
    ).reset_index(drop=True)


# ============================================================
# CUSTOMER × ROUTE
# ============================================================

def customer_route_metrics(
    df: pd.DataFrame,
    customer: str | None = None,
) -> pd.DataFrame:
    """Customer × route exposure table."""
    work = _ensure_analysis_columns(df)

    if customer is not None:
        work = work[
            work["Customer_Name"].eq(customer)
        ]

    grouped = (
        work.groupby(
            [
                "Customer_Name",
                "Route_Canonical",
            ],
            dropna=False,
        )
        .agg(
            Shipments=(
                "Shipment_ID",
                "nunique",
            ),
            Contracted_Revenue_USD=(
                "Contracted_Freight_Revenue_USD",
                "sum",
            ),
            Recognized_Revenue_USD=(
                "Revenue_Recognized_USD",
                "sum",
            ),
            Gross_Margin_USD=(
                "Gross_Margin_USD",
                "sum",
            ),
            DIFOT=(
                "DIFOT_Flag",
                "mean",
            ),
            Held_Shipments=(
                "Is_Held",
                "sum",
            ),
            Cargo_Value_USD=(
                "Cargo_Value_USD",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["Margin_Loss_USD"] = _loss_from_margin(
        grouped["Gross_Margin_USD"]
    )

    grouped["DIFOT_Pct"] = (
        grouped["DIFOT"] * 100.0
    )

    return grouped.sort_values(
        "Margin_Loss_USD",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# CUSTOMER × PRODUCT
# ============================================================

def customer_product_metrics(
    df: pd.DataFrame,
    customer: str,
) -> pd.DataFrame:
    """Customer × product exposure for the command panel."""
    work = _ensure_analysis_columns(df)
    work = work[
        work["Customer_Name"].eq(customer)
    ]

    grouped = (
        work.groupby(
            "Product_Category",
            dropna=False,
        )
        .agg(
            Shipments=(
                "Shipment_ID",
                "nunique",
            ),
            Contracted_Revenue_USD=(
                "Contracted_Freight_Revenue_USD",
                "sum",
            ),
            Gross_Margin_USD=(
                "Gross_Margin_USD",
                "sum",
            ),
            DIFOT=(
                "DIFOT_Flag",
                "mean",
            ),
            Held_Shipments=(
                "Is_Held",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["DIFOT_Pct"] = (
        grouped["DIFOT"] * 100.0
    )

    grouped["Margin_Loss_USD"] = _loss_from_margin(
        grouped["Gross_Margin_USD"]
    )

    route_concentration = (
        work.groupby(
            [
                "Product_Category",
                "Route_Canonical",
            ]
        )["Shipment_ID"]
        .nunique()
        .reset_index(
            name="Shipments"
        )
    )

    if not route_concentration.empty:
        dominant = (
            route_concentration
            .groupby(
                "Product_Category"
            )["Shipments"]
            .max()
            /
            route_concentration
            .groupby(
                "Product_Category"
            )["Shipments"]
            .sum()
        ).rename(
            "Route_Concentration"
        )

        grouped = grouped.merge(
            dominant,
            on="Product_Category",
            how="left",
        )
    else:
        grouped[
            "Route_Concentration"
        ] = 0.0

    return grouped.sort_values(
        "Margin_Loss_USD",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# HELD
# ============================================================

def held_metrics(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Held/trapped cargo exposure metrics."""
    work = _ensure_analysis_columns(df)
    held = work[
        work["Is_Held"]
    ].copy()

    return {
        "held_shipments": int(
            len(held)
        ),
        "trapped_contracted_revenue": float(
            held[
                "Contracted_Freight_Revenue_USD"
            ].sum()
        ),
        "immobilised_cargo_value": float(
            held[
                "Cargo_Value_USD"
            ].sum()
        ),
        "held_penalty_cost": float(
            held[
                "Penalty_Cost_USD"
            ].sum()
        ),
        "held_insurance_cost": float(
            held[
                "Insurance_Cost_USD"
            ].sum()
        ),
        "held_total_cost": float(
            held[
                "Total_Cost_to_Serve_USD"
            ].sum()
        ),
        "held_gross_margin": float(
            held[
                "Gross_Margin_USD"
            ].sum()
        ),
    }


# ============================================================
# CONCENTRATION / LOSS
# ============================================================

def concentration_metrics(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Revenue concentration and HHI across customers."""
    work = _ensure_analysis_columns(df)

    customer_rev = (
        work.groupby(
            "Customer_Name"
        )[
            "Contracted_Freight_Revenue_USD"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    total = float(
        customer_rev.sum()
    )

    shares = (
        customer_rev / total
        if total
        else customer_rev * 0
    )

    def top_share(n: int) -> float:
        return float(
            shares.head(n).sum()
        )

    return {
        "top_1_revenue_share":
            top_share(1),

        "top_3_revenue_share":
            top_share(3),

        "top_5_revenue_share":
            top_share(5),

        # Standard 0–10,000 HHI scale.
        "customer_revenue_hhi":
            float(
                (
                    (shares * 100.0) ** 2
                ).sum()
            ),
    }


def loss_concentration(
    df: pd.DataFrame,
    threshold: float = 0.80,
) -> Dict[str, float]:
    """Find how many shipments account for a chosen share of portfolio loss."""
    work = _ensure_analysis_columns(df)[
        [
            "Shipment_ID",
            "Gross_Margin_USD",
        ]
    ].copy()

    work["Loss_USD"] = _loss_from_margin(
        work["Gross_Margin_USD"]
    )

    work = work.sort_values(
        "Loss_USD",
        ascending=False,
    ).reset_index(drop=True)

    total_loss = float(
        work["Loss_USD"].sum()
    )

    if total_loss <= 0:
        return {
            "total_loss": 0.0,
            "shipments_to_threshold": 0,
            "threshold_share": 0.0,
            "shipment_share_of_book": 0.0,
        }

    work["Cumulative_Loss_USD"] = (
        work["Loss_USD"].cumsum()
    )

    work["Cumulative_Share"] = (
        work["Cumulative_Loss_USD"]
        / total_loss
    )

    reached = work[
        work["Cumulative_Share"]
        >= threshold
    ]

    if reached.empty:
        n = len(work)
        share = 1.0
    else:
        first = reached.iloc[0]
        n = int(
            reached.index[0] + 1
        )
        share = float(
            first["Cumulative_Share"]
        )

    return {
        "total_loss": total_loss,
        "shipments_to_threshold": n,
        "threshold_share": share,
        "shipment_share_of_book": _safe_div(
            n,
            len(work),
        ),
    }


# ============================================================
# PRODUCT × ROUTE BENCHMARK TABLE
# ============================================================

def _product_route_benchmarks(
    df: pd.DataFrame,
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD,
) -> pd.DataFrame:
    """Observed Product × Route benchmark statistics.

    This table is intentionally richer than the previous version:
    it preserves the raw observed economics, service performance,
    evidence strength, Wilson confidence interval, and leave-one-out
    threshold robustness.
    """
    work = _ensure_analysis_columns(df)

    rows = []

    for (
        product,
        route,
    ), group in work.groupby(
        [
            "Product_Category",
            "Route_Canonical",
        ],
        dropna=False,
    ):
        costs = (
            group[
                "Cost_per_Ton_USD"
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        margins = (
            group[
                "Margin_per_Ton_USD"
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        n = int(
            group["Shipment_ID"].nunique()
        )

        successes = int(
            group["DIFOT_Flag"].sum()
        )

        difot = (
            successes / n
            if n
            else np.nan
        )

        lower, upper = _wilson_interval(
            successes,
            n,
        )

        loo = _leave_one_out_difot(
            group,
            service_threshold,
        )

        rows.append(
            {
                "Product_Category":
                    product,

                "Route_Canonical":
                    route,

                "Available_Shipments":
                    n,

                "Observed_Cost_per_Ton_USD":
                    float(
                        np.median(costs)
                    )
                    if len(costs)
                    else np.nan,

                "Observed_Mean_Cost_per_Ton_USD":
                    float(
                        np.mean(costs)
                    )
                    if len(costs)
                    else np.nan,

                "Observed_Margin_per_Ton_USD":
                    float(
                        np.median(margins)
                    )
                    if len(margins)
                    else np.nan,

                "Observed_DIFOT":
                    difot,

                "Observed_DIFOT_Pct":
                    difot * 100.0
                    if pd.notna(difot)
                    else np.nan,

                "Service_Compliant":
                    (
                        bool(
                            difot
                            >= (
                                service_threshold
                                - 1e-10
                            )
                        )
                        if pd.notna(difot)
                        else False
                    ),

                "Historical_Eligible":
                    route
                    not in HISTORICAL_EXCLUDED_ROUTES,

                "Actionable_Eligible":
                    route
                    not in ACTIONABLE_EXCLUDED_ROUTES,

                "Actionable_Evidence_Eligible":
                    (
                        route
                        not in ACTIONABLE_EXCLUDED_ROUTES
                    )
                    and (
                        n
                        >= MIN_ACTIONABLE_BENCHMARK_SHIPMENTS
                    ),

                "Wilson_DIFOT_Lower":
                    lower,

                "Wilson_DIFOT_Upper":
                    upper,

                "Wilson_DIFOT_Lower_Pct":
                    lower * 100.0,

                "Wilson_DIFOT_Upper_Pct":
                    upper * 100.0,

                "Evidence_Level":
                    _evidence_level(n),

                **loo,
            }
        )

    bench = pd.DataFrame(rows)

    bench["Borderline_Service_Evidence"] = (
        bench[
            "Borderline_At_Threshold"
        ]
        .fillna(False)
        .astype(bool)
    )

    return bench


# ============================================================
# BENCHMARK SELECTION
# ============================================================

def _select_benchmark_routes(
    bench: pd.DataFrame,
    service_threshold: float,
    mode: str,
) -> pd.DataFrame:
    """Select the cheapest compliant route per product.

    mode='historical':
        Held excluded; Direct allowed.

    mode='actionable':
        Direct and Held excluded; at least five observations required.
    """
    if mode == "historical":
        candidates = bench[
            bench["Historical_Eligible"]
            &
            _threshold_mask(
                bench["Observed_DIFOT"],
                service_threshold,
            )
        ].copy()

    elif mode == "actionable":
        candidates = bench[
            bench["Actionable_Evidence_Eligible"]
            &
            _threshold_mask(
                bench["Observed_DIFOT"],
                service_threshold,
            )
        ].copy()

    else:
        raise ValueError(
            "mode must be 'historical' or 'actionable'."
        )

    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        [
            "Product_Category",
            "Observed_Cost_per_Ton_USD",
            "Observed_DIFOT",
            "Available_Shipments",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    )

    best = (
        candidates
        .groupby(
            "Product_Category",
            as_index=False,
        )
        .first()
    )

    return best.reset_index(
        drop=True
    )


def _select_conditional_fallbacks(
    bench: pd.DataFrame,
    service_threshold: float,
) -> pd.DataFrame:
    """Choose the least-service-gap observed non-Direct/non-Held fallback.

    If no compliant disruption-era route exists, the fallback is the
    route with the smallest observed service shortfall; ties are resolved
    by lower median cost and then larger sample size.

    This is explicitly a CONDITIONAL fallback, not a compliant benchmark.
    """
    candidates = bench[
        bench["Actionable_Eligible"]
        &
        ~_threshold_mask(
            bench["Observed_DIFOT"],
            service_threshold,
        )
    ].copy()

    if candidates.empty:
        return candidates

    candidates["Service_Gap"] = (
        service_threshold
        - candidates["Observed_DIFOT"]
    )

    candidates = candidates.sort_values(
        [
            "Product_Category",
            "Service_Gap",
            "Observed_Cost_per_Ton_USD",
            "Available_Shipments",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    )

    fallbacks = (
        candidates
        .groupby(
            "Product_Category",
            as_index=False,
        )
        .first()
    )

    return fallbacks.reset_index(
        drop=True
    )


def benchmark_tables(
    df: pd.DataFrame,
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD,
) -> Dict[str, pd.DataFrame]:
    """Return the historical, actionable and conditional fallback tables."""
    bench = _product_route_benchmarks(
        df,
        service_threshold,
    )

    historical = _select_benchmark_routes(
        bench,
        service_threshold,
        mode="historical",
    )

    actionable = _select_benchmark_routes(
        bench,
        service_threshold,
        mode="actionable",
    )

    fallback = _select_conditional_fallbacks(
        bench,
        service_threshold,
    )

    return {
        "all_product_route_evidence": bench,
        "historical_benchmark": historical,
        "actionable_benchmark": actionable,
        "conditional_fallback": fallback,
    }


# ============================================================
# ROUTE EVIDENCE LOOKUP
# ============================================================

def route_evidence(
    df: pd.DataFrame,
    product: str,
    route: str,
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD,
) -> Dict[str, object]:
    """Return evidence statistics for one Product × Route combination."""
    work = _ensure_analysis_columns(df)

    group = work[
        (
            work["Product_Category"]
            == product
        )
        &
        (
            work["Route_Canonical"]
            == route
        )
    ].copy()

    if group.empty:
        raise KeyError(
            f"No observations found for {product} × {route}."
        )

    n = len(group)

    successes = int(
        group["DIFOT_Flag"].sum()
    )

    difot = (
        successes / n
    )

    wilson_low, wilson_high = _wilson_interval(
        successes,
        n,
    )

    loo = _leave_one_out_difot(
        group,
        service_threshold,
    )

    costs = (
        group[
            "Cost_per_Ton_USD"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    return {
        "product": product,
        "route": route,
        "n": n,
        "difot": difot,
        "difot_pct": difot * 100.0,
        "wilson_lower": wilson_low,
        "wilson_upper": wilson_high,
        "wilson_lower_pct": wilson_low * 100.0,
        "wilson_upper_pct": wilson_high * 100.0,
        "evidence_level": _evidence_level(n),
        "median_cost_per_ton": (
            float(np.median(costs))
            if len(costs)
            else np.nan
        ),
        "loo_min_difot": loo[
            "LOO_Min_DIFOT"
        ],
        "loo_max_difot": loo[
            "LOO_Max_DIFOT"
        ],
        "loo_mean_difot": loo[
            "LOO_Mean_DIFOT"
        ],
        "loo_below_threshold_count":
            loo[
                "LOO_Below_Threshold_Count"
            ],
        "loo_at_least_threshold_count":
            loo[
                "LOO_At_Least_Threshold_Count"
            ],
        "borderline_at_threshold":
            loo[
                "Borderline_At_Threshold"
            ],
        "meets_observed_threshold":
            bool(
                difot
                >= (
                    service_threshold
                    - 1e-10
                )
            ),
    }


# ============================================================
# ROUTE COST COMPARISON
# ============================================================

def compare_route_costs(
    df: pd.DataFrame,
    product: str,
    route_a: str,
    route_b: str,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, object]:
    """Compare Product × Route cost/ton distributions.

    Returns median difference, bootstrap CI, Mann–Whitney p-value,
    and Cliff's delta.
    """
    work = _ensure_analysis_columns(df)

    a = (
        work[
            (
                work["Product_Category"]
                == product
            )
            &
            (
                work["Route_Canonical"]
                == route_a
            )
        ]["Cost_per_Ton_USD"]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    b = (
        work[
            (
                work["Product_Category"]
                == product
            )
            &
            (
                work["Route_Canonical"]
                == route_b
            )
        ]["Cost_per_Ton_USD"]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(a) == 0:
        raise KeyError(
            f"No cost observations for {product} × {route_a}."
        )

    if len(b) == 0:
        raise KeyError(
            f"No cost observations for {product} × {route_b}."
        )

    median_a = float(
        np.median(a)
    )
    median_b = float(
        np.median(b)
    )

    difference, ci_low, ci_high = (
        _bootstrap_median_difference(
            a,
            b,
            iterations=bootstrap_iterations,
            seed=random_seed,
        )
    )

    test = mannwhitneyu(
        a,
        b,
        alternative="two-sided",
    )

    cliffs_delta = _cliffs_delta(
        a,
        b,
    )

    return {
        "product": product,
        "route_a": route_a,
        "route_b": route_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "median_a_usd_per_ton": median_a,
        "median_b_usd_per_ton": median_b,
        "median_difference_a_minus_b_usd_per_ton":
            difference,
        "bootstrap_ci_lower":
            ci_low,
        "bootstrap_ci_upper":
            ci_high,
        "mann_whitney_p":
            float(test.pvalue),
        "cliffs_delta":
            cliffs_delta,
    }


# ============================================================
# COUNTERFACTUALS
# ============================================================

def _merge_selected_benchmark(
    work: pd.DataFrame,
    selected: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Merge one benchmark table into shipment rows."""
    if selected.empty:
        return work.copy()

    benchmark = selected[
        [
            "Product_Category",
            "Route_Canonical",
            "Observed_Cost_per_Ton_USD",
            "Observed_DIFOT",
            "Available_Shipments",
            "Evidence_Level",
            "Borderline_Service_Evidence",
            "Wilson_DIFOT_Lower",
            "Wilson_DIFOT_Upper",
            "LOO_Min_DIFOT",
            "LOO_Max_DIFOT",
        ]
    ].rename(
        columns={
            "Route_Canonical":
                f"{prefix}_Route",
            "Observed_Cost_per_Ton_USD":
                f"{prefix}_Cost_per_Ton_USD",
            "Observed_DIFOT":
                f"{prefix}_DIFOT",
            "Available_Shipments":
                f"{prefix}_N",
            "Evidence_Level":
                f"{prefix}_Evidence_Level",
            "Borderline_Service_Evidence":
                f"{prefix}_Borderline_Evidence",
            "Wilson_DIFOT_Lower":
                f"{prefix}_Wilson_Lower",
            "Wilson_DIFOT_Upper":
                f"{prefix}_Wilson_Upper",
            "LOO_Min_DIFOT":
                f"{prefix}_LOO_Min_DIFOT",
            "LOO_Max_DIFOT":
                f"{prefix}_LOO_Max_DIFOT",
        }
    )

    return work.merge(
        benchmark,
        on="Product_Category",
        how="left",
    )


def counterfactual_metrics(
    df: pd.DataFrame,
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate historical and actionable counterfactuals.

    Counterfactual A — Historical:
        Cheapest observed non-held route meeting the service threshold.
        Direct is allowed because this is a pre-blockade/reference question.

    Counterfactual B — Actionable:
        Cheapest observed non-Direct, non-Held route meeting the service
        threshold with at least five observed shipments.

    Conditional fallback:
        If no actionable compliant route exists for a product, return the
        observed non-Direct/non-Held route with the smallest service gap
        below the threshold.

    The original workbook is never modified.

    Returns
    -------
    shipment_cf:
        One row per shipment with both counterfactual frameworks.

    route_summary:
        Current-route aggregation of actionable avoidable/residual exposure.
    """
    work = _ensure_analysis_columns(df)

    bench = _product_route_benchmarks(
        work,
        service_threshold,
    )

    historical = _select_benchmark_routes(
        bench,
        service_threshold,
        mode="historical",
    )

    actionable = _select_benchmark_routes(
        bench,
        service_threshold,
        mode="actionable",
    )

    fallback = _select_conditional_fallbacks(
        bench,
        service_threshold,
    )

    # --------------------------------------------------------
    # Historical benchmark
    # --------------------------------------------------------

    work = _merge_selected_benchmark(
        work,
        historical,
        "Historical_Benchmark",
    )

    # --------------------------------------------------------
    # Actionable benchmark
    # --------------------------------------------------------

    work = _merge_selected_benchmark(
        work,
        actionable,
        "Actionable_Benchmark",
    )

    # --------------------------------------------------------
    # Conditional fallback
    # --------------------------------------------------------

    if not fallback.empty:
        fallback_merge = fallback[
            [
                "Product_Category",
                "Route_Canonical",
                "Observed_Cost_per_Ton_USD",
                "Observed_DIFOT",
                "Available_Shipments",
                "Evidence_Level",
                "Borderline_Service_Evidence",
            ]
        ].rename(
            columns={
                "Route_Canonical":
                    "Conditional_Fallback_Route",
                "Observed_Cost_per_Ton_USD":
                    "Conditional_Fallback_Cost_per_Ton_USD",
                "Observed_DIFOT":
                    "Conditional_Fallback_DIFOT",
                "Available_Shipments":
                    "Conditional_Fallback_N",
                "Evidence_Level":
                    "Conditional_Fallback_Evidence_Level",
                "Borderline_Service_Evidence":
                    "Conditional_Fallback_Borderline_Evidence",
            }
        )

        work = work.merge(
            fallback_merge,
            on="Product_Category",
            how="left",
        )

    # --------------------------------------------------------
    # Guarantee expected columns even if a benchmark table is empty.
    # --------------------------------------------------------

    expected_columns = [
        "Historical_Benchmark_Route",
        "Historical_Benchmark_Cost_per_Ton_USD",
        "Historical_Benchmark_DIFOT",
        "Historical_Benchmark_N",
        "Historical_Benchmark_Evidence_Level",
        "Historical_Benchmark_Borderline_Evidence",
        "Historical_Benchmark_Wilson_Lower",
        "Historical_Benchmark_Wilson_Upper",
        "Historical_Benchmark_LOO_Min_DIFOT",
        "Historical_Benchmark_LOO_Max_DIFOT",
        "Actionable_Benchmark_Route",
        "Actionable_Benchmark_Cost_per_Ton_USD",
        "Actionable_Benchmark_DIFOT",
        "Actionable_Benchmark_N",
        "Actionable_Benchmark_Evidence_Level",
        "Actionable_Benchmark_Borderline_Evidence",
        "Actionable_Benchmark_Wilson_Lower",
        "Actionable_Benchmark_Wilson_Upper",
        "Actionable_Benchmark_LOO_Min_DIFOT",
        "Actionable_Benchmark_LOO_Max_DIFOT",
        "Conditional_Fallback_Route",
        "Conditional_Fallback_Cost_per_Ton_USD",
        "Conditional_Fallback_DIFOT",
        "Conditional_Fallback_N",
        "Conditional_Fallback_Evidence_Level",
        "Conditional_Fallback_Borderline_Evidence",
    ]

    for column in expected_columns:
        if column not in work.columns:
            work[column] = np.nan

    # --------------------------------------------------------
    # Availability flags
    # --------------------------------------------------------

    work["Historical_Benchmark_Available"] = (
        work[
            "Historical_Benchmark_Cost_per_Ton_USD"
        ].notna()
    )

    work["Actionable_Benchmark_Available"] = (
        work[
            "Actionable_Benchmark_Cost_per_Ton_USD"
        ].notna()
    )

    work["Conditional_Fallback_Available"] = (
        work[
            "Conditional_Fallback_Cost_per_Ton_USD"
        ].notna()
    )

    # --------------------------------------------------------
    # Current economics
    # --------------------------------------------------------

    work["Current_Cost_USD"] = (
        work["Cost_per_Ton_USD"]
        * work["Cargo_Weight_Tons"]
    )

    # --------------------------------------------------------
    # Historical counterfactual
    # --------------------------------------------------------

    work["Historical_Counterfactual_Cost_USD"] = (
        work[
            "Historical_Benchmark_Cost_per_Ton_USD"
        ]
        * work["Cargo_Weight_Tons"]
    )

    work["Historical_Counterfactual_Margin_USD"] = (
        work[
            "Contracted_Freight_Revenue_USD"
        ]
        - work[
            "Historical_Counterfactual_Cost_USD"
        ]
    )

    historical_available = work[
        "Historical_Benchmark_Available"
    ]

    work[
        "Historical_Avoidable_Exposure_USD"
    ] = np.where(
        historical_available,
        (
            work[
                "Historical_Counterfactual_Margin_USD"
            ]
            -
            work[
                "Gross_Margin_USD"
            ]
        ).clip(lower=0),
        np.nan,
    )

    work[
        "Historical_Residual_Exposure_USD"
    ] = np.where(
        historical_available,
        (
            -work[
                "Historical_Counterfactual_Margin_USD"
            ]
        ).clip(lower=0),
        np.nan,
    )

    work[
        "Historical_Cost_Gap_USD"
    ] = np.where(
        historical_available,
        (
            work["Current_Cost_USD"]
            -
            work[
                "Historical_Counterfactual_Cost_USD"
            ]
        ),
        np.nan,
    )

    # --------------------------------------------------------
    # Actionable counterfactual
    # --------------------------------------------------------

    work["Counterfactual_Cost_USD"] = (
        work[
            "Actionable_Benchmark_Cost_per_Ton_USD"
        ]
        * work["Cargo_Weight_Tons"]
    )

    work["Counterfactual_Margin_USD"] = (
        work[
            "Contracted_Freight_Revenue_USD"
        ]
        -
        work[
            "Counterfactual_Cost_USD"
        ]
    )

    actionable_available = work[
        "Actionable_Benchmark_Available"
    ]

    work["Avoidable_Exposure_USD"] = np.where(
        actionable_available,
        (
            work[
                "Counterfactual_Margin_USD"
            ]
            -
            work[
                "Gross_Margin_USD"
            ]
        ).clip(lower=0),
        np.nan,
    )

    work["Residual_Exposure_USD"] = np.where(
        actionable_available,
        (
            -work[
                "Counterfactual_Margin_USD"
            ]
        ).clip(lower=0),
        np.nan,
    )

    work["Cost_Gap_USD"] = np.where(
        actionable_available,
        (
            work["Current_Cost_USD"]
            -
            work["Counterfactual_Cost_USD"]
        ),
        np.nan,
    )

    work["Actionable_Cost_Gap_Pct"] = np.where(
        actionable_available
        &
        (
            work[
                "Counterfactual_Cost_USD"
            ].abs()
            > 1e-12
        ),
        (
            work["Cost_Gap_USD"]
            /
            work[
                "Counterfactual_Cost_USD"
            ].abs()
        ),
        np.nan,
    )

    # Current route's observed Product × Route DIFOT.
    route_evidence_table = bench[
        [
            "Product_Category",
            "Route_Canonical",
            "Observed_DIFOT",
            "Wilson_DIFOT_Lower",
            "Wilson_DIFOT_Upper",
            "Evidence_Level",
            "Borderline_Service_Evidence",
        ]
    ].rename(
        columns={
            "Route_Canonical":
                "Current_Route_For_Merge",
            "Observed_DIFOT":
                "Current_Route_DIFOT",
            "Wilson_DIFOT_Lower":
                "Current_Route_Wilson_Lower",
            "Wilson_DIFOT_Upper":
                "Current_Route_Wilson_Upper",
            "Evidence_Level":
                "Current_Route_Evidence_Level",
            "Borderline_Service_Evidence":
                "Current_Route_Borderline_Evidence",
        }
    )

    work = work.merge(
        route_evidence_table,
        left_on=[
            "Product_Category",
            "Route_Canonical",
        ],
        right_on=[
            "Product_Category",
            "Current_Route_For_Merge",
        ],
        how="left",
    ).drop(
        columns=["Current_Route_For_Merge"]
    )

    # --------------------------------------------------------
    # Backward-compatible aliases
    #
    # Existing pages that used Best_Service_Compliant_* now point
    # to the actionable benchmark rather than the historical one.
    # --------------------------------------------------------

    work["Best_Service_Compliant_Route"] = work[
        "Actionable_Benchmark_Route"
    ]

    work[
        "Best_Service_Compliant_Cost_per_Ton_USD"
    ] = work[
        "Actionable_Benchmark_Cost_per_Ton_USD"
    ]

    work[
        "Best_Service_Compliant_DIFOT"
    ] = work[
        "Actionable_Benchmark_DIFOT"
    ]

    work[
        "Service_Compliant_Benchmark_Available"
    ] = work[
        "Actionable_Benchmark_Available"
    ]

    # --------------------------------------------------------
    # Explicit framework labels
    # --------------------------------------------------------

    work[
        "Actionable_Benchmark_Type"
    ] = np.where(
        work[
            "Actionable_Benchmark_Available"
        ],
        "Disruption-era actionable",
        "None",
    )

    work[
        "Historical_Benchmark_Type"
    ] = np.where(
        work[
            "Historical_Benchmark_Available"
        ],
        "Pre-blockade historical reference",
        "None",
    )

    # --------------------------------------------------------
    # Current route vs actionable benchmark
    # --------------------------------------------------------

    work[
        "Current_Route_Is_Actionable_Benchmark"
    ] = (
        work[
            "Actionable_Benchmark_Route"
        ]
        == work[
            "Route_Canonical"
        ]
    )

    work[
        "Actionable_Benchmark_Service_Gap_Pp"
    ] = (
        work[
            "Actionable_Benchmark_DIFOT"
        ]
        - service_threshold
    ) * 100.0

    work[
        "Conditional_Fallback_Service_Gap_Pp"
    ] = np.where(
        work[
            "Conditional_Fallback_DIFOT"
        ].notna(),
        (
            work[
                "Conditional_Fallback_DIFOT"
            ]
            -
            service_threshold
        ) * 100.0,
        np.nan,
    )

    # --------------------------------------------------------
    # Route summary for current-route management view
    # --------------------------------------------------------

    route_summary = (
        work.groupby(
            "Route_Canonical",
            dropna=False,
        )
        .agg(
            Shipments=(
                "Shipment_ID",
                "nunique",
            ),
            Avoidable_Exposure_USD=(
                "Avoidable_Exposure_USD",
                "sum",
            ),
            Residual_Exposure_USD=(
                "Residual_Exposure_USD",
                "sum",
            ),
            Cost_Gap_USD=(
                "Cost_Gap_USD",
                "sum",
            ),
            Contracted_Revenue_USD=(
                "Contracted_Freight_Revenue_USD",
                "sum",
            ),
            Actionable_Benchmark_Available_Shipments=(
                "Actionable_Benchmark_Available",
                "sum",
            ),
        )
        .reset_index()
    )

    route_summary = route_summary.sort_values(
        "Avoidable_Exposure_USD",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return (
        work,
        route_summary,
    )


# ============================================================
# THRESHOLD SENSITIVITY
# ============================================================

def threshold_sensitivity(
    df: pd.DataFrame,
    thresholds: Iterable[float] = tuple(
        np.round(
            np.arange(
                0.50,
                1.001,
                0.01,
            ),
            2,
        )
    ),
) -> pd.DataFrame:
    """Run the deterministic service-threshold sensitivity analysis.

    Actionable alternatives:
        - exclude Direct and Held
        - require at least five observations
        - require observed DIFOT >= threshold

    Economic coverage is measured over non-held shipments because this
    analysis asks how much of the currently moving book has an observed
    disruption-era alternative.
    """
    work = _ensure_analysis_columns(df)

    nonheld = work[
        ~work["Is_Held"]
    ].copy()

    total_nonheld_revenue = float(
        nonheld[
            "Contracted_Freight_Revenue_USD"
        ].sum()
    )

    total_nonheld_shipments = len(
        nonheld
    )

    total_products = (
        nonheld[
            "Product_Category"
        ]
        .nunique()
    )

    results = []

    for threshold in thresholds:
        threshold = float(threshold)

        bench = _product_route_benchmarks(
            work,
            threshold,
        )

        selected = _select_benchmark_routes(
            bench,
            threshold,
            mode="actionable",
        )

        if selected.empty:
            selected_products = set()
        else:
            selected_products = set(
                selected[
                    "Product_Category"
                ]
            )

        covered = nonheld[
            "Product_Category"
        ].isin(
            selected_products
        )

        shipment_coverage = (
            covered.mean()
            if total_nonheld_shipments
            else 0.0
        )

        revenue_coverage = (
            nonheld.loc[
                covered,
                "Contracted_Freight_Revenue_USD",
            ].sum()
            / total_nonheld_revenue
            if total_nonheld_revenue
            else 0.0
        )

        # Apply product-level benchmark economics.
        merged = nonheld.merge(
            selected[
                [
                    "Product_Category",
                    "Route_Canonical",
                    "Observed_Cost_per_Ton_USD",
                    "Observed_DIFOT",
                ]
            ].rename(
                columns={
                    "Route_Canonical":
                        "Benchmark_Route",
                    "Observed_Cost_per_Ton_USD":
                        "Benchmark_Cost_per_Ton",
                    "Observed_DIFOT":
                        "Benchmark_DIFOT",
                }
            )
            if not selected.empty
            else pd.DataFrame(
                columns=[
                    "Product_Category",
                    "Benchmark_Route",
                    "Benchmark_Cost_per_Ton",
                    "Benchmark_DIFOT",
                ]
            ),
            on="Product_Category",
            how="left",
        )

        benchmark_available = (
            merged[
                "Benchmark_Route"
            ].notna()
        )

        merged["Benchmark_Cost_USD"] = (
            merged[
                "Benchmark_Cost_per_Ton"
            ]
            * merged[
                "Cargo_Weight_Tons"
            ]
        )

        merged["Benchmark_Margin_USD"] = (
            merged[
                "Contracted_Freight_Revenue_USD"
            ]
            -
            merged[
                "Benchmark_Cost_USD"
            ]
        )

        merged["Addressable_Loss_USD"] = np.where(
            benchmark_available,
            (
                merged[
                    "Benchmark_Margin_USD"
                ]
                -
                merged[
                    "Gross_Margin_USD"
                ]
            ).clip(lower=0),
            0.0,
        )

        merged["Residual_Loss_USD"] = np.where(
            benchmark_available,
            (
                -merged[
                    "Benchmark_Margin_USD"
                ]
            ).clip(lower=0),
            0.0,
        )

        results.append(
            {
                "Threshold": threshold,
                "Products_Covered":
                    int(
                        len(
                            selected_products
                        )
                    ),
                "Product_Coverage_Pct":
                    (
                        len(
                            selected_products
                        )
                        / total_products
                        * 100.0
                        if total_products
                        else 0.0
                    ),
                "Shipment_Coverage_Pct":
                    shipment_coverage
                    * 100.0,
                "Revenue_Coverage_Pct":
                    revenue_coverage
                    * 100.0,
                "Addressable_Loss_USD":
                    float(
                        merged[
                            "Addressable_Loss_USD"
                        ].sum()
                    ),
                "Residual_Loss_USD":
                    float(
                        merged[
                            "Residual_Loss_USD"
                        ].sum()
                    ),
                "Addressable_Loss_M":
                    float(
                        merged[
                            "Addressable_Loss_USD"
                        ].sum()
                    ) / 1e6,
                "Residual_Loss_M":
                    float(
                        merged[
                            "Residual_Loss_USD"
                        ].sum()
                    ) / 1e6,
            }
        )

    return pd.DataFrame(
        results
    ).sort_values(
        "Threshold"
    ).reset_index(
        drop=True
    )


# ============================================================
# PUBLIC SUMMARY OF THE FINAL FRAMEWORK
# ============================================================

def decision_framework_summary(
    df: pd.DataFrame,
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD,
) -> Dict[str, object]:
    """Return a compact, audit-friendly summary of the benchmark framework."""
    tables = benchmark_tables(
        df,
        service_threshold,
    )

    actionable = tables[
        "actionable_benchmark"
    ].copy()

    fallback = tables[
        "conditional_fallback"
    ].copy()

    summary = {
        "service_threshold": service_threshold,
        "historical_benchmark":
            tables[
                "historical_benchmark"
            ],
        "actionable_benchmark":
            actionable,
        "conditional_fallback":
            fallback,
        "actionable_products_covered":
            (
                int(
                    actionable[
                        "Product_Category"
                    ].nunique()
                )
                if not actionable.empty
                else 0
            ),
        "total_products":
            int(
                df[
                    "Product_Category"
                ].nunique()
            ),
    }

    return summary
