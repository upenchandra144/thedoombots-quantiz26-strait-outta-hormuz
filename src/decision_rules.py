from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import DEFAULT_SERVICE_THRESHOLD, counterfactual_metrics, product_route_metrics


# Central decision parameters. Keep these in one place so the competition team
# can tune the decision engine without rewriting any Streamlit page.
DEFAULT_COST_GAP_PCT = 0.15
DEFAULT_SERVICE_ADVANTAGE_PCT = 0.05
HIGH_CONSEQUENCE_PRODUCTS = {
    "High-Tech Components",
    "Pharmaceuticals",
}
PREMIUM_MODES = {"Air"}


@dataclass(frozen=True)
class DecisionThresholds:
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD
    material_cost_gap_pct: float = DEFAULT_COST_GAP_PCT
    service_advantage_pct: float = DEFAULT_SERVICE_ADVANTAGE_PCT


@dataclass
class ShipmentDecision:
    shipment_id: str
    primary_action: str
    secondary_action: Optional[str]
    explanation: List[str]
    warnings: List[str]
    current_route: str
    current_cost_per_ton: float
    current_total_cost: float
    current_margin: float
    current_margin_per_ton: float
    current_difot_pct: float
    best_service_compliant_route: Optional[str]
    best_service_compliant_cost_per_ton: Optional[float]
    best_service_compliant_difot_pct: Optional[float]
    counterfactual_cost: Optional[float]
    counterfactual_margin: Optional[float]
    cost_gap: Optional[float]
    cost_gap_pct: Optional[float]
    avoidable_exposure: Optional[float]
    residual_exposure: Optional[float]
    benchmark_available: bool

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def _pct(value: float | int | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _money(value: float | int | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _route_benchmarks(df: pd.DataFrame, product: str, service_threshold: float) -> pd.DataFrame:
    routes = product_route_metrics(df)
    return routes[
        (routes["Product_Category"] == product)
        & (routes["DIFOT"] >= service_threshold)
    ].sort_values("Median_Cost_per_Ton_USD")


def _get_selected_row(df: pd.DataFrame, shipment_id: str) -> pd.Series:
    match = df[df["Shipment_ID"].astype(str).eq(str(shipment_id))]
    if match.empty:
        raise KeyError(f"Shipment not found: {shipment_id}")
    return match.iloc[0]


def _build_reasoning(
    row: pd.Series,
    best: Optional[pd.Series],
    primary_action: str,
    thresholds: DecisionThresholds,
    cost_gap_pct: Optional[float],
    avoidable_exposure: Optional[float],
    residual_exposure: Optional[float],
) -> List[str]:
    reasons: List[str] = []
    route = str(row["Route_Canonical"])
    current_difot = _pct(row["DIFOT_Flag"])

    if route == "Held":
        reasons.append("Revenue is currently trapped, DIFOT is 0%, and delay exposure continues to accumulate.")
        if avoidable_exposure is not None:
            reasons.append(
                f"The modeled benchmark framework identifies ${avoidable_exposure:,.0f} of potentially addressable exposure."
            )
        reasons.append("Release should be assessed before preserving the current held position.")
        return reasons

    if best is None:
        reasons.append(
            f"No observed {int(thresholds.service_threshold * 100)}% DIFOT service-compliant benchmark is available for this product."
        )
        if primary_action == "REPRICE":
            reasons.append("Commercial repricing is therefore required rather than assuming an operational substitute.")
        return reasons

    best_route = str(best["Route_Canonical"])
    best_cost = _money(best["Median_Cost_per_Ton_USD"])
    best_difot = _pct(best["DIFOT"])
    current_cost = _money(row["Cost_per_Ton_USD"])

    if cost_gap_pct is not None and cost_gap_pct > 0:
        reasons.append(
            f"Current cost is {cost_gap_pct * 100:.0f}% above the best observed service-compliant benchmark ({best_route}, ${best_cost:,.2f}/ton)."
        )

    if current_difot < thresholds.service_threshold:
        reasons.append(
            f"Current DIFOT is {current_difot * 100:.1f}%, below the {thresholds.service_threshold * 100:.0f}% service threshold."
        )

    service_advantage = current_difot - best_difot
    if service_advantage >= thresholds.service_advantage_pct:
        reasons.append(
            f"Current service is {service_advantage * 100:.1f} percentage points better than the benchmark, providing a documented service advantage."
        )

    if avoidable_exposure is not None and avoidable_exposure > 0:
        reasons.append(f"Modeled avoidable exposure is ${avoidable_exposure:,.0f} under the benchmark framework.")

    if residual_exposure is not None and residual_exposure > 0:
        reasons.append(
            f"${residual_exposure:,.0f} remains as residual exposure after applying the observed benchmark."
        )

    if primary_action == "SELECTIVE PROTECT":
        reasons.append(
            "The route is a premium service mode, so protection is reserved for high-consequence cargo rather than treated as a universal route preference."
        )

    if primary_action == "MONITOR / RETAIN":
        reasons.append(
            "Current economics and service performance do not trigger a material intervention under the stated decision rules."
        )

    return reasons


def assess_shipment(
    df: pd.DataFrame,
    shipment_id: str,
    thresholds: DecisionThresholds | None = None,
) -> ShipmentDecision:
    """Return a transparent management action for one shipment."""
    thresholds = thresholds or DecisionThresholds()
    row = _get_selected_row(df, shipment_id)

    # Reuse the canonical benchmark logic from metrics.py.
    cf_df, _ = counterfactual_metrics(df, thresholds.service_threshold)
    cf = _get_selected_row(cf_df, shipment_id)

    current_route = str(row["Route_Canonical"])
    current_cost_per_ton = _money(row["Cost_per_Ton_USD"])
    current_total_cost = _money(row["Total_Cost_to_Serve_USD"])
    current_margin = _money(row["Gross_Margin_USD"])
    current_weight = _money(row["Cargo_Weight_Tons"])
    current_margin_per_ton = _money(
        current_margin / current_weight if current_weight else np.nan
    )
    current_difot_pct = _pct(row["DIFOT_Flag"]) * 100.0

    best_route = cf.get("Best_Service_Compliant_Route")
    best_cost = cf.get("Best_Service_Compliant_Cost_per_Ton_USD")
    best_difot = cf.get("Best_Service_Compliant_DIFOT")

    benchmark_available = pd.notna(best_route) and pd.notna(best_cost)
    best_row = None
    if benchmark_available:
        product_routes = _route_benchmarks(
            df,
            str(row["Product_Category"]),
            thresholds.service_threshold,
        )
        if not product_routes.empty:
            candidate = product_routes[
                product_routes["Route_Canonical"].eq(str(best_route))
            ]
            if not candidate.empty:
                best_row = candidate.iloc[0]

    counterfactual_cost = _money(cf.get("Counterfactual_Cost_USD")) if benchmark_available else None
    counterfactual_margin = _money(cf.get("Counterfactual_Margin_USD")) if benchmark_available else None
    cost_gap = _money(cf.get("Cost_Gap_USD")) if benchmark_available else None
    cost_gap_pct = (
        (current_cost_per_ton - _money(best_cost)) / _money(best_cost)
        if benchmark_available and _money(best_cost) > 0
        else None
    )
    avoidable_exposure = _money(cf.get("Avoidable_Exposure_USD")) if benchmark_available else None
    residual_exposure = _money(cf.get("Residual_Exposure_USD")) if benchmark_available else None

    warnings: List[str] = []
    primary_action = "MONITOR / RETAIN"
    secondary_action: Optional[str] = None

    # RULE 1 — held cargo is an operational release problem first.
    if current_route == "Held":
        primary_action = "RELEASE"
        if current_route == "Held" and pd.isna(row.get("Actual_Transit_Days")):
            warnings.append("Transit time: Held / unresolved")

    # RULE 4 — premium service should only be protected selectively.
    elif (
        current_route in PREMIUM_MODES
        and bool(row["DIFOT_Flag"])
        and str(row["Product_Category"]) in HIGH_CONSEQUENCE_PRODUCTS
    ):
        primary_action = "SELECTIVE PROTECT"
        secondary_action = "REPRICE" if current_margin < 0 else None
        warnings.append("Premium mode: use only where service consequence justifies the cost.")

    # No benchmark means no invented alternative.
    elif not benchmark_available:
        primary_action = "REPRICE"
        warnings.append(
            f"No observed route for {row['Product_Category']} meets the {thresholds.service_threshold * 100:.0f}% DIFOT threshold."
        )

    else:
        service_advantage = _pct(row["DIFOT_Flag"]) - _pct(best_difot)
        material_cost_gap = bool(cost_gap_pct is not None and cost_gap_pct >= thresholds.material_cost_gap_pct)
        benchmark_is_loss_making = bool(counterfactual_margin is not None and counterfactual_margin < 0)

        if material_cost_gap and service_advantage < thresholds.service_advantage_pct:
            primary_action = "REROUTE"
            # If the compliant benchmark is still structurally loss-making,
            # make repricing the commercial follow-on action.
            if benchmark_is_loss_making:
                secondary_action = "REPRICE"
        elif benchmark_is_loss_making and current_margin < 0:
            primary_action = "REPRICE"
        elif bool(row["DIFOT_Flag"]) and current_margin >= 0:
            primary_action = "MONITOR / RETAIN"
        elif bool(row["DIFOT_Flag"]) and current_margin < 0:
            primary_action = "REPRICE"
        else:
            primary_action = "REROUTE"

    # Direct is explicitly a historical/pre-blockade benchmark, not a guaranteed
    # future capacity commitment.
    if current_route == "Direct":
        warnings.append("Direct is the pre-blockade benchmark, not a guaranteed post-blockade route.")

    if benchmark_available and str(best_route) == "Direct":
        warnings.append("Benchmark route is Direct; treat it as a historical service-compliant benchmark.")

    if benchmark_available:
        service_text = f"{_pct(best_difot) * 100:.1f}% DIFOT"
        warnings.append(f"Best observed service-compliant benchmark: {best_route} ({service_text}).")

    explanation = _build_reasoning(
        row,
        best_row,
        primary_action,
        thresholds,
        cost_gap_pct,
        avoidable_exposure,
        residual_exposure,
    )

    return ShipmentDecision(
        shipment_id=str(shipment_id),
        primary_action=primary_action,
        secondary_action=secondary_action,
        explanation=explanation,
        warnings=warnings,
        current_route=current_route,
        current_cost_per_ton=current_cost_per_ton,
        current_total_cost=current_total_cost,
        current_margin=current_margin,
        current_margin_per_ton=current_margin_per_ton,
        current_difot_pct=current_difot_pct,
        best_service_compliant_route=str(best_route) if benchmark_available else None,
        best_service_compliant_cost_per_ton=_money(best_cost) if benchmark_available else None,
        best_service_compliant_difot_pct=_pct(best_difot) * 100.0 if benchmark_available else None,
        counterfactual_cost=counterfactual_cost,
        counterfactual_margin=counterfactual_margin,
        cost_gap=cost_gap,
        cost_gap_pct=cost_gap_pct,
        avoidable_exposure=avoidable_exposure,
        residual_exposure=residual_exposure,
        benchmark_available=benchmark_available,
    )


def decision_for_row(
    df: pd.DataFrame,
    row: pd.Series,
    thresholds: DecisionThresholds | None = None,
) -> ShipmentDecision:
    """Convenience wrapper when a page already has a selected row."""
    return assess_shipment(df, str(row["Shipment_ID"]), thresholds=thresholds)
