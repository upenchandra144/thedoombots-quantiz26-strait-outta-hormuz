from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import (
    DEFAULT_SERVICE_THRESHOLD,
    MIN_ACTIONABLE_BENCHMARK_SHIPMENTS,
    counterfactual_metrics,
)


# ============================================================
# DECISION ENGINE PARAMETERS
# ============================================================

# Premium modes are not automatically protected. They receive a
# separate service-premium check based on observed route evidence.
PREMIUM_MODES = {"Air"}

# Numerical tolerance only; this is NOT a business threshold.
NUMERIC_TOLERANCE = 1e-10


@dataclass(frozen=True)
class DecisionThresholds:
    """Parameters governing the deterministic decision framework.

    The only business service constraint is the minimum observed DIFOT
    requirement. Cost-gap and service-advantage percentages have been
    deliberately removed; the engine now uses direct economic comparison
    and observed route-service evidence instead of arbitrary trigger values.
    """
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD


@dataclass
class ShipmentDecision:
    """Transparent, explainable management decision for one shipment."""

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

    # --------------------------------------------------------
    # Backward-compatible aliases used by existing pages.
    # These now refer to the ACTIONABLE disruption-era benchmark.
    # --------------------------------------------------------
    best_service_compliant_route: Optional[str]
    best_service_compliant_cost_per_ton: Optional[float]
    best_service_compliant_difot_pct: Optional[float]

    # --------------------------------------------------------
    # Explicit actionable benchmark fields.
    # --------------------------------------------------------
    actionable_benchmark_route: Optional[str]
    actionable_benchmark_cost_per_ton: Optional[float]
    actionable_benchmark_difot_pct: Optional[float]
    actionable_benchmark_n: Optional[int]
    actionable_benchmark_evidence_level: Optional[str]
    actionable_benchmark_borderline: bool
    actionable_benchmark_wilson_lower_pct: Optional[float]
    actionable_benchmark_wilson_upper_pct: Optional[float]
    actionable_benchmark_loo_min_difot_pct: Optional[float]
    actionable_benchmark_loo_max_difot_pct: Optional[float]

    # --------------------------------------------------------
    # Historical / pre-blockade benchmark.
    # --------------------------------------------------------
    historical_benchmark_route: Optional[str]
    historical_benchmark_cost_per_ton: Optional[float]
    historical_benchmark_difot_pct: Optional[float]

    # --------------------------------------------------------
    # Conditional fallback when no compliant actionable route exists.
    # --------------------------------------------------------
    conditional_fallback_route: Optional[str]
    conditional_fallback_cost_per_ton: Optional[float]
    conditional_fallback_difot_pct: Optional[float]
    conditional_fallback_n: Optional[int]

    # --------------------------------------------------------
    # Current Product × Route evidence.
    # --------------------------------------------------------
    current_route_observed_difot_pct: Optional[float]
    current_route_evidence_level: Optional[str]
    current_route_wilson_lower_pct: Optional[float]
    current_route_wilson_upper_pct: Optional[float]

    # --------------------------------------------------------
    # Economics under the actionable benchmark.
    # --------------------------------------------------------
    counterfactual_cost: Optional[float]
    counterfactual_margin: Optional[float]
    cost_gap: Optional[float]
    cost_gap_pct: Optional[float]
    avoidable_exposure: Optional[float]
    residual_exposure: Optional[float]

    benchmark_available: bool

    # --------------------------------------------------------
    # Audit-friendly basis.
    # --------------------------------------------------------
    decision_basis: str

    def to_dict(self) -> Dict[str, object]:
        """Return a serialisable dictionary representation."""
        return self.__dict__.copy()


# ============================================================
# BASIC HELPERS
# ============================================================

def _pct(value: float | int | bool | None) -> float:
    """Convert a nullable value to float."""
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _optional_float(value: object) -> Optional[float]:
    """Convert nullable values to Optional[float]."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: object) -> Optional[int]:
    """Convert nullable values to Optional[int]."""
    if value is None or pd.isna(value):
        return None
    return int(value)


def _money(value: float | int | None) -> float:
    """Convert nullable money values to a safe float."""
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _get_selected_row(
    df: pd.DataFrame,
    shipment_id: str,
) -> pd.Series:
    """Retrieve one shipment from the source dataframe."""
    match = df[
        df["Shipment_ID"]
        .astype(str)
        .eq(str(shipment_id))
    ]

    if match.empty:
        raise KeyError(
            f"Shipment not found: {shipment_id}"
        )

    return match.iloc[0]


def _route_is_premium(route: str) -> bool:
    return route in PREMIUM_MODES


def _same_route(
    route_a: Optional[str],
    route_b: Optional[str],
) -> bool:
    if route_a is None or route_b is None:
        return False

    return str(route_a) == str(route_b)


# ============================================================
# REASONING
# ============================================================

def _build_reasoning(
    row: pd.Series,
    cf: pd.Series,
    primary_action: str,
    thresholds: DecisionThresholds,
    current_route_observed_difot: Optional[float],
    cost_gap_pct: Optional[float],
    benchmark_available: bool,
) -> List[str]:
    """Create a concise audit trail for the recommendation."""

    reasons: List[str] = []

    route = str(
        row["Route_Canonical"]
    )

    product = str(
        row["Product_Category"]
    )

    current_shipment_difot = (
        _pct(row["DIFOT_Flag"])
        * 100.0
    )

    current_margin = _money(
        row["Gross_Margin_USD"]
    )

    benchmark_route = cf.get(
        "Actionable_Benchmark_Route"
    )

    benchmark_cost = _optional_float(
        cf.get(
            "Actionable_Benchmark_Cost_per_Ton_USD"
        )
    )

    benchmark_difot = _optional_float(
        cf.get(
            "Actionable_Benchmark_DIFOT"
        )
    )

    fallback_route = cf.get(
        "Conditional_Fallback_Route"
    )

    fallback_difot = _optional_float(
        cf.get(
            "Conditional_Fallback_DIFOT"
        )
    )

    avoidable_exposure = _optional_float(
        cf.get(
            "Avoidable_Exposure_USD"
        )
    )

    residual_exposure = _optional_float(
        cf.get(
            "Residual_Exposure_USD"
        )
    )

    borderline = bool(
        cf.get(
            "Actionable_Benchmark_Borderline_Evidence",
            False,
        )
    )

    current_route_level = (
        current_route_observed_difot
        if current_route_observed_difot is not None
        else np.nan
    )

    # --------------------------------------------------------
    # Held
    # --------------------------------------------------------

    if route == "Held":
        reasons.append(
            "Cargo is currently Held, so the first management problem is release rather than route optimisation."
        )

        reasons.append(
            "Held shipments have 0% observed DIFOT and currently recognise no freight revenue."
        )

        if benchmark_available and benchmark_route:
            reasons.append(
                f"After release, the observed disruption-era benchmark for {product} is {benchmark_route}."
            )
        elif fallback_route:
            reasons.append(
                f"No compliant observed reroute exists for {product}; the closest observed fallback is {fallback_route} at {fallback_difot * 100:.1f}% DIFOT."
            )

        return reasons

    # --------------------------------------------------------
    # Direct / pre-blockade
    # --------------------------------------------------------

    if route == "Direct":
        reasons.append(
            "Direct is treated as a pre-blockade historical reference, not as a guaranteed post-blockade operating option."
        )

        if primary_action == "MONITOR / RETAIN":
            reasons.append(
                "The shipment is therefore not forced into a disruption-era reroute recommendation merely because a different actionable benchmark exists."
            )

        return reasons

    # --------------------------------------------------------
    # No actionable benchmark
    # --------------------------------------------------------

    if not benchmark_available:
        reasons.append(
            f"No observed non-Direct, non-Held route for {product} meets the {thresholds.service_threshold * 100:.0f}% DIFOT service constraint with sufficient observed evidence."
        )

        if fallback_route and fallback_difot is not None:
            reasons.append(
                f"The closest observed operational fallback is {fallback_route} at {fallback_difot * 100:.1f}% DIFOT, below the required service constraint."
            )

        reasons.append(
            "The model therefore does not invent an operational substitute; the recommended response is commercial recovery through repricing or renegotiation."
        )

        if current_margin < 0:
            reasons.append(
                f"The current shipment is already loss-making at ${abs(current_margin):,.0f} of negative gross margin."
            )

        return reasons

    # --------------------------------------------------------
    # Actionable benchmark exists
    # --------------------------------------------------------

    benchmark_route = str(
        benchmark_route
    )

    if benchmark_cost is not None:
        reasons.append(
            f"Cheapest observed disruption-era compliant benchmark: {benchmark_route} at ${benchmark_cost:,.2f}/ton."
        )

    if benchmark_difot is not None:
        reasons.append(
            f"Benchmark observed DIFOT is {benchmark_difot * 100:.1f}%, against the {thresholds.service_threshold * 100:.0f}% service constraint."
        )

    if current_route_level is not None and np.isfinite(current_route_level):
        reasons.append(
            f"Current Product × Route observed DIFOT is {current_route_level * 100:.1f}%."
        )
    else:
        reasons.append(
            f"This shipment itself achieved {current_shipment_difot:.1f}% DIFOT; the decision uses the route-level observed service evidence for feasibility."
        )

    if cost_gap_pct is not None:
        if cost_gap_pct > NUMERIC_TOLERANCE:
            reasons.append(
                f"Current shipment cost is {cost_gap_pct * 100:.1f}% above the benchmark cost/ton."
            )
        elif cost_gap_pct < -NUMERIC_TOLERANCE:
            reasons.append(
                f"This shipment's observed cost/ton is {-cost_gap_pct * 100:.1f}% below the benchmark median, so an automatic cost-driven reroute is not justified on this row alone."
            )

    if avoidable_exposure is not None and avoidable_exposure > 0:
        reasons.append(
            f"Modeled avoidable exposure under the actionable benchmark is ${avoidable_exposure:,.0f}."
        )

    if residual_exposure is not None and residual_exposure > 0:
        reasons.append(
            f"${residual_exposure:,.0f} remains as residual exposure after applying the observed disruption-era benchmark."
        )

    if borderline:
        reasons.append(
            "Benchmark evidence is borderline at the service threshold under leave-one-out sensitivity; treat the route as compliant on the observed sample, but not as high-confidence reliability evidence."
        )

    if primary_action == "SELECTIVE PROTECT":
        reasons.append(
            "The premium route is retained only because its observed route-level service performance exceeds the cheaper compliant benchmark; premium cost is therefore being exchanged for a documented service advantage rather than assumed to be superior."
        )

    if primary_action == "REROUTE":
        reasons.append(
            "Rerouting is driven by service feasibility plus observed economic advantage, not by an arbitrary percentage cost-gap trigger."
        )

    if primary_action == "REPRICE":
        reasons.append(
            "The operational benchmark does not remove the underlying economic loss, so commercial recovery is required alongside or instead of routing intervention."
        )

    if primary_action == "MONITOR / RETAIN":
        reasons.append(
            "The current route does not show enough observed service or economic evidence to justify an automatic intervention."
        )

    return reasons


# ============================================================
# DECISION ENGINE
# ============================================================

def assess_shipment(
    df: pd.DataFrame,
    shipment_id: str,
    thresholds: DecisionThresholds | None = None,
) -> ShipmentDecision:
    """Return a transparent management action for one shipment.

    Decision hierarchy
    ------------------
    1. Held → RELEASE.
    2. Direct → historical/pre-blockade reference; do not treat as a
       guaranteed post-blockade reroute option.
    3. No observed actionable compliant benchmark → REPRICE / RENEGOTIATE.
    4. If the current route is the actionable benchmark:
         - positive margin → MONITOR / RETAIN
         - negative margin → REPRICE
    5. If a different actionable benchmark exists:
         - non-compliant current route → REROUTE
         - premium route with a demonstrated observed service advantage →
           SELECTIVE PROTECT
         - compliant current route with higher observed cost than the
           actionable benchmark → REROUTE
         - otherwise → MONITOR / RETAIN
    6. If the compliant benchmark itself is loss-making, REPRICE becomes
       a commercial secondary action rather than pretending the disruption
       has been fully solved.

    No arbitrary 15% cost-gap or 5 percentage-point service-advantage
    threshold is used.
    """
    thresholds = (
        thresholds
        or DecisionThresholds()
    )

    if not (
        0 < thresholds.service_threshold <= 1
    ):
        raise ValueError(
            "service_threshold must be in the open interval (0, 1]."
        )

    row = _get_selected_row(
        df,
        shipment_id,
    )

    # The metrics layer is the single source of truth for both:
    # historical and actionable counterfactuals.
    cf_df, _ = counterfactual_metrics(
        df,
        thresholds.service_threshold,
    )

    cf = _get_selected_row(
        cf_df,
        shipment_id,
    )

    current_route = str(
        row["Route_Canonical"]
    )

    current_cost_per_ton = _money(
        row["Cost_per_Ton_USD"]
    )

    current_total_cost = _money(
        row["Total_Cost_to_Serve_USD"]
    )

    current_margin = _money(
        row["Gross_Margin_USD"]
    )

    current_weight = _money(
        row["Cargo_Weight_Tons"]
    )

    current_margin_per_ton = _money(
        (
            current_margin
            / current_weight
            if current_weight
            else np.nan
        )
    )

    current_difot_pct = (
        _pct(row["DIFOT_Flag"])
        * 100.0
    )

    # --------------------------------------------------------
    # Explicit actionable benchmark
    # --------------------------------------------------------

    benchmark_route = cf.get(
        "Actionable_Benchmark_Route"
    )

    benchmark_cost = _optional_float(
        cf.get(
            "Actionable_Benchmark_Cost_per_Ton_USD"
        )
    )

    benchmark_difot = _optional_float(
        cf.get(
            "Actionable_Benchmark_DIFOT"
        )
    )

    benchmark_n = _optional_int(
        cf.get(
            "Actionable_Benchmark_N"
        )
    )

    benchmark_evidence_level = (
        None
        if pd.isna(
            cf.get(
                "Actionable_Benchmark_Evidence_Level"
            )
        )
        else str(
            cf.get(
                "Actionable_Benchmark_Evidence_Level"
            )
        )
    )

    benchmark_borderline = bool(
        cf.get(
            "Actionable_Benchmark_Borderline_Evidence",
            False,
        )
    )

    benchmark_wilson_lower_pct = (
        _optional_float(
            cf.get(
                "Actionable_Benchmark_Wilson_Lower"
            )
        )
        * 100.0
        if pd.notna(
            cf.get(
                "Actionable_Benchmark_Wilson_Lower"
            )
        )
        else None
    )

    benchmark_wilson_upper_pct = (
        _optional_float(
            cf.get(
                "Actionable_Benchmark_Wilson_Upper"
            )
        )
        * 100.0
        if pd.notna(
            cf.get(
                "Actionable_Benchmark_Wilson_Upper"
            )
        )
        else None
    )

    benchmark_loo_min_pct = (
        _optional_float(
            cf.get(
                "Actionable_Benchmark_LOO_Min_DIFOT"
            )
        )
        * 100.0
        if pd.notna(
            cf.get(
                "Actionable_Benchmark_LOO_Min_DIFOT"
            )
        )
        else None
    )

    benchmark_loo_max_pct = (
        _optional_float(
            cf.get(
                "Actionable_Benchmark_LOO_Max_DIFOT"
            )
        )
        * 100.0
        if pd.notna(
            cf.get(
                "Actionable_Benchmark_LOO_Max_DIFOT"
            )
        )
        else None
    )

    benchmark_available = bool(
        cf.get(
            "Actionable_Benchmark_Available",
            False,
        )
    )

    # --------------------------------------------------------
    # Historical benchmark
    # --------------------------------------------------------

    historical_route = cf.get(
        "Historical_Benchmark_Route"
    )

    historical_cost = _optional_float(
        cf.get(
            "Historical_Benchmark_Cost_per_Ton_USD"
        )
    )

    historical_difot = _optional_float(
        cf.get(
            "Historical_Benchmark_DIFOT"
        )
    )

    # --------------------------------------------------------
    # Conditional fallback
    # --------------------------------------------------------

    fallback_route = cf.get(
        "Conditional_Fallback_Route"
    )

    fallback_cost = _optional_float(
        cf.get(
            "Conditional_Fallback_Cost_per_Ton_USD"
        )
    )

    fallback_difot = _optional_float(
        cf.get(
            "Conditional_Fallback_DIFOT"
        )
    )

    fallback_n = _optional_int(
        cf.get(
            "Conditional_Fallback_N"
        )
    )

    # --------------------------------------------------------
    # Current route evidence
    # --------------------------------------------------------

    current_route_observed_difot = (
        _optional_float(
            cf.get(
                "Current_Route_DIFOT"
            )
        )
    )

    current_route_evidence_level = (
        None
        if pd.isna(
            cf.get(
                "Current_Route_Evidence_Level"
            )
        )
        else str(
            cf.get(
                "Current_Route_Evidence_Level"
            )
        )
    )

    current_route_wilson_lower_pct = (
        _optional_float(
            cf.get(
                "Current_Route_Wilson_Lower"
            )
        )
        * 100.0
        if pd.notna(
            cf.get(
                "Current_Route_Wilson_Lower"
            )
        )
        else None
    )

    current_route_wilson_upper_pct = (
        _optional_float(
            cf.get(
                "Current_Route_Wilson_Upper"
            )
        )
        * 100.0
        if pd.notna(
            cf.get(
                "Current_Route_Wilson_Upper"
            )
        )
        else None
    )

    # --------------------------------------------------------
    # Actionable economics
    # --------------------------------------------------------

    counterfactual_cost = (
        _optional_float(
            cf.get(
                "Counterfactual_Cost_USD"
            )
        )
        if benchmark_available
        else None
    )

    counterfactual_margin = (
        _optional_float(
            cf.get(
                "Counterfactual_Margin_USD"
            )
        )
        if benchmark_available
        else None
    )

    cost_gap = (
        _optional_float(
            cf.get(
                "Cost_Gap_USD"
            )
        )
        if benchmark_available
        else None
    )

    cost_gap_pct = None

    if (
        benchmark_available
        and benchmark_cost is not None
        and benchmark_cost > NUMERIC_TOLERANCE
    ):
        cost_gap_pct = (
            current_cost_per_ton
            - benchmark_cost
        ) / benchmark_cost

    avoidable_exposure = (
        _optional_float(
            cf.get(
                "Avoidable_Exposure_USD"
            )
        )
        if benchmark_available
        else None
    )

    residual_exposure = (
        _optional_float(
            cf.get(
                "Residual_Exposure_USD"
            )
        )
        if benchmark_available
        else None
    )

    # --------------------------------------------------------
    # Initial action state
    # --------------------------------------------------------

    warnings: List[str] = []

    primary_action = "MONITOR / RETAIN"
    secondary_action: Optional[str] = None

    decision_basis = ""

    # ========================================================
    # RULE 1 — HELD
    # ========================================================

    if current_route == "Held":
        primary_action = "RELEASE"
        decision_basis = (
            "Held shipment release priority."
        )

        if pd.isna(
            row.get(
                "Actual_Transit_Days"
            )
        ):
            warnings.append(
                "Transit time: Held / unresolved."
            )

        warnings.append(
            "Held is not treated as a transport benchmark; release is the immediate operational action."
        )

    # ========================================================
    # RULE 2 — DIRECT / PRE-BLOCKADE
    # ========================================================

    elif current_route == "Direct":
        primary_action = "MONITOR / RETAIN"
        decision_basis = (
            "Direct is retained as a historical/pre-blockade reference only."
        )

        warnings.append(
            "Direct is the pre-blockade reference and is not treated as guaranteed post-blockade capacity."
        )

        if benchmark_available:
            warnings.append(
                f"Actionable disruption-era benchmark for this product is {benchmark_route}; it is shown for comparison, not used to force a reroute from the historical Direct observation."
            )

    # ========================================================
    # RULE 3 — NO ACTIONABLE COMPLIANT BENCHMARK
    # ========================================================

    elif not benchmark_available:
        primary_action = "REPRICE"
        decision_basis = (
            "No observed disruption-era route satisfies the service constraint with sufficient evidence."
        )

        warnings.append(
            f"No observed non-Direct, non-Held route for {row['Product_Category']} meets the {thresholds.service_threshold * 100:.0f}% DIFOT service constraint with at least {MIN_ACTIONABLE_BENCHMARK_SHIPMENTS} observations."
        )

        if fallback_route and fallback_difot is not None:
            warnings.append(
                f"Conditional fallback only: {fallback_route} at {fallback_difot * 100:.1f}% DIFOT; this route does not satisfy the service constraint."
            )

    # Primary action is already REPRICE; no duplicate secondary action.

    # ========================================================
    # RULE 4 — ACTIONABLE BENCHMARK EXISTS
    # ========================================================

    else:
        benchmark_is_current = _same_route(
            current_route,
            str(benchmark_route),
        )

        current_route_compliant = bool(
            current_route_observed_difot is not None
            and current_route_observed_difot
            >= (
                thresholds.service_threshold
                - NUMERIC_TOLERANCE
            )
        )

        current_route_is_premium = (
            _route_is_premium(
                current_route
            )
        )

        benchmark_service_advantage_for_current = (
            (
                current_route_observed_difot
                - float(benchmark_difot)
            )
            if (
                current_route_observed_difot
                is not None
                and benchmark_difot is not None
            )
            else 0.0
        )

        # ----------------------------------------------------
        # 4A. Current route is already the actionable benchmark.
        # ----------------------------------------------------

        if benchmark_is_current:
            if current_margin < 0:
                primary_action = "REPRICE"
                decision_basis = (
                    "Current route is the cheapest observed compliant disruption-era benchmark, but the shipment remains loss-making."
                )
                secondary_action = None
            else:
                primary_action = "MONITOR / RETAIN"
                decision_basis = (
                    "Current route is already the cheapest observed compliant disruption-era benchmark."
                )

            if benchmark_borderline:
                warnings.append(
                    "The current benchmark route sits at a borderline evidence threshold under leave-one-out sensitivity."
                )

        # ----------------------------------------------------
        # 4B. Premium route has demonstrated service advantage.
        # ----------------------------------------------------

        elif (
            current_route_is_premium
            and current_route_compliant
            and benchmark_service_advantage_for_current
            > NUMERIC_TOLERANCE
        ):
            primary_action = "SELECTIVE PROTECT"
            decision_basis = (
                "Premium route retained because its observed route-level service performance exceeds the cheaper compliant benchmark."
            )

            if current_margin < 0:
                secondary_action = "REPRICE"

            warnings.append(
                "Premium mode is not protected automatically; this recommendation requires a documented observed service advantage over the cheaper compliant benchmark."
            )

        # ----------------------------------------------------
        # 4C. Current route fails service constraint.
        # ----------------------------------------------------

        elif not current_route_compliant:
            primary_action = "REROUTE"
            decision_basis = (
                "Current Product × Route service performance is below the minimum service constraint and a compliant disruption-era benchmark exists."
            )

            if counterfactual_margin is not None and counterfactual_margin < 0:
                secondary_action = "REPRICE"

        # ----------------------------------------------------
        # 4D. Current route is compliant but economically inferior.
        # ----------------------------------------------------

        elif (
            cost_gap_pct is not None
            and cost_gap_pct > NUMERIC_TOLERANCE
        ):
            primary_action = "REROUTE"
            decision_basis = (
                "Current route is service-compliant but has a higher observed shipment cost/ton than the cheapest compliant disruption-era benchmark."
            )

            if counterfactual_margin is not None and counterfactual_margin < 0:
                secondary_action = "REPRICE"

        # ----------------------------------------------------
        # 4E. Current route has a service advantage but is not
        # premium mode. Do not invent a percentage trigger.
        # ----------------------------------------------------

        elif (
            benchmark_service_advantage_for_current
            > NUMERIC_TOLERANCE
        ):
            primary_action = "MONITOR / RETAIN"
            decision_basis = (
                "Current route is service-compliant and shows higher observed route-level DIFOT than the cheaper benchmark; automatic rerouting would trade away demonstrated service performance."
            )

            if current_margin < 0:
                secondary_action = "REPRICE"

        # ----------------------------------------------------
        # 4F. No clear economic advantage for intervention.
        # ----------------------------------------------------

        else:
            primary_action = "MONITOR / RETAIN"
            decision_basis = (
                "Current route is service-compliant and the observed shipment economics do not establish a clear reason for automatic rerouting."
            )

            if current_margin < 0:
                secondary_action = "REPRICE"

    # ========================================================
    # COMMON WARNINGS / EVIDENCE CONTEXT
    # ========================================================

    if benchmark_available:
        warnings.append(
            f"Actionable benchmark: {benchmark_route} ({benchmark_difot * 100:.1f}% observed DIFOT; n={benchmark_n})."
        )

        if benchmark_evidence_level:
            warnings.append(
                f"Benchmark evidence level: {benchmark_evidence_level}."
            )

        if benchmark_borderline:
            warnings.append(
                "Benchmark evidence is borderline at the 80% service boundary under leave-one-out sensitivity."
            )

        if (
            benchmark_wilson_lower_pct is not None
            and benchmark_wilson_upper_pct is not None
        ):
            warnings.append(
                f"Benchmark DIFOT Wilson 95% CI: {benchmark_wilson_lower_pct:.1f}%–{benchmark_wilson_upper_pct:.1f}%."
            )

    if (
        current_route_observed_difot is not None
        and current_route_observed_difot
        < (
            thresholds.service_threshold
            - NUMERIC_TOLERANCE
        )
        and current_route != "Held"
        and current_route != "Direct"
    ):
        warnings.append(
            f"Current Product × Route observed DIFOT ({current_route_observed_difot * 100:.1f}%) is below the {thresholds.service_threshold * 100:.0f}% service constraint."
        )

    if (
        counterfactual_margin is not None
        and counterfactual_margin < 0
        and benchmark_available
    ):
        warnings.append(
            "The compliant benchmark remains loss-making; operational rerouting does not eliminate the underlying commercial loss."
        )

    if (
        primary_action == "REPRICE"
        and fallback_route
        and fallback_difot is not None
        and not benchmark_available
    ):
        warnings.append(
            f"Commercial recovery should be evaluated even if {fallback_route} is used as a conditional operational fallback."
        )

    explanation = _build_reasoning(
        row,
        cf,
        primary_action,
        thresholds,
        current_route_observed_difot,
        cost_gap_pct,
        benchmark_available,
    )

    # ========================================================
    # RETURN
    # ========================================================

    return ShipmentDecision(
        shipment_id=str(
            shipment_id
        ),
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

        # Backward-compatible actionable benchmark aliases.
        best_service_compliant_route=(
            str(benchmark_route)
            if benchmark_available
            else None
        ),
        best_service_compliant_cost_per_ton=(
            benchmark_cost
            if benchmark_available
            else None
        ),
        best_service_compliant_difot_pct=(
            benchmark_difot * 100.0
            if (
                benchmark_available
                and benchmark_difot is not None
            )
            else None
        ),

        actionable_benchmark_route=(
            str(benchmark_route)
            if benchmark_available
            else None
        ),
        actionable_benchmark_cost_per_ton=(
            benchmark_cost
        ),
        actionable_benchmark_difot_pct=(
            benchmark_difot * 100.0
            if benchmark_difot is not None
            else None
        ),
        actionable_benchmark_n=(
            benchmark_n
        ),
        actionable_benchmark_evidence_level=(
            benchmark_evidence_level
        ),
        actionable_benchmark_borderline=(
            benchmark_borderline
        ),
        actionable_benchmark_wilson_lower_pct=(
            benchmark_wilson_lower_pct
        ),
        actionable_benchmark_wilson_upper_pct=(
            benchmark_wilson_upper_pct
        ),
        actionable_benchmark_loo_min_difot_pct=(
            benchmark_loo_min_pct
        ),
        actionable_benchmark_loo_max_difot_pct=(
            benchmark_loo_max_pct
        ),

        historical_benchmark_route=(
            str(historical_route)
            if pd.notna(historical_route)
            else None
        ),
        historical_benchmark_cost_per_ton=(
            historical_cost
        ),
        historical_benchmark_difot_pct=(
            historical_difot * 100.0
            if historical_difot is not None
            else None
        ),

        conditional_fallback_route=(
            str(fallback_route)
            if pd.notna(fallback_route)
            else None
        ),
        conditional_fallback_cost_per_ton=(
            fallback_cost
        ),
        conditional_fallback_difot_pct=(
            fallback_difot * 100.0
            if fallback_difot is not None
            else None
        ),
        conditional_fallback_n=(
            fallback_n
        ),

        current_route_observed_difot_pct=(
            current_route_observed_difot * 100.0
            if current_route_observed_difot is not None
            else None
        ),
        current_route_evidence_level=(
            current_route_evidence_level
        ),
        current_route_wilson_lower_pct=(
            current_route_wilson_lower_pct
        ),
        current_route_wilson_upper_pct=(
            current_route_wilson_upper_pct
        ),

        counterfactual_cost=(
            counterfactual_cost
        ),
        counterfactual_margin=(
            counterfactual_margin
        ),
        cost_gap=cost_gap,
        cost_gap_pct=cost_gap_pct,
        avoidable_exposure=(
            avoidable_exposure
        ),
        residual_exposure=(
            residual_exposure
        ),
        benchmark_available=(
            benchmark_available
        ),
        decision_basis=decision_basis,
    )


def decision_for_row(
    df: pd.DataFrame,
    row: pd.Series,
    thresholds: DecisionThresholds | None = None,
) -> ShipmentDecision:
    """Convenience wrapper when a page already has a selected row."""
    return assess_shipment(
        df,
        str(row["Shipment_ID"]),
        thresholds=thresholds,
    )
