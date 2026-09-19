from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .metrics import (
    DEFAULT_SERVICE_THRESHOLD,
    MIN_ACTIONABLE_BENCHMARK_SHIPMENTS,
    counterfactual_metrics,
    portfolio_metrics,
)


@dataclass(frozen=True)
class ScenarioInputs:
    """User-controlled scenario levers.

    Route interventions:
    - pipeline_to_cape: explicit Pipeline → Cape scenario.
    - pipeline_to_benchmark: move Pipeline shipments to the same
      disruption-era actionable benchmark used by metrics.py and
      decision_rules.py.

    Important:
    rerouted revenue is modeled as EXPECTED revenue:
        E[Revenue] = Contracted Revenue × observed route DIFOT
    It is not presented as a deterministic shipment-level outcome.
    """

    freight_adjustment_pct: float = 0.0
    fuel_adjustment_pct: float = 0.0
    insurance_adjustment_pct: float = 0.0
    penalty_adjustment_pct: float = 0.0
    customer_pass_through_pct: float = 0.0
    held_release_days_earlier: int = 0
    pipeline_to_cape: bool = False
    pipeline_to_benchmark: bool = False
    # Optional single-shipment override used by Page 3 for interactive route simulation.
    # This does not alter the portfolio intervention levers used by Page 4.
    shipment_id: Optional[str] = None
    shipment_route: Optional[str] = None
    service_threshold: float = DEFAULT_SERVICE_THRESHOLD


def _clamp_pct(value: float) -> float:
    """Prevent an invalid <= -100% adjustment."""
    return max(-0.999999, float(value))


def _threshold_ok(
    value: pd.Series,
    threshold: float,
) -> pd.Series:
    return value >= (float(threshold) - 1e-10)


def _route_product_rates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Observed median cost-component rates and route DIFOT."""
    work = df.copy()

    weight = work[
        "Cargo_Weight_Tons"
    ].replace(
        0,
        np.nan,
    )

    for col in [
        "Freight_Cost_USD",
        "Fuel_Cost_USD",
        "Insurance_Cost_USD",
        "Penalty_Cost_USD",
    ]:
        work[
            f"{col}_per_ton"
        ] = (
            work[col]
            / weight
        )

    return (
        work.groupby(
            [
                "Product_Category",
                "Route_Canonical",
            ],
            dropna=False,
        )
        .agg(
            Freight_Cost_USD_per_ton=(
                "Freight_Cost_USD_per_ton",
                "median",
            ),
            Fuel_Cost_USD_per_ton=(
                "Fuel_Cost_USD_per_ton",
                "median",
            ),
            Insurance_Cost_USD_per_ton=(
                "Insurance_Cost_USD_per_ton",
                "median",
            ),
            Penalty_Cost_USD_per_ton=(
                "Penalty_Cost_USD_per_ton",
                "median",
            ),
            DIFOT=(
                "DIFOT_Flag",
                "mean",
            ),
            Observations=(
                "Shipment_ID",
                "nunique",
            ),
        )
        .reset_index()
    )


def _actionable_benchmark_table(
    base: pd.DataFrame,
    service_threshold: float,
) -> pd.DataFrame:
    """Return the exact actionable benchmark table used by metrics.py."""
    cf, _ = counterfactual_metrics(
        base,
        service_threshold=service_threshold,
    )

    cols = [
        "Product_Category",
        "Actionable_Benchmark_Route",
        "Actionable_Benchmark_Cost_per_Ton_USD",
        "Actionable_Benchmark_DIFOT",
        "Actionable_Benchmark_N",
    ]

    table = (
        cf[cols]
        .drop_duplicates(
            subset=["Product_Category"]
        )
        .copy()
    )

    table = table[
        table[
            "Actionable_Benchmark_Route"
        ].notna()
        &
        table[
            "Actionable_Benchmark_N"
        ].ge(
            MIN_ACTIONABLE_BENCHMARK_SHIPMENTS
        )
    ].reset_index(drop=True)

    return table


def _apply_route_intervention(
    work: pd.DataFrame,
    base: pd.DataFrame,
    scenario: ScenarioInputs,
) -> pd.DataFrame:
    """Apply explicit Pipeline interventions.

    pipeline_to_benchmark uses the same actionable definition as
    metrics.py and decision_rules.py. Direct and Held cannot be selected.
    """
    work = work.copy()

    work["Scenario_Route"] = (
        work["Route_Canonical"]
    )
    work["Scenario_Route_Source"] = (
        "Current route"
    )

    pipeline_mask = work[
        "Route_Canonical"
    ].eq("Pipeline")

    # Portfolio-level Pipeline interventions.
    if scenario.pipeline_to_cape and pipeline_mask.any():
        work.loc[
            pipeline_mask,
            "Scenario_Route",
        ] = "Cape"

        work.loc[
            pipeline_mask,
            "Scenario_Route_Source",
        ] = "Explicit Pipeline → Cape scenario"

    elif scenario.pipeline_to_benchmark and pipeline_mask.any():
        benchmarks = _actionable_benchmark_table(
            base,
            scenario.service_threshold,
        )

        if benchmarks.empty:
            work.loc[
                pipeline_mask,
                "Scenario_Route_Source",
            ] = (
                "Current route — no actionable benchmark"
            )
        else:
            best_by_product = (
                benchmarks.set_index(
                    "Product_Category"
                )[
                    "Actionable_Benchmark_Route"
                ]
            )

            mapped = (
                work.loc[
                    pipeline_mask,
                    "Product_Category",
                ]
                .map(best_by_product)
            )

            valid = mapped.notna()

            valid_index = mapped.index[valid]
            unresolved_index = mapped.index[~valid]

            work.loc[
                valid_index,
                "Scenario_Route",
            ] = mapped.loc[valid_index]

            work.loc[
                valid_index,
                "Scenario_Route_Source",
            ] = "Actionable disruption-era benchmark"

            if len(unresolved_index):
                work.loc[
                    unresolved_index,
                    "Scenario_Route_Source",
                ] = (
                    "Current route — no compliant actionable benchmark"
                )

    # Single-shipment interactive override. Applied last so Page 3 can
    # simulate any observed Product × Route combination without changing
    # the portfolio-wide scenario semantics used elsewhere.
    if scenario.shipment_id is not None and scenario.shipment_route is not None:
        shipment_mask = work[
            "Shipment_ID"
        ].astype(str).eq(str(scenario.shipment_id))

        if shipment_mask.any():
            selected_product = str(
                work.loc[shipment_mask, "Product_Category"].iloc[0]
            )
            selected_route = str(scenario.shipment_route)

            valid_route = (
                base["Product_Category"].astype(str).eq(selected_product)
                & base["Route_Canonical"].astype(str).eq(selected_route)
            ).any()

            if not valid_route:
                raise ValueError(
                    f"No observed Product × Route combination exists for "
                    f"{selected_product} → {selected_route}."
                )

            work.loc[
                shipment_mask,
                "Scenario_Route",
            ] = selected_route

            work.loc[
                shipment_mask,
                "Scenario_Route_Source",
            ] = "Interactive shipment route override"

    return work


def _apply_route_economics(
    work: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    """Apply observed median Product × Route cost components for changed routes."""
    work = work.copy()

    rates = _route_product_rates(
        base
    )

    for col in [
        "Freight_Cost_USD",
        "Fuel_Cost_USD",
        "Insurance_Cost_USD",
        "Penalty_Cost_USD",
    ]:
        work[
            f"Scenario_{col}"
        ] = work[col]

    changed = work[
        "Scenario_Route"
    ].ne(
        work[
            "Route_Canonical"
        ]
    )

    if not changed.any():
        work[
            "Scenario_Observed_DIFOT"
        ] = np.nan
        work[
            "Scenario_Route_Observations"
        ] = np.nan

        return work

    # Do NOT create Scenario_Observed_DIFOT before merging, because
    # the route-rate table supplies the same field name.
    merge_rates = rates.rename(
        columns={
            "Route_Canonical":
                "Scenario_Route",

            "Freight_Cost_USD_per_ton":
                "Scenario_Freight_Cost_USD_per_ton",

            "Fuel_Cost_USD_per_ton":
                "Scenario_Fuel_Cost_USD_per_ton",

            "Insurance_Cost_USD_per_ton":
                "Scenario_Insurance_Cost_USD_per_ton",

            "Penalty_Cost_USD_per_ton":
                "Scenario_Penalty_Cost_USD_per_ton",

            "DIFOT":
                "Scenario_Observed_DIFOT",

            "Observations":
                "Scenario_Route_Observations",
        }
    )

    work = work.merge(
        merge_rates,
        on=[
            "Product_Category",
            "Scenario_Route",
        ],
        how="left",
    )

    weight = work[
        "Cargo_Weight_Tons"
    ]

    work.loc[
        changed,
        "Scenario_Freight_Cost_USD",
    ] = (
        work.loc[
            changed,
            "Scenario_Freight_Cost_USD_per_ton",
        ]
        * weight.loc[
            changed
        ]
    )

    work.loc[
        changed,
        "Scenario_Fuel_Cost_USD",
    ] = (
        work.loc[
            changed,
            "Scenario_Fuel_Cost_USD_per_ton",
        ]
        * weight.loc[
            changed
        ]
    )

    work.loc[
        changed,
        "Scenario_Insurance_Cost_USD",
    ] = (
        work.loc[
            changed,
            "Scenario_Insurance_Cost_USD_per_ton",
        ]
        * weight.loc[
            changed
        ]
    )

    work.loc[
        changed,
        "Scenario_Penalty_Cost_USD",
    ] = (
        work.loc[
            changed,
            "Scenario_Penalty_Cost_USD_per_ton",
        ]
        * weight.loc[
            changed
        ]
    )

    return work


def _expected_value_view(
    scenario_work: pd.DataFrame,
) -> pd.DataFrame:
    """Create a local view for expected-value counterfactual calculations."""
    view = scenario_work.copy(
        deep=True
    )

    view[
        "Revenue_Recognized_USD"
    ] = view[
        "Scenario_Expected_Revenue_Recognized_USD"
    ]

    view[
        "Total_Cost_to_Serve_USD"
    ] = view[
        "Scenario_Expected_Total_Cost_USD"
    ]

    view[
        "Gross_Margin_USD"
    ] = view[
        "Scenario_Expected_Gross_Margin_USD"
    ]

    view[
        "Cost_per_Ton_USD"
    ] = view[
        "Scenario_Expected_Cost_per_Ton_USD"
    ]

    view[
        "Revenue_per_Ton_USD"
    ] = view[
        "Scenario_Expected_Revenue_per_Ton_USD"
    ]

    return view


def _scenario_exposure_against_baseline_benchmark(
    scenario_work: pd.DataFrame,
    baseline_cf: pd.DataFrame,
) -> tuple[float, float]:
    """Compute scenario avoidable/residual exposure using the BASELINE
    actionable benchmark, not a re-estimated benchmark after the scenario.

    This preserves the causal question:
    'How does the scenario change economic exposure relative to the
    same observed benchmark used to build the original decision?'
    """
    benchmark = (
        baseline_cf[
            [
                "Product_Category",
                "Actionable_Benchmark_Route",
                "Actionable_Benchmark_Cost_per_Ton_USD",
            ]
        ]
        .drop_duplicates(
            subset=["Product_Category"]
        )
        .copy()
    )

    work = scenario_work.merge(
        benchmark,
        on="Product_Category",
        how="left",
    )

    available = (
        work[
            "Actionable_Benchmark_Cost_per_Ton_USD"
        ].notna()
    )

    benchmark_cost = (
        work[
            "Actionable_Benchmark_Cost_per_Ton_USD"
        ]
        * work[
            "Cargo_Weight_Tons"
        ]
    )

    benchmark_margin = (
        work[
            "Scenario_Expected_Contracted_Revenue_USD"
        ]
        - benchmark_cost
    )

    actual_scenario_margin = work[
        "Scenario_Expected_Gross_Margin_USD"
    ]

    avoidable = np.where(
        available,
        (
            benchmark_margin
            - actual_scenario_margin
        ).clip(
            lower=0
        ),
        0.0,
    )

    residual = np.where(
        available,
        (
            -benchmark_margin
        ).clip(
            lower=0
        ),
        0.0,
    )

    return (
        float(
            np.sum(avoidable)
        ),
        float(
            np.sum(residual)
        ),
    )


def simulate_scenario(
    df: pd.DataFrame,
    inputs: ScenarioInputs,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Run a transparent expected-value scenario.

    Methodological guarantees
    -------------------------
    1. Actionable benchmark = non-Direct, non-Held, observed DIFOT >=
       service threshold, at least five observations, lowest observed
       median cost/ton.
    2. Direct is never selected by pipeline_to_benchmark.
    3. Rerouted revenue is explicitly EXPECTED revenue:
           E[Revenue] = Contracted Revenue × observed route DIFOT.
    4. Source financial columns are never mutated in-place.
    5. Scenario values live in explicit Scenario_* columns.
    6. Held-release uses the documented 30-day planning-window assumption.
    7. Scenario exposure is evaluated against the same BASELINE actionable
       benchmark used by the decision engine; the benchmark is not
       re-estimated from the simulated scenario.
    """
    if not (
        0 < inputs.service_threshold <= 1
    ):
        raise ValueError(
            "service_threshold must be in the open interval (0, 1]."
        )

    freight_adj = _clamp_pct(
        inputs.freight_adjustment_pct
    )
    fuel_adj = _clamp_pct(
        inputs.fuel_adjustment_pct
    )
    insurance_adj = _clamp_pct(
        inputs.insurance_adjustment_pct
    )
    penalty_adj = _clamp_pct(
        inputs.penalty_adjustment_pct
    )
    pass_through_adj = _clamp_pct(
        inputs.customer_pass_through_pct
    )

    base = df.copy(
        deep=True
    )

    release_days = max(
        int(
            inputs.held_release_days_earlier
        ),
        0,
    )

    # --------------------------------------------------------
    # Exact control case
    # --------------------------------------------------------

    no_change = (
        freight_adj == 0.0
        and fuel_adj == 0.0
        and insurance_adj == 0.0
        and penalty_adj == 0.0
        and pass_through_adj == 0.0
        and release_days == 0
        and not inputs.pipeline_to_cape
        and not inputs.pipeline_to_benchmark
        and inputs.shipment_id is None
        and inputs.shipment_route is None
    )

    if no_change:
        baseline = portfolio_metrics(
            base
        )

        baseline_cf, _ = (
            counterfactual_metrics(
                base,
                inputs.service_threshold,
            )
        )

        baseline_avoidable = float(
            baseline_cf[
                "Avoidable_Exposure_USD"
            ].sum(
                skipna=True
            )
        )

        baseline_residual = float(
            baseline_cf[
                "Residual_Exposure_USD"
            ].sum(
                skipna=True
            )
        )

        baseline_held = float(
            base.loc[
                base["Is_Held"],
                "Contracted_Freight_Revenue_USD",
            ].sum()
        )

        control = base.copy(
            deep=True
        )

        control[
            "Scenario_Route"
        ] = control[
            "Route_Canonical"
        ]

        control[
            "Scenario_Route_Source"
        ] = "Current route"

        for col in [
            "Freight_Cost_USD",
            "Fuel_Cost_USD",
            "Insurance_Cost_USD",
            "Penalty_Cost_USD",
        ]:
            control[
                f"Scenario_{col}"
            ] = control[col]

        control[
            "Scenario_Observed_DIFOT"
        ] = np.nan

        control[
            "Scenario_Route_Observations"
        ] = np.nan

        control[
            "Scenario_Expected_Contracted_Revenue_USD"
        ] = control[
            "Contracted_Freight_Revenue_USD"
        ]

        control[
            "Scenario_Expected_Revenue_Recognized_USD"
        ] = control[
            "Revenue_Recognized_USD"
        ]

        control[
            "Scenario_Expected_Total_Cost_USD"
        ] = control[
            "Total_Cost_to_Serve_USD"
        ]

        control[
            "Scenario_Expected_Gross_Margin_USD"
        ] = control[
            "Gross_Margin_USD"
        ]

        control[
            "Scenario_Expected_Gross_Margin_Pct"
        ] = np.divide(
            control[
                "Scenario_Expected_Gross_Margin_USD"
            ],
            control[
                "Scenario_Expected_Contracted_Revenue_USD"
            ].replace(
                0,
                np.nan,
            ),
        )

        control[
            "Scenario_Expected_Cost_per_Ton_USD"
        ] = np.divide(
            control[
                "Scenario_Expected_Total_Cost_USD"
            ],
            control[
                "Cargo_Weight_Tons"
            ].replace(
                0,
                np.nan,
            ),
        )

        control[
            "Scenario_Expected_Revenue_per_Ton_USD"
        ] = np.divide(
            control[
                "Scenario_Expected_Contracted_Revenue_USD"
            ],
            control[
                "Cargo_Weight_Tons"
            ].replace(
                0,
                np.nan,
            ),
        )

        control[
            "Scenario_Service_Compliant"
        ] = np.nan

        control[
            "Scenario_Revenue_Recognized_USD"
        ] = control[
            "Scenario_Expected_Revenue_Recognized_USD"
        ]

        control[
            "Scenario_Total_Cost_to_Serve_USD"
        ] = control[
            "Scenario_Expected_Total_Cost_USD"
        ]

        control[
            "Scenario_Gross_Margin_USD"
        ] = control[
            "Scenario_Expected_Gross_Margin_USD"
        ]

        control[
            "Scenario_Gross_Margin_Pct"
        ] = control[
            "Scenario_Expected_Gross_Margin_Pct"
        ]

        summary = {
            "baseline_gross_margin":
                baseline[
                    "gross_margin"
                ],
            "scenario_gross_margin":
                baseline[
                    "gross_margin"
                ],
            "baseline_margin_pct":
                baseline[
                    "gross_margin_pct_of_contracted"
                ],
            "scenario_margin_pct":
                baseline[
                    "gross_margin_pct_of_contracted"
                ],
            "baseline_total_cost":
                baseline[
                    "total_cost"
                ],
            "scenario_total_cost":
                baseline[
                    "total_cost"
                ],
            "baseline_expected_revenue_recognized":
                baseline[
                    "recognized_revenue"
                ],
            "scenario_expected_revenue_recognized":
                baseline[
                    "recognized_revenue"
                ],
            "baseline_held_exposure":
                baseline_held,
            "scenario_held_exposure":
                baseline_held,
            "baseline_avoidable_exposure":
                baseline_avoidable,
            "scenario_avoidable_exposure":
                baseline_avoidable,
            "baseline_residual_exposure":
                baseline_residual,
            "scenario_residual_exposure":
                baseline_residual,
            "gross_margin_change":
                0.0,
            "total_cost_change":
                0.0,
            "margin_pct_change":
                0.0,
            "held_exposure_change":
                0.0,
            "avoidable_exposure_change":
                0.0,
            "residual_exposure_change":
                0.0,
            "expected_value_method":
                "For rerouted shipments, expected recognised revenue = contracted revenue × observed route DIFOT.",
            "service_threshold":
                inputs.service_threshold,
            "route_intervention_shipments":
                0,
            "benchmark_excludes_direct":
                True,
            "benchmark_min_observations":
                MIN_ACTIONABLE_BENCHMARK_SHIPMENTS,
        }

        return (
            control,
            summary,
        )

    # --------------------------------------------------------
    # Route assignment + route economics
    # --------------------------------------------------------

    work = _apply_route_intervention(
        base,
        base,
        inputs,
    )

    work = _apply_route_economics(
        work,
        base,
    )

    changed = work[
        "Scenario_Route"
    ].ne(
        work[
            "Route_Canonical"
        ]
    )

    # --------------------------------------------------------
    # Cost adjustments
    # --------------------------------------------------------

    work[
        "Scenario_Freight_Cost_USD"
    ] *= (
        1.0 + freight_adj
    )

    work[
        "Scenario_Fuel_Cost_USD"
    ] *= (
        1.0 + fuel_adj
    )

    work[
        "Scenario_Insurance_Cost_USD"
    ] *= (
        1.0 + insurance_adj
    )

    work[
        "Scenario_Penalty_Cost_USD"
    ] *= (
        1.0 + penalty_adj
    )

    # --------------------------------------------------------
    # Customer pass-through
    # --------------------------------------------------------

    pass_through_factor = (
        1.0 + pass_through_adj
    )

    work[
        "Scenario_Expected_Contracted_Revenue_USD"
    ] = (
        work[
            "Contracted_Freight_Revenue_USD"
        ]
        * pass_through_factor
    )

    # --------------------------------------------------------
    # Expected recognised revenue
    # --------------------------------------------------------

    work[
        "Scenario_Expected_Revenue_Recognized_USD"
    ] = (
        work[
            "Revenue_Recognized_USD"
        ]
        * pass_through_factor
    )

    routed_with_difot = (
        changed
        &
        work[
            "Scenario_Observed_DIFOT"
        ].notna()
    )

    work.loc[
        routed_with_difot,
        "Scenario_Expected_Revenue_Recognized_USD",
    ] = (
        work.loc[
            routed_with_difot,
            "Scenario_Expected_Contracted_Revenue_USD",
        ]
        *
        work.loc[
            routed_with_difot,
            "Scenario_Observed_DIFOT",
        ]
    )

    # --------------------------------------------------------
    # Held release acceleration
    # --------------------------------------------------------

    release_fraction = min(
        release_days / 30.0,
        1.0,
    )

    held_mask = work[
        "Is_Held"
    ]

    if (
        release_fraction > 0
        and held_mask.any()
    ):
        expected_contract = work.loc[
            held_mask,
            "Scenario_Expected_Contracted_Revenue_USD",
        ]

        expected_recognised = work.loc[
            held_mask,
            "Scenario_Expected_Revenue_Recognized_USD",
        ]

        unrecognised = (
            expected_contract
            - expected_recognised
        ).clip(
            lower=0
        )

        work.loc[
            held_mask,
            "Scenario_Expected_Revenue_Recognized_USD",
        ] += (
            unrecognised
            * release_fraction
        )

        work.loc[
            held_mask,
            "Scenario_Penalty_Cost_USD",
        ] *= (
            1.0
            - release_fraction
        )

        work.loc[
            held_mask,
            "Scenario_Insurance_Cost_USD",
        ] *= (
            1.0
            - release_fraction
        )

    # --------------------------------------------------------
    # Expected scenario financials
    # --------------------------------------------------------

    work[
        "Scenario_Expected_Total_Cost_USD"
    ] = (
        work[
            "Scenario_Freight_Cost_USD"
        ]
        + work[
            "Scenario_Fuel_Cost_USD"
        ]
        + work[
            "Scenario_Insurance_Cost_USD"
        ]
        + work[
            "Scenario_Penalty_Cost_USD"
        ]
    )

    work[
        "Scenario_Expected_Gross_Margin_USD"
    ] = (
        work[
            "Scenario_Expected_Revenue_Recognized_USD"
        ]
        -
        work[
            "Scenario_Expected_Total_Cost_USD"
        ]
    )

    work[
        "Scenario_Expected_Gross_Margin_Pct"
    ] = np.divide(
        work[
            "Scenario_Expected_Gross_Margin_USD"
        ],
        work[
            "Scenario_Expected_Contracted_Revenue_USD"
        ].replace(
            0,
            np.nan,
        ),
    )

    work[
        "Scenario_Expected_Cost_per_Ton_USD"
    ] = np.divide(
        work[
            "Scenario_Expected_Total_Cost_USD"
        ],
        work[
            "Cargo_Weight_Tons"
        ].replace(
            0,
            np.nan,
        ),
    )

    work[
        "Scenario_Expected_Revenue_per_Ton_USD"
    ] = np.divide(
        work[
            "Scenario_Expected_Contracted_Revenue_USD"
        ],
        work[
            "Cargo_Weight_Tons"
        ].replace(
            0,
            np.nan,
        ),
    )

    # --------------------------------------------------------
    # Scenario service evidence
    # --------------------------------------------------------

    current_route_rates = _route_product_rates(
        base
    )[
        [
            "Product_Category",
            "Route_Canonical",
            "DIFOT",
            "Observations",
        ]
    ].rename(
        columns={
            "Route_Canonical":
                "_Scenario_Current_Route",
            "DIFOT":
                "_Scenario_Current_DIFOT",
            "Observations":
                "_Scenario_Current_Route_N",
        }
    )

    work = work.merge(
        current_route_rates,
        left_on=[
            "Product_Category",
            "Route_Canonical",
        ],
        right_on=[
            "Product_Category",
            "_Scenario_Current_Route",
        ],
        how="left",
    )

    unchanged = ~changed

    if "Scenario_Observed_DIFOT" not in work.columns:
        work[
            "Scenario_Observed_DIFOT"
        ] = np.nan

    if "Scenario_Route_Observations" not in work.columns:
        work[
            "Scenario_Route_Observations"
        ] = np.nan

    work.loc[
        unchanged,
        "Scenario_Observed_DIFOT",
    ] = work.loc[
        unchanged,
        "_Scenario_Current_DIFOT",
    ]

    work.loc[
        unchanged,
        "Scenario_Route_Observations",
    ] = work.loc[
        unchanged,
        "_Scenario_Current_Route_N",
    ]

    work[
        "Scenario_Service_Compliant"
    ] = _threshold_ok(
        work[
            "Scenario_Observed_DIFOT"
        ],
        inputs.service_threshold,
    )

    work.drop(
        columns=[
            "_Scenario_Current_Route",
            "_Scenario_Current_DIFOT",
            "_Scenario_Current_Route_N",
            "Scenario_Freight_Cost_USD_per_ton",
            "Scenario_Fuel_Cost_USD_per_ton",
            "Scenario_Insurance_Cost_USD_per_ton",
            "Scenario_Penalty_Cost_USD_per_ton",
        ],
        inplace=True,
        errors="ignore",
    )

    # --------------------------------------------------------
    # Backward-compatible aliases.
    # They explicitly represent EXPECTED scenario outcomes.
    # --------------------------------------------------------

    work[
        "Scenario_Revenue_Recognized_USD"
    ] = work[
        "Scenario_Expected_Revenue_Recognized_USD"
    ]

    work[
        "Scenario_Total_Cost_to_Serve_USD"
    ] = work[
        "Scenario_Expected_Total_Cost_USD"
    ]

    work[
        "Scenario_Gross_Margin_USD"
    ] = work[
        "Scenario_Expected_Gross_Margin_USD"
    ]

    work[
        "Scenario_Gross_Margin_Pct"
    ] = work[
        "Scenario_Expected_Gross_Margin_Pct"
    ]

    # --------------------------------------------------------
    # Portfolio summary
    # --------------------------------------------------------

    baseline = portfolio_metrics(
        base
    )

    baseline_cf, _ = (
        counterfactual_metrics(
            base,
            inputs.service_threshold,
        )
    )

    baseline_avoidable = float(
        baseline_cf[
            "Avoidable_Exposure_USD"
        ].sum(
            skipna=True
        )
    )

    baseline_residual = float(
        baseline_cf[
            "Residual_Exposure_USD"
        ].sum(
            skipna=True
        )
    )

    scenario_gross_margin = float(
        work[
            "Scenario_Expected_Gross_Margin_USD"
        ].sum()
    )

    scenario_total_cost = float(
        work[
            "Scenario_Expected_Total_Cost_USD"
        ].sum()
    )

    scenario_expected_revenue = float(
        work[
            "Scenario_Expected_Revenue_Recognized_USD"
        ].sum()
    )

    scenario_contracted_revenue = float(
        work[
            "Scenario_Expected_Contracted_Revenue_USD"
        ].sum()
    )

    scenario_margin_pct = (
        scenario_gross_margin
        / scenario_contracted_revenue
        if scenario_contracted_revenue
        else 0.0
    )

    baseline_held_revenue = float(
        base.loc[
            base["Is_Held"],
            "Contracted_Freight_Revenue_USD",
        ].sum()
    )

    scenario_held_revenue = (
        baseline_held_revenue
        * (
            1.0
            - release_fraction
        )
    )

    scenario_avoidable, scenario_residual = (
        _scenario_exposure_against_baseline_benchmark(
            work,
            baseline_cf,
        )
    )

    summary = {
        "baseline_gross_margin":
            baseline["gross_margin"],

        "scenario_gross_margin":
            scenario_gross_margin,

        "baseline_margin_pct":
            baseline[
                "gross_margin_pct_of_contracted"
            ],

        "scenario_margin_pct":
            scenario_margin_pct,

        "baseline_total_cost":
            baseline["total_cost"],

        "scenario_total_cost":
            scenario_total_cost,

        "baseline_expected_revenue_recognized":
            baseline[
                "recognized_revenue"
            ],

        "scenario_expected_revenue_recognized":
            scenario_expected_revenue,

        "baseline_held_exposure":
            baseline_held_revenue,

        "scenario_held_exposure":
            scenario_held_revenue,

        "baseline_avoidable_exposure":
            baseline_avoidable,

        "scenario_avoidable_exposure":
            scenario_avoidable,

        "baseline_residual_exposure":
            baseline_residual,

        "scenario_residual_exposure":
            scenario_residual,

        "gross_margin_change":
            (
                scenario_gross_margin
                - baseline[
                    "gross_margin"
                ]
            ),

        "total_cost_change":
            (
                scenario_total_cost
                - baseline[
                    "total_cost"
                ]
            ),

        "margin_pct_change":
            (
                scenario_margin_pct
                - baseline[
                    "gross_margin_pct_of_contracted"
                ]
            ),

        "held_exposure_change":
            (
                scenario_held_revenue
                - baseline_held_revenue
            ),

        "avoidable_exposure_change":
            (
                scenario_avoidable
                - baseline_avoidable
            ),

        "residual_exposure_change":
            (
                scenario_residual
                - baseline_residual
            ),

        "expected_value_method":
            "For rerouted shipments, E[Revenue] = contracted revenue × observed route DIFOT. Route costs use observed median Product × Route component rates. Held-release effects use a transparent 30-day planning-window assumption.",

        "service_threshold":
            inputs.service_threshold,

        "actionable_benchmark_rule":
            "Non-Direct, non-Held; observed DIFOT >= service threshold; at least five observations; lowest observed median cost/ton.",

        "benchmark_excludes_direct":
            True,

        "benchmark_min_observations":
            MIN_ACTIONABLE_BENCHMARK_SHIPMENTS,

        "route_intervention_shipments":
            int(changed.sum()),
    }

    return (
        work,
        summary,
    )
