from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    decision_card,
    fmt_money,
    inject_css,
    info_card,
    metric_card,
    page_header,
    section_header,
    warning_card,
)
from src.data_loader import load_data
from src.decision_rules import (
    DecisionThresholds,
    assess_shipment,
)
from src.metrics import (
    DEFAULT_SERVICE_THRESHOLD,
    product_route_metrics,
)
from src.scenario_engine import (
    ScenarioInputs,
    simulate_scenario,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Shipment Decision Simulator | Hormuz War Room",
    page_icon="⚓",
    layout="wide",
)

inject_css()
df = load_data()

SERVICE_THRESHOLD = DEFAULT_SERVICE_THRESHOLD


# =============================================================================
# HEADER
# =============================================================================

page_header(
    "SHIPMENT DECISION SIMULATOR",
    "Test the management decision for one shipment — separate the historical "
    "benchmark from the actionable disruption-era benchmark, inspect the route "
    "evidence, and simulate the expected economic consequence.",
)


# =============================================================================
# 1. SHIPMENT SELECTION
# =============================================================================

filters = st.columns(3)

with filters[0]:
    customer_filter = st.selectbox(
        "Filter by customer",
        [
            "All customers"
        ]
        + sorted(
            df["Customer_Name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    )

with filters[1]:
    product_filter = st.selectbox(
        "Filter by product",
        [
            "All products"
        ]
        + sorted(
            df["Product_Category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    )

with filters[2]:
    route_filter = st.selectbox(
        "Filter by current route",
        [
            "All routes"
        ]
        + sorted(
            df["Route_Canonical"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    )

work = df.copy()

if customer_filter != "All customers":
    work = work[
        work["Customer_Name"].eq(customer_filter)
    ]

if product_filter != "All products":
    work = work[
        work["Product_Category"].eq(product_filter)
    ]

if route_filter != "All routes":
    work = work[
        work["Route_Canonical"].eq(route_filter)
    ]

shipment_ids = (
    work["Shipment_ID"]
    .astype(str)
    .tolist()
)

if not shipment_ids:
    st.warning(
        "No shipments match the selected filters."
    )
    st.stop()

shipment_id = st.selectbox(
    "Shipment ID",
    shipment_ids,
)

row = df[
    df["Shipment_ID"]
    .astype(str)
    .eq(str(shipment_id))
].iloc[0]

decision = assess_shipment(
    df,
    shipment_id,
    thresholds=DecisionThresholds(
        service_threshold=SERVICE_THRESHOLD
    ),
)

product = str(
    row["Product_Category"]
)

current_route = str(
    row["Route_Canonical"]
)


# =============================================================================
# 2. ROUTE EVIDENCE TABLE
# =============================================================================

route_table = product_route_metrics(
    df,
    service_threshold=SERVICE_THRESHOLD,
)

route_table = route_table[
    route_table[
        "Product_Category"
    ].eq(product)
].copy()

route_table = route_table.sort_values(
    "Median_Cost_per_Ton_USD"
).reset_index(drop=True)

route_table["Service_Compliant"] = (
    route_table["DIFOT"]
    >= SERVICE_THRESHOLD - 1e-10
)

route_table["Current"] = (
    route_table["Route_Canonical"]
    .eq(current_route)
)

route_table["Actionable_Eligible"] = (
    ~route_table[
        "Route_Canonical"
    ].isin(
        ["Direct", "Held"]
    )
    &
    route_table[
        "Shipments"
    ].ge(5)
)

actionable_service_routes = route_table[
    route_table["Actionable_Eligible"]
    &
    route_table["Service_Compliant"]
].copy()

historical_service_routes = route_table[
    ~route_table[
        "Route_Canonical"
    ].eq("Held")
    &
    route_table["Service_Compliant"]
].copy()

# The first row after sorting by median cost is the cheapest benchmark.
historical_benchmark = (
    historical_service_routes.iloc[0]
    if not historical_service_routes.empty
    else None
)

actionable_benchmark = (
    actionable_service_routes.iloc[0]
    if not actionable_service_routes.empty
    else None
)


# =============================================================================
# 3. SHIPMENT PROFILE
# =============================================================================

section_header(
    "SHIPMENT PROFILE",
    "The operational and commercial context behind the decision.",
)

profile_cols = st.columns(6)

profile = [
    (
        "Customer",
        str(row["Customer_Name"]),
        str(row["Customer_Region"]),
    ),
    (
        "Product",
        product,
        str(row["Cargo_Type"]),
    ),
    (
        "Cargo Weight",
        f'{row["Cargo_Weight_Tons"]:,.1f} t',
        "shipment weight",
    ),
    (
        "Cargo Value",
        fmt_money(row["Cargo_Value_USD"]),
        "cargo value",
    ),
    (
        "Current Route",
        current_route,
        "observed route",
    ),
    (
        "DIFOT",
        f'{row["DIFOT_Pct"]:.1f}%',
        "actual shipment result",
    ),
]

for c, (label, value, sub) in zip(
    profile_cols,
    profile,
):
    with c:
        metric_card(
            label,
            value,
            sub,
        )

meta = st.columns(5)

meta_values = [
    (
        row["Departure_Date"].strftime(
            "%d %b %Y"
        )
        if pd.notna(
            row["Departure_Date"]
        )
        else "—"
    ),
    (
        f'{row["Planned_Transit_Days"]:.0f} days'
        if pd.notna(
            row["Planned_Transit_Days"]
        )
        else "—"
    ),
    (
        f'{row["Actual_Transit_Days"]:.0f} days'
        if pd.notna(
            row["Actual_Transit_Days"]
        )
        else "Held / unresolved"
    ),
    (
        f'{row["Delay_Days"]:.0f} days'
        if pd.notna(
            row["Delay_Days"]
        )
        else "—"
    ),
    str(row["Cargo_Type"]),
]

for c, label, value in zip(
    meta,
    [
        "Departure",
        "Planned Transit",
        "Actual Transit",
        "Delay",
        "Cargo Type",
    ],
    meta_values,
):
    with c:
        st.caption(label)
        st.write(value)


# =============================================================================
# 4. CURRENT ECONOMICS
# =============================================================================

section_header(
    "CURRENT ECONOMICS",
    "Observed economics of the selected shipment.",
)

econ_cols = st.columns(6)

econ = [
    (
        "Cost / Ton",
        fmt_money(
            row["Cost_per_Ton_USD"],
            2,
        ),
        "current",
    ),
    (
        "Revenue / Ton",
        fmt_money(
            row["Revenue_per_Ton_USD"],
            2,
        ),
        "contracted",
    ),
    (
        "Margin / Ton",
        fmt_money(
            row["Margin_per_Ton_USD"],
            2,
        ),
        "observed",
    ),
    (
        "Total Cost",
        fmt_money(
            row["Total_Cost_to_Serve_USD"]
        ),
        "observed",
    ),
    (
        "Gross Margin",
        fmt_money(
            row["Gross_Margin_USD"]
        ),
        "observed",
    ),
    (
        "Recognised Revenue",
        fmt_money(
            row["Revenue_Recognized_USD"]
        ),
        "observed",
    ),
]

for c, (label, value, sub) in zip(
    econ_cols,
    econ,
):
    with c:
        metric_card(
            label,
            value,
            sub,
        )


# =============================================================================
# 5. TWO-BENCHMARK FRAMEWORK
# =============================================================================

section_header(
    "TWO-BENCHMARK ROUTE LENS",
    "Historical performance and actionable rerouting answer different questions. "
    "Direct may be used as a pre-blockade reference, but it is never treated as "
    "guaranteed post-blockade capacity for an actionable reroute.",
)

bench_cols = st.columns(2)

with bench_cols[0]:
    if historical_benchmark is not None:
        metric_card(
            "Historical / Canonical Benchmark",
            str(
                historical_benchmark[
                    "Route_Canonical"
                ]
            ),
            (
                f'{historical_benchmark["DIFOT_Pct"]:.1f}% DIFOT · '
                f'{fmt_money(historical_benchmark["Median_Cost_per_Ton_USD"], 2)}/ton'
            ),
        )

        if (
            str(
                historical_benchmark[
                    "Route_Canonical"
                ]
            )
            == "Direct"
        ):
            warning_card(
                "Direct is a pre-blockade historical reference here — "
                "not a guaranteed post-blockade capacity commitment."
            )
    else:
        metric_card(
            "Historical / Canonical Benchmark",
            "None observed",
            "No non-held route meets the service threshold.",
        )

with bench_cols[1]:
    if actionable_benchmark is not None:
        metric_card(
            "Actionable Disruption-Era Benchmark",
            str(
                actionable_benchmark[
                    "Route_Canonical"
                ]
            ),
            (
                f'{actionable_benchmark["DIFOT_Pct"]:.1f}% DIFOT · '
                f'{fmt_money(actionable_benchmark["Median_Cost_per_Ton_USD"], 2)}/ton · '
                f'n={int(actionable_benchmark["Shipments"])}'
            ),
        )

        if bool(
            actionable_benchmark.get(
                "Borderline_Service_Evidence",
                False,
            )
        ):
            warning_card(
                "This benchmark meets the observed 80% service boundary, "
                "but leave-one-out sensitivity makes the service evidence "
                "borderline around that threshold."
            )
    else:
        metric_card(
            "Actionable Disruption-Era Benchmark",
            "None observed",
            (
                "No observed non-Direct, non-Held route meets the 80% "
                "service constraint with sufficient evidence."
            ),
        )

        fallback_route = decision.conditional_fallback_route
        fallback_difot = decision.conditional_fallback_difot_pct

        if fallback_route:
            warning_card(
                f"Conditional fallback only: {fallback_route} at "
                f"{fallback_difot:.1f}% DIFOT. It is not service-compliant."
            )


info_card(
    "Service constraint: 80% minimum observed DIFOT. "
    "Actionable benchmark: non-Direct, non-Held, at least 5 observed shipments, "
    "and lowest observed median cost/ton among routes meeting the service constraint."
)


# =============================================================================
# 6. ROUTE LANDSCAPE
# =============================================================================

section_header(
    "ROUTE LANDSCAPE FOR THIS PRODUCT",
    "Each bubble is an observed Product × Route combination. The horizontal line "
    "is the 80% management service constraint; Direct is shown as historical "
    "reference only, while the actionable benchmark excludes Direct and Held.",
)

plot_routes = route_table.copy()

plot_routes["Route_Label"] = (
    plot_routes["Route_Canonical"]
)

plot_routes["Benchmark_Type"] = (
    "Other observed route"
)

plot_routes.loc[
    plot_routes["Current"],
    "Benchmark_Type",
] = "Current route"

if historical_benchmark is not None:
    plot_routes.loc[
        plot_routes[
            "Route_Canonical"
        ].eq(
            str(
                historical_benchmark[
                    "Route_Canonical"
                ]
            )
        ),
        "Benchmark_Type",
    ] = "Historical benchmark"

if actionable_benchmark is not None:
    plot_routes.loc[
        plot_routes[
            "Route_Canonical"
        ].eq(
            str(
                actionable_benchmark[
                    "Route_Canonical"
                ]
            )
        ),
        "Benchmark_Type",
    ] = "Actionable benchmark"

bubble = px.scatter(
    plot_routes,
    x="Median_Cost_per_Ton_USD",
    y="DIFOT_Pct",
    size="Shipments",
    color="Benchmark_Type",
    text="Route_Label",
    hover_data={
        "Route_Label": True,
        "Shipments": True,
        "Median_Cost_per_Ton_USD": ":.2f",
        "Median_Margin_per_Ton_USD": ":.2f",
        "DIFOT_Pct": ":.1f",
        "Evidence_Level": True,
        "Wilson_DIFOT_Lower_Pct": ":.1f",
        "Wilson_DIFOT_Upper_Pct": ":.1f",
        "Borderline_Service_Evidence": True,
        "Benchmark_Type": True,
    },
    height=440,
)

bubble.add_hline(
    y=SERVICE_THRESHOLD * 100,
    line_dash="dash",
    line_width=1.5,
    annotation_text="80% service constraint",
    annotation_position="top left",
)

bubble.update_traces(
    textposition="top center",
    marker=dict(
        line=dict(
            width=1,
            color="#E2E8F0",
        )
    ),
)

bubble.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(
        l=10,
        r=20,
        t=10,
        b=10,
    ),
    xaxis_title="Observed median cost / ton (USD)",
    yaxis_title="Observed DIFOT (%)",
    legend_title="Route lens",
)

st.plotly_chart(
    bubble,
    use_container_width=True,
    config={
        "displayModeBar": False
    },
)


# =============================================================================
# 7. ALTERNATIVE ROUTE TABLE
# =============================================================================

section_header(
    "OBSERVED ALTERNATIVE ROUTES",
    "Audit view of the evidence used by the benchmark and decision engine. "
    "Wilson intervals and leave-one-out sensitivity are shown so small samples "
    "are not treated as exact route reliability.",
)

route_display = route_table[
    [
        "Route_Canonical",
        "Shipments",
        "Median_Cost_per_Ton_USD",
        "Median_Margin_per_Ton_USD",
        "DIFOT_Pct",
        "Wilson_DIFOT_Lower_Pct",
        "Wilson_DIFOT_Upper_Pct",
        "Evidence_Level",
        "Borderline_Service_Evidence",
        "LOO_Min_DIFOT",
        "LOO_Max_DIFOT",
        "Service_Compliant",
        "Actionable_Eligible",
        "Current",
    ]
].copy()

route_display["Service"] = (
    route_display[
        "Service_Compliant"
    ]
    .map(
        lambda x:
            "✓ Compliant"
            if x
            else "Below threshold"
    )
)

route_display["Actionable"] = np.select(
    [
        route_display[
            "Actionable_Eligible"
        ]
        &
        route_display[
            "Service_Compliant"
        ],
        route_display[
            "Actionable_Eligible"
        ]
        &
        ~route_display[
            "Service_Compliant"
        ],
        route_display[
            "Route_Canonical"
        ].isin(
            ["Direct", "Held"]
        ),
    ],
    [
        "✓ Candidate",
        "Below 80%",
        "Reference / excluded",
    ],
    default="Observed",
)

route_display["LOO Range"] = (
    route_display[
        "LOO_Min_DIFOT"
    ] * 100
).round(1).astype(str) + "% – " + (
    route_display[
        "LOO_Max_DIFOT"
    ] * 100
).round(1).astype(str) + "%"

route_display = route_display[
    [
        "Route_Canonical",
        "Shipments",
        "Median_Cost_per_Ton_USD",
        "Median_Margin_per_Ton_USD",
        "DIFOT_Pct",
        "Wilson_DIFOT_Lower_Pct",
        "Wilson_DIFOT_Upper_Pct",
        "Evidence_Level",
        "Borderline_Service_Evidence",
        "LOO Range",
        "Actionable",
        "Current",
    ]
]

route_display.columns = [
    "Route",
    "Shipments",
    "Median Cost / Ton",
    "Median Margin / Ton",
    "DIFOT",
    "Wilson Lower",
    "Wilson Upper",
    "Evidence",
    "Borderline",
    "LOO Range",
    "Actionable Status",
    "Current",
]

st.dataframe(
    route_display.style.format(
        {
            "Median Cost / Ton": "${:,.2f}",
            "Median Margin / Ton": "${:,.2f}",
            "DIFOT": "{:.1f}%",
            "Wilson Lower": "{:.1f}%",
            "Wilson Upper": "{:.1f}%",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# =============================================================================
# 8. CONTEXTUAL DECISION / BENCHMARK EXPOSURE
# =============================================================================

section_header(
    "DECISION EVIDENCE",
    "The recommendation is generated independently from the simulation control. "
    "This section exposes the historical reference, actionable benchmark and "
    "commercial exposure behind the decision.",
)

evidence_cols = st.columns(5)

evidence_items = [
    (
        "Action",
        decision.primary_action,
        (
            decision.decision_basis
            if decision.decision_basis
            else "Shared deterministic decision engine"
        ),
    ),
    (
        "Current Route",
        decision.current_route,
        f'{decision.current_difot_pct:.1f}% shipment DIFOT',
    ),
    (
        "Actionable Benchmark",
        (
            decision.actionable_benchmark_route
            if decision.actionable_benchmark_route
            else "None"
        ),
        (
            f'{decision.actionable_benchmark_difot_pct:.1f}% DIFOT'
            if decision.actionable_benchmark_difot_pct is not None
            else "No compliant route"
        ),
    ),
    (
        "Avoidable Exposure",
        fmt_money(
            decision.avoidable_exposure
        ),
        "under actionable benchmark",
    ),
    (
        "Residual Exposure",
        fmt_money(
            decision.residual_exposure
        ),
        "after benchmark",
    ),
]

for c, (label, value, sub) in zip(
    evidence_cols,
    evidence_items,
):
    with c:
        metric_card(
            label,
            value,
            sub,
        )


# =============================================================================
# 9. SHIPMENT-LEVEL EXPECTED-VALUE SIMULATOR
# =============================================================================

section_header(
    "SIMULATE THE SHIPMENT",
    "Choose an observed route and see the expected shipment-level economic "
    "consequence. For a route move, recognised revenue is an expected value "
    "based on observed route DIFOT — it is not a deterministic guarantee.",
)

available_sim_routes = (
    route_table[
        route_table[
            "Route_Canonical"
        ].ne("Held")
    ][
        "Route_Canonical"
    ]
    .astype(str)
    .tolist()
)

if current_route not in available_sim_routes:
    available_sim_routes = [
        current_route
    ] + available_sim_routes

recommended_default = (
    decision.actionable_benchmark_route
    or decision.conditional_fallback_route
    or decision.historical_benchmark_route
    or current_route
)

if recommended_default not in available_sim_routes:
    recommended_default = (
        available_sim_routes[0]
    )

selected_index = available_sim_routes.index(
    recommended_default
)

sim_route = st.selectbox(
    "Simulate route",
    available_sim_routes,
    index=selected_index,
    key=f"sim_route_{shipment_id}",
)

sim_row = route_table[
    route_table[
        "Route_Canonical"
    ].eq(sim_route)
].iloc[0]

# -------------------------------------------------------------------------
# Use the scenario engine for the actual simulation so this page and
# Page 4 share exactly the same expected-value mathematics.
# -------------------------------------------------------------------------

sim_inputs = ScenarioInputs(
    # Page 3 is a single-shipment simulator, so the selected route must be
    # applied to this shipment regardless of its current route. Portfolio-wide
    # Pipeline levers remain reserved for Page 4.
    shipment_id=str(shipment_id),
    shipment_route=str(sim_route),
    service_threshold=SERVICE_THRESHOLD,
)

scenario_df, scenario_summary = simulate_scenario(
    df,
    sim_inputs,
)

scenario_row = scenario_df[
    scenario_df[
        "Shipment_ID"
    ].astype(str).eq(
        str(shipment_id)
    )
].iloc[0]

weight = float(
    row["Cargo_Weight_Tons"]
)

contracted_revenue = float(
    row["Contracted_Freight_Revenue_USD"]
)

current_cost = float(
    row["Total_Cost_to_Serve_USD"]
)

current_recognised_revenue = float(
    row["Revenue_Recognized_USD"]
)

current_margin = float(
    row["Gross_Margin_USD"]
)

sim_cost_per_ton = float(
    scenario_row[
        "Scenario_Expected_Cost_per_Ton_USD"
    ]
)

sim_cost = float(
    scenario_row[
        "Scenario_Expected_Total_Cost_USD"
    ]
)

sim_difot = (
    float(
        scenario_row[
            "Scenario_Observed_DIFOT"
        ]
    )
    if pd.notna(
        scenario_row[
            "Scenario_Observed_DIFOT"
        ]
    )
    else float(
        sim_row["DIFOT_Pct"]
    ) / 100.0
)

sim_expected_revenue = float(
    scenario_row[
        "Scenario_Expected_Revenue_Recognized_USD"
    ]
)

sim_margin = float(
    scenario_row[
        "Scenario_Expected_Gross_Margin_USD"
    ]
)

sim_margin_per_ton = (
    sim_margin / weight
    if weight
    else np.nan
)

cost_change = (
    sim_cost
    - current_cost
)

expected_revenue_change = (
    sim_expected_revenue
    - current_recognised_revenue
)

margin_change = (
    sim_margin
    - current_margin
)

sim_service_compliant = (
    bool(
        scenario_row[
            "Scenario_Service_Compliant"
        ]
    )
    if pd.notna(
        scenario_row[
            "Scenario_Service_Compliant"
        ]
    )
    else False
)

sim_evidence_n = (
    int(
        scenario_row[
            "Scenario_Route_Observations"
        ]
    )
    if pd.notna(
        scenario_row[
            "Scenario_Route_Observations"
        ]
    )
    else int(
        sim_row["Shipments"]
    )
)


sim_cols = st.columns(5)

sim_kpis = [
    (
        "Expected Cost / Ton",
        fmt_money(
            sim_cost_per_ton,
            2,
        ),
        f"vs {fmt_money(row['Cost_per_Ton_USD'], 2)} current",
    ),
    (
        "Observed Route DIFOT",
        f"{sim_difot:.1%}",
        (
            f"{'meets' if sim_service_compliant else 'below'} "
            f"{SERVICE_THRESHOLD:.0%} service constraint"
        ),
    ),
    (
        "Expected Revenue",
        fmt_money(
            sim_expected_revenue
        ),
        f"Δ {fmt_money(expected_revenue_change)}",
    ),
    (
        "Expected Gross Margin",
        fmt_money(
            sim_margin
        ),
        f"Δ {fmt_money(margin_change)}",
    ),
    (
        "Expected Margin / Ton",
        fmt_money(
            sim_margin_per_ton,
            2,
        ),
        f"route evidence n={sim_evidence_n}",
    ),
]

for c, (label, value, sub) in zip(
    sim_cols,
    sim_kpis,
):
    with c:
        metric_card(
            label,
            value,
            sub,
        )


comparison = pd.DataFrame(
    {
        "Route": [
            "Current",
            f"Expected · {sim_route}",
        ],
        "Cost / Ton": [
            float(
                row[
                    "Cost_per_Ton_USD"
                ]
            ),
            sim_cost_per_ton,
        ],
        "DIFOT": [
            float(
                row["DIFOT_Pct"]
            ),
            float(
                sim_difot * 100
            ),
        ],
        "Gross Margin": [
            current_margin,
            sim_margin,
        ],
    }
)


chart_cols = st.columns(2)

with chart_cols[0]:
    fig_cost = go.Figure()

    fig_cost.add_trace(
        go.Bar(
            x=comparison[
                "Route"
            ],
            y=comparison[
                "Cost / Ton"
            ],
            text=[
                f"${v:,.2f}"
                for v in comparison[
                    "Cost / Ton"
                ]
            ],
            textposition="outside",
            name="Cost / ton",
        )
    )

    fig_cost.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(
            l=10,
            r=20,
            t=50,
            b=10,
        ),
        showlegend=False,
        yaxis_title="USD / ton",
        title="Expected unit economics",
    )

    st.plotly_chart(
        fig_cost,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )

with chart_cols[1]:
    fig_outcome = go.Figure()

    fig_outcome.add_trace(
        go.Bar(
            x=[
                "Current",
                f"Expected · {sim_route}",
            ],
            y=[
                current_margin,
                sim_margin,
            ],
            text=[
                fmt_money(
                    current_margin
                ),
                fmt_money(
                    sim_margin
                ),
            ],
            textposition="outside",
            name="Gross margin",
        )
    )

    fig_outcome.add_hline(
        y=0,
        line_dash="dash",
        line_width=1,
    )

    fig_outcome.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(
            l=10,
            r=20,
            t=50,
            b=10,
        ),
        showlegend=False,
        yaxis_title="USD",
        title="Expected gross-margin outcome",
    )

    st.plotly_chart(
        fig_outcome,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )


waterfall = go.Figure(
    go.Waterfall(
        orientation="v",
        measure=[
            "absolute",
            "relative",
            "relative",
            "total",
        ],
        x=[
            "Current Gross Margin",
            "Expected Revenue Δ",
            "Cost Δ",
            "Expected Gross Margin",
        ],
        y=[
            current_margin,
            expected_revenue_change,
            -cost_change,
            sim_margin,
        ],
        text=[
            fmt_money(
                current_margin
            ),
            fmt_money(
                expected_revenue_change
            ),
            fmt_money(
                -cost_change
            ),
            fmt_money(
                sim_margin
            ),
        ],
        textposition="outside",
        connector={
            "line": {
                "width": 1
            }
        },
    )
)

waterfall.add_hline(
    y=0,
    line_dash="dash",
    line_width=1,
)

waterfall.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=390,
    margin=dict(
        l=10,
        r=20,
        t=20,
        b=10,
    ),
    title=(
        "How the expected route move changes shipment economics"
    ),
    yaxis_title="USD",
)

st.plotly_chart(
    waterfall,
    use_container_width=True,
    config={
        "displayModeBar": False
    },
)


if sim_route == current_route:
    info_card(
        "Simulation is set to the current route. "
        "The expected economic change should be approximately zero; "
        "any small difference comes from benchmark medians rather than a route intervention."
    )

elif not sim_service_compliant:
    warning_card(
        f"{sim_route} is below the {SERVICE_THRESHOLD:.0%} service constraint "
        f"for this Product × Route combination. It is therefore a conditional "
        "operational fallback, not a compliant primary benchmark."
    )

elif margin_change > 0:
    info_card(
        f"The observed {sim_route} route produces an expected gross-margin "
        f"improvement of {fmt_money(margin_change)} for this shipment. "
        "Expected revenue uses the route's observed DIFOT and therefore represents "
        "an expected value rather than a guaranteed shipment outcome."
    )

elif margin_change < 0:
    warning_card(
        f"Moving this shipment to the observed {sim_route} route reduces expected "
        f"gross margin by {fmt_money(abs(margin_change))}. "
        "The scenario therefore does not support the move economically under the "
        "observed route evidence."
    )

else:
    info_card(
        "The selected route produces approximately the same expected gross margin "
        "as the current shipment under the observed benchmark assumptions."
    )


# =============================================================================
# 10. DECISION ENGINE
# =============================================================================

section_header(
    "DECISION ENGINE",
    "The recommendation below is generated by the shared deterministic rules, "
    "not by the user-selected simulation route.",
)

decision_card(
    decision.primary_action,
    decision.explanation,
    decision.secondary_action,
)

for warning in decision.warnings:
    warning_card(warning)


# =============================================================================
# 11. DECISION EVIDENCE
# =============================================================================

st.markdown(
    "### Decision evidence"
)

reason_cols = st.columns(5)

reason_items = [
    (
        "Service",
        f"{decision.current_difot_pct:.1f}%",
        (
            f"Minimum: "
            f"{SERVICE_THRESHOLD * 100:.0f}%"
        ),
    ),
    (
        "Actionable benchmark",
        (
            decision.actionable_benchmark_route
            or "None"
        ),
        (
            (
                f"{decision.actionable_benchmark_difot_pct:.1f}% DIFOT · "
                f"n={decision.actionable_benchmark_n}"
            )
            if decision.actionable_benchmark_difot_pct
            is not None
            else (
                "No compliant observed route"
            )
        ),
    ),
    (
        "Economics",
        (
            fmt_money(
                decision.current_cost_per_ton,
                2,
            )
        ),
        (
            (
                f"Benchmark "
                f"{fmt_money(decision.actionable_benchmark_cost_per_ton, 2)}/ton"
            )
            if decision.actionable_benchmark_cost_per_ton
            is not None
            else "No benchmark"
        ),
    ),
    (
        "Avoidable",
        fmt_money(
            decision.avoidable_exposure
        ),
        (
            "actionable exposure"
            if decision.avoidable_exposure
            is not None
            else "benchmark unavailable"
        ),
    ),
    (
        "Residual",
        fmt_money(
            decision.residual_exposure
        ),
        (
            "after actionable benchmark"
            if decision.residual_exposure
            is not None
            else "benchmark unavailable"
        ),
    ),
]

for c, (label, value, sub) in zip(
    reason_cols,
    reason_items,
):
    with c:
        metric_card(
            label,
            value,
            sub,
        )


# =============================================================================
# 12. METHOD NOTE
# =============================================================================

info_card(
    "Method note: Direct is retained only as the historical/pre-blockade "
    "benchmark. The actionable benchmark excludes Direct and Held, requires "
    f"observed DIFOT ≥ {SERVICE_THRESHOLD:.0%} and at least 5 observations, and "
    "selects the lowest observed median cost/ton. For scenario reroutes, "
    "recognised revenue is shown as an expected value based on observed route "
    "DIFOT rather than as a guaranteed shipment-level outcome."
)
