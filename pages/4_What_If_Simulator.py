from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    fmt_money,
    fmt_pct,
    inject_css,
    info_card,
    metric_card,
    page_header,
    section_header,
    warning_card,
)
from src.data_loader import load_data
from src.metrics import DEFAULT_SERVICE_THRESHOLD
from src.scenario_engine import (
    ScenarioInputs,
    simulate_scenario,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="What-If Simulator | Hormuz War Room",
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
    "WHAT-IF SIMULATOR",
    "Pull the levers. See how route, cost, commercial recovery and held-cargo "
    "actions reshape the portfolio under the same decision framework used by "
    "the shipment simulator.",
)


# =============================================================================
# SCENARIO CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown(
        "### SCENARIO CONTROL ROOM"
    )

    st.caption(
        "Every lever below is an explicit scenario assumption — not a forecast."
    )

    route_choice = st.radio(
        "Route intervention",
        [
            "No route intervention",
            "Pipeline → Cape",
            "Pipeline → actionable disruption-era benchmark",
        ],
        index=0,
    )

    st.markdown(
        "#### COST LEVERS"
    )

    freight = (
        st.slider(
            "Freight cost adjustment",
            -30,
            30,
            0,
            1,
            format="%d%%",
        )
        / 100
    )

    fuel = (
        st.slider(
            "Fuel cost adjustment",
            -30,
            30,
            0,
            1,
            format="%d%%",
        )
        / 100
    )

    insurance = (
        st.slider(
            "Insurance adjustment",
            -30,
            30,
            0,
            1,
            format="%d%%",
        )
        / 100
    )

    penalty = (
        st.slider(
            "Penalty exposure adjustment",
            -50,
            20,
            0,
            1,
            format="%d%%",
        )
        / 100
    )

    st.markdown(
        "#### COMMERCIAL / SERVICE LEVERS"
    )

    pass_through = (
        st.slider(
            "Customer pass-through / surcharge",
            0,
            30,
            0,
            1,
            format="%d%%",
        )
        / 100
    )

    held_days = st.slider(
        "Held-cargo release acceleration",
        0,
        30,
        0,
        1,
    )

    st.divider()

    st.markdown(
        "#### INTERPRETATION"
    )

    st.caption(
        "Cost levers affect scenario cost. Pass-through affects contracted "
        "and expected recognised revenue. Release acceleration unlocks a "
        "fraction of currently trapped revenue and scales held penalty / "
        "insurance exposure."
    )

    st.info(
        f"Service constraint: {SERVICE_THRESHOLD:.0%}. "
        "Actionable benchmark excludes Direct and Held and requires at least "
        "5 observed shipments."
    )


pipeline_to_cape = (
    route_choice
    == "Pipeline → Cape"
)

pipeline_to_benchmark = (
    route_choice
    == "Pipeline → actionable disruption-era benchmark"
)

inputs = ScenarioInputs(
    freight_adjustment_pct=freight,
    fuel_adjustment_pct=fuel,
    insurance_adjustment_pct=insurance,
    penalty_adjustment_pct=penalty,
    customer_pass_through_pct=pass_through,
    held_release_days_earlier=held_days,
    pipeline_to_cape=pipeline_to_cape,
    pipeline_to_benchmark=pipeline_to_benchmark,
    service_threshold=SERVICE_THRESHOLD,
)

scenario_df, result = simulate_scenario(
    df,
    inputs,
)


# =============================================================================
# SCENARIO STATUS
# =============================================================================

active_levers: list[str] = []

if route_choice != "No route intervention":
    active_levers.append(
        route_choice
    )

if freight:
    active_levers.append(
        f"Freight {freight:+.0%}"
    )

if fuel:
    active_levers.append(
        f"Fuel {fuel:+.0%}"
    )

if insurance:
    active_levers.append(
        f"Insurance {insurance:+.0%}"
    )

if penalty:
    active_levers.append(
        f"Penalty {penalty:+.0%}"
    )

if pass_through:
    active_levers.append(
        f"Pass-through {pass_through:+.0%}"
    )

if held_days:
    active_levers.append(
        f"Release +{held_days}d"
    )

if not active_levers:
    info_card(
        "BASELINE MODE · No scenario levers are active. "
        "Turn on one or more controls to simulate a management intervention."
    )
else:
    info_card(
        "ACTIVE SCENARIO · "
        + " · ".join(active_levers)
    )


# =============================================================================
# 1. PORTFOLIO RESPONSE
# =============================================================================

section_header(
    "PORTFOLIO RESPONSE",
    "The first question is not whether one metric improves — it is whether "
    "the intervention changes the overall economic position.",
)

kpis = st.columns(8)

kpi_items = [
    (
        "Expected Gross Margin",
        fmt_money(
            result["scenario_gross_margin"]
        ),
        f"Δ {fmt_money(result['gross_margin_change'])}",
    ),
    (
        "Expected Margin %",
        fmt_pct(
            result["scenario_margin_pct"]
        ),
        f"Δ {result['margin_pct_change'] * 100:+.1f} pp",
    ),
    (
        "Expected Total Cost",
        fmt_money(
            result["scenario_total_cost"]
        ),
        f"Δ {fmt_money(result['total_cost_change'])}",
    ),
    (
        "Expected Revenue",
        fmt_money(
            result[
                "scenario_expected_revenue_recognized"
            ]
        ),
        f"Δ {fmt_money(result['scenario_expected_revenue_recognized'] - result['baseline_expected_revenue_recognized'])}",
    ),
    (
        "Held Exposure",
        fmt_money(
            result["scenario_held_exposure"]
        ),
        f"Δ {fmt_money(result['held_exposure_change'])}",
    ),
    (
        "Avoidable Exposure",
        fmt_money(
            result[
                "scenario_avoidable_exposure"
            ]
        ),
        f"Δ {fmt_money(result['avoidable_exposure_change'])}",
    ),
    (
        "Residual Exposure",
        fmt_money(
            result[
                "scenario_residual_exposure"
            ]
        ),
        f"Δ {fmt_money(result['residual_exposure_change'])}",
    ),
    (
        "Held Shipments",
        f"{int(scenario_df['Is_Held'].sum()):,}",
        f"baseline {int(df['Is_Held'].sum()):,}",
    ),
]

for col, (label, value, sub) in zip(
    kpis,
    kpi_items,
):
    with col:
        metric_card(
            label,
            value,
            sub,
        )


# =============================================================================
# 2. MARGIN BRIDGE
# =============================================================================

section_header(
    "HOW DID THE SCENARIO CHANGE THE BOTTOM LINE?",
    "The bridge separates expected revenue movement from cost movement. "
    "For rerouted shipments, expected recognised revenue uses the observed "
    "route DIFOT rather than assuming a deterministic shipment outcome.",
)

baseline_margin = float(
    result[
        "baseline_gross_margin"
    ]
)

scenario_margin = float(
    result[
        "scenario_gross_margin"
    ]
)

revenue_delta = (
    float(
        result[
            "scenario_expected_revenue_recognized"
        ]
    )
    -
    float(
        result[
            "baseline_expected_revenue_recognized"
        ]
    )
)

cost_delta = float(
    result[
        "total_cost_change"
    ]
)

bridge = go.Figure(
    go.Waterfall(
        orientation="v",
        measure=[
            "absolute",
            "relative",
            "relative",
            "total",
        ],
        x=[
            "Baseline GM",
            "Expected Revenue Δ",
            "Cost movement",
            "Scenario Expected GM",
        ],
        y=[
            baseline_margin,
            revenue_delta,
            -cost_delta,
            scenario_margin,
        ],
        text=[
            fmt_money(
                baseline_margin
            ),
            f"{revenue_delta:+,.0f}",
            f"{-cost_delta:+,.0f}",
            fmt_money(
                scenario_margin
            ),
        ],
        textposition="outside",
        connector={
            "line": {
                "color": "#475569"
            }
        },
    )
)

bridge.add_hline(
    y=0,
    line_dash="dash",
    line_width=1,
)

bridge.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=390,
    margin=dict(
        l=20,
        r=20,
        t=15,
        b=10,
    ),
    showlegend=False,
    yaxis_title="USD",
)

st.plotly_chart(
    bridge,
    use_container_width=True,
    config={
        "displayModeBar": False
    },
)


# =============================================================================
# 3. COST MIX: BASELINE VS SCENARIO
# =============================================================================

left, right = st.columns(2)

with left:
    section_header(
        "COST STRUCTURE · BASELINE VS SCENARIO",
        "Shows which part of the cost-to-serve actually moved.",
    )

    cost_cols = [
        "Freight_Cost_USD",
        "Fuel_Cost_USD",
        "Insurance_Cost_USD",
        "Penalty_Cost_USD",
    ]

    cost_labels = {
        "Freight_Cost_USD":
            "Freight",
        "Fuel_Cost_USD":
            "Fuel",
        "Insurance_Cost_USD":
            "Insurance",
        "Penalty_Cost_USD":
            "Penalty",
    }

    cost_df = pd.DataFrame(
        {
            "Component": [
                cost_labels[c]
                for c in cost_cols
            ],
            "Baseline": [
                float(df[c].sum())
                for c in cost_cols
            ],
            "Scenario Expected": [
                float(
                    scenario_df[
                        f"Scenario_{c}"
                    ].sum()
                )
                for c in cost_cols
            ],
        }
    )

    cost_long = cost_df.melt(
        "Component",
        var_name="State",
        value_name="USD",
    )

    fig_cost = px.bar(
        cost_long,
        x="State",
        y="USD",
        color="Component",
        barmode="stack",
        text_auto=False,
    )

    fig_cost.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        yaxis_title="USD",
        xaxis_title="",
        legend_title="Cost component",
    )

    st.plotly_chart(
        fig_cost,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )

with right:
    section_header(
        "EXPOSURE RESPONSE",
        "The scenario should ideally reduce trapped cargo exposure and "
        "modeled avoidable exposure while making the residual loss explicit.",
    )

    exposure_df = pd.DataFrame(
        {
            "Metric": [
                "Held exposure",
                "Avoidable exposure",
                "Residual exposure",
            ],
            "Baseline": [
                result[
                    "baseline_held_exposure"
                ],
                result[
                    "baseline_avoidable_exposure"
                ],
                result[
                    "baseline_residual_exposure"
                ],
            ],
            "Scenario": [
                result[
                    "scenario_held_exposure"
                ],
                result[
                    "scenario_avoidable_exposure"
                ],
                result[
                    "scenario_residual_exposure"
                ],
            ],
        }
    )

    exposure_long = exposure_df.melt(
        "Metric",
        var_name="State",
        value_name="USD",
    )

    fig_exp = px.bar(
        exposure_long,
        x="Metric",
        y="USD",
        color="State",
        barmode="group",
        text_auto=False,
    )

    fig_exp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        yaxis_title="USD",
        xaxis_title="",
    )

    st.plotly_chart(
        fig_exp,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )


# =============================================================================
# 4. ROUTE SHIFT / OPERATIONAL CONSEQUENCE
# =============================================================================

if route_choice != "No route intervention":
    section_header(
        "ROUTE PLAN IMPACT",
        "Shows where cargo moves when a Pipeline intervention is activated. "
        "The benchmark scenario uses the same actionable route universe as "
        "the decision engine.",
    )

    route_base = (
        df.groupby(
            "Route_Canonical"
        )["Cargo_Weight_Tons"]
        .sum()
        .rename("Baseline Tons")
        .rename_axis("Route")
    )

    route_scn = (
        scenario_df.groupby(
            "Scenario_Route"
        )["Cargo_Weight_Tons"]
        .sum()
        .rename("Scenario Tons")
        .rename_axis("Route")
    )

    route_shift = (
        pd.concat(
            [
                route_base,
                route_scn,
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
    )

    route_shift = route_shift.melt(
        id_vars="Route",
        var_name="State",
        value_name="Tons",
    )

    fig_route = px.bar(
        route_shift,
        x="Route",
        y="Tons",
        color="State",
        barmode="group",
        text_auto=False,
    )

    fig_route.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        yaxis_title="Cargo weight (tons)",
        xaxis_title="",
    )

    st.plotly_chart(
        fig_route,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )

else:
    section_header(
        "WHERE DOES THE SCENARIO MOVE THE MARGIN?",
        "With no route intervention active, this view isolates route-level "
        "gross-margin movement created by cost and commercial levers.",
    )

    route_base_margin = (
        df.groupby(
            "Route_Canonical"
        )[
            "Gross_Margin_USD"
        ]
        .sum()
        .rename(
            "Baseline"
        )
    )

    route_scn_margin = (
        scenario_df.groupby(
            "Scenario_Route"
        )[
            "Scenario_Expected_Gross_Margin_USD"
        ]
        .sum()
        .rename(
            "Scenario"
        )
    )

    route_margin = (
        pd.concat(
            [
                route_base_margin,
                route_scn_margin,
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
    )

    route_margin = route_margin.rename(
        columns={
            "index": "Route",
            "Route_Canonical": "Route",
        }
    )

    route_margin["Change"] = (
        route_margin[
            "Scenario"
        ]
        -
        route_margin[
            "Baseline"
        ]
    )

    route_margin = route_margin.sort_values(
        "Change"
    )

    fig_route = px.bar(
        route_margin,
        x="Change",
        y="Route",
        orientation="h",
        text="Change",
    )

    fig_route.update_traces(
        texttemplate="$%{text:.3s}",
        textposition="outside",
    )

    fig_route.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(
            l=10,
            r=90,
            t=10,
            b=10,
        ),
        xaxis_title="Expected gross-margin change (USD)",
        yaxis_title="",
        showlegend=False,
    )

    st.plotly_chart(
        fig_route,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )


# =============================================================================
# 5. LEVER SENSITIVITY / CONTRIBUTION VIEW
# =============================================================================

section_header(
    "WHICH LEVER IS DOING THE WORK?",
    "Each bar applies one selected lever in isolation from the baseline. "
    "This is a sensitivity view, not an additive attribution.",
)

lever_cases: list[
    tuple[str, ScenarioInputs]
] = []

if freight:
    lever_cases.append(
        (
            f"Freight {freight:+.0%}",
            ScenarioInputs(
                freight_adjustment_pct=freight,
                service_threshold=SERVICE_THRESHOLD,
            ),
        )
    )

if fuel:
    lever_cases.append(
        (
            f"Fuel {fuel:+.0%}",
            ScenarioInputs(
                fuel_adjustment_pct=fuel,
                service_threshold=SERVICE_THRESHOLD,
            ),
        )
    )

if insurance:
    lever_cases.append(
        (
            f"Insurance {insurance:+.0%}",
            ScenarioInputs(
                insurance_adjustment_pct=insurance,
                service_threshold=SERVICE_THRESHOLD,
            ),
        )
    )

if penalty:
    lever_cases.append(
        (
            f"Penalty {penalty:+.0%}",
            ScenarioInputs(
                penalty_adjustment_pct=penalty,
                service_threshold=SERVICE_THRESHOLD,
            ),
        )
    )

if pass_through:
    lever_cases.append(
        (
            f"Pass-through {pass_through:+.0%}",
            ScenarioInputs(
                customer_pass_through_pct=pass_through,
                service_threshold=SERVICE_THRESHOLD,
            ),
        )
    )

if held_days:
    lever_cases.append(
        (
            f"Release +{held_days}d",
            ScenarioInputs(
                held_release_days_earlier=held_days,
                service_threshold=SERVICE_THRESHOLD,
            ),
        )
    )

if pipeline_to_cape:
    lever_cases.append(
        (
            "Pipeline → Cape",
            ScenarioInputs(
                pipeline_to_cape=True,
                service_threshold=SERVICE_THRESHOLD,
            ),
        )
    )

if pipeline_to_benchmark:
    lever_cases.append(
        (
            "Pipeline → actionable benchmark",
            ScenarioInputs(
                pipeline_to_benchmark=True,
                service_threshold=SERVICE_THRESHOLD,
            ),
        )
    )

if lever_cases:
    lever_names = []
    lever_changes = []

    for name, lever_input in lever_cases:
        _, lever_result = simulate_scenario(
            df,
            lever_input,
        )

        lever_names.append(name)
        lever_changes.append(
            lever_result[
                "gross_margin_change"
            ]
        )

    lever_df = (
        pd.DataFrame(
            {
                "Lever": lever_names,
                "Expected Gross Margin Change":
                    lever_changes,
            }
        )
        .sort_values(
            "Expected Gross Margin Change"
        )
    )

    fig_lev = px.bar(
        lever_df,
        x="Expected Gross Margin Change",
        y="Lever",
        orientation="h",
        text="Expected Gross Margin Change",
    )

    fig_lev.update_traces(
        texttemplate="$%{text:.3s}",
        textposition="outside",
    )

    fig_lev.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(
            280,
            55 * len(lever_df),
        ),
        margin=dict(
            l=10,
            r=90,
            t=10,
            b=10,
        ),
        xaxis_title="Isolated expected gross-margin change (USD)",
        yaxis_title="",
        showlegend=False,
    )

    st.plotly_chart(
        fig_lev,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )

else:
    info_card(
        "No active levers to decompose yet. Activate one or more controls "
        "in the sidebar to reveal the sensitivity view."
    )


# =============================================================================
# 6. SCENARIO AUDIT TABLE
# =============================================================================

section_header(
    "SCENARIO AUDIT",
    "The underlying portfolio numbers remain visible so the scenario is "
    "reproducible and explainable. Rerouted revenue and margin are shown as "
    "expected values rather than guaranteed outcomes.",
)

comparison = pd.DataFrame(
    [
        [
            "Gross Margin",
            result[
                "baseline_gross_margin"
            ],
            result[
                "scenario_gross_margin"
            ],
            result[
                "gross_margin_change"
            ],
        ],
        [
            "Margin %",
            result[
                "baseline_margin_pct"
            ],
            result[
                "scenario_margin_pct"
            ],
            result[
                "margin_pct_change"
            ],
        ],
        [
            "Expected Revenue",
            result[
                "baseline_expected_revenue_recognized"
            ],
            result[
                "scenario_expected_revenue_recognized"
            ],
            (
                result[
                    "scenario_expected_revenue_recognized"
                ]
                -
                result[
                    "baseline_expected_revenue_recognized"
                ]
            ),
        ],
        [
            "Total Cost",
            result[
                "baseline_total_cost"
            ],
            result[
                "scenario_total_cost"
            ],
            result[
                "total_cost_change"
            ],
        ],
        [
            "Held Exposure",
            result[
                "baseline_held_exposure"
            ],
            result[
                "scenario_held_exposure"
            ],
            result[
                "held_exposure_change"
            ],
        ],
        [
            "Avoidable Exposure",
            result[
                "baseline_avoidable_exposure"
            ],
            result[
                "scenario_avoidable_exposure"
            ],
            result[
                "avoidable_exposure_change"
            ],
        ],
        [
            "Residual Exposure",
            result[
                "baseline_residual_exposure"
            ],
            result[
                "scenario_residual_exposure"
            ],
            result[
                "residual_exposure_change"
            ],
        ],
    ],
    columns=[
        "Metric",
        "Baseline",
        "Scenario",
        "Change",
    ],
)

comparison_display = comparison.copy()

for col in [
    "Baseline",
    "Scenario",
]:
    comparison_display[col] = comparison_display.apply(
        lambda r:
            (
                f"{r[col] * 100:.1f}%"
                if r["Metric"] == "Margin %"
                else f"${r[col]:,.0f}"
            ),
        axis=1,
    )

comparison_display[
    "Change"
] = comparison_display.apply(
    lambda r:
        (
            f"{r['Change'] * 100:+.1f} pp"
            if r["Metric"] == "Margin %"
            else f"${r['Change']:+,.0f}"
        ),
    axis=1,
)

st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True,
)


# =============================================================================
# 7. MANAGEMENT INTERPRETATION
# =============================================================================

section_header(
    "MANAGEMENT READOUT"
)

margin_change = result[
    "gross_margin_change"
]

held_change = result[
    "held_exposure_change"
]

avoid_change = result[
    "avoidable_exposure_change"
]

residual_change = result[
    "residual_exposure_change"
]

positive_margin = margin_change > 0
reduces_avoidable = avoid_change < 0
reduces_held = held_change < 0
reduces_residual = residual_change < 0

if (
    positive_margin
    and reduces_avoidable
    and reduces_held
):
    insight = (
        "The scenario improves expected portfolio gross margin while reducing "
        "both modeled avoidable exposure and trapped revenue exposure. "
        "This is a broad-based intervention rather than a single-metric improvement."
    )

elif (
    positive_margin
    and reduces_avoidable
):
    insight = (
        "The scenario improves expected gross margin and reduces modeled "
        "avoidable exposure, but trapped-revenue exposure remains comparatively "
        "unchanged. Operational and commercial actions should be considered together."
    )

elif (
    positive_margin
    and reduces_held
):
    insight = (
        "The scenario improves expected gross margin and releases trapped "
        "revenue exposure, but route economics still leave modeled avoidable "
        "exposure on the book."
    )

elif (
    reduces_avoidable
    and reduces_held
):
    insight = (
        "The scenario reduces both actionable route exposure and trapped-cargo "
        "exposure, although the portfolio does not yet show a positive expected "
        "gross-margin movement."
    )

elif (
    positive_margin
    and reduces_residual
):
    insight = (
        "The scenario improves expected gross margin and reduces residual "
        "exposure under the observed benchmark framework."
    )

elif positive_margin:
    insight = (
        "The scenario improves expected gross margin, but the remaining "
        "exposure indicates that cost or commercial improvements alone do not "
        "fully resolve the network economics."
    )

elif (
    reduces_avoidable
    or reduces_held
    or reduces_residual
):
    insight = (
        "The scenario reduces an important operational exposure, but it does "
        "not yet translate into a positive expected portfolio gross-margin movement. "
        "The result points toward a combined route + commercial response."
    )

else:
    insight = (
        "The selected assumptions do not materially improve portfolio economics. "
        "Revisit the route intervention, cost assumptions or commercial recovery lever."
    )

st.markdown(
    f"### {insight}"
)


# =============================================================================
# 8. METHOD / SCENARIO INTERPRETATION
# =============================================================================

if (
    route_choice
    == "Pipeline → actionable disruption-era benchmark"
):
    warning_card(
        "Benchmark note: this intervention uses the lowest observed non-Direct, "
        f"non-Held Product × Route alternative meeting DIFOT ≥ {SERVICE_THRESHOLD:.0%} "
        "with at least five observations. Where no such route exists, the affected "
        "Pipeline shipments remain on Pipeline rather than being assigned an invented route."
    )

elif route_choice == "Pipeline → Cape":
    info_card(
        "Cape is treated here as an explicit observed non-Direct reroute scenario. "
        "The simulation does not imply unlimited future Cape capacity; it applies "
        "observed Product × Route economics and observed DIFOT to the affected Pipeline shipments."
    )


info_card(
    "Expected-value method: for rerouted shipments, "
    "E[Revenue] = contracted revenue × observed route DIFOT. "
    "Route costs use observed median Product × Route component rates. "
    "Scenario levers are assumptions, not forecasts or guarantees of savings."
)

st.caption(
    "The scenario engine is intentionally aligned with the shipment decision "
    "engine: historical Direct performance remains a reference, while actionable "
    "rerouting excludes Direct and Held and respects the 80% service constraint."
)
