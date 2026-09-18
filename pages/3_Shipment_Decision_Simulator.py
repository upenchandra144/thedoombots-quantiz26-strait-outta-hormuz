from __future__ import annotations

import html
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
from src.decision_rules import DecisionThresholds, assess_shipment
from src.metrics import DEFAULT_SERVICE_THRESHOLD, product_route_metrics


st.set_page_config(
    page_title="Shipment Decision Simulator | Hormuz War Room",
    page_icon="⚓",
    layout="wide",
)
inject_css()
df = load_data()

page_header(
    "SHIPMENT DECISION SIMULATOR",
    "Test the management decision for one shipment — compare route benchmarks, simulate the move, and see the economic consequence.",
)

# -----------------------------------------------------------------------------
# 1. SHIPMENT SELECTION
# -----------------------------------------------------------------------------
filters = st.columns(3)
with filters[0]:
    customer_filter = st.selectbox(
        "Filter by customer",
        ["All customers"] + sorted(df["Customer_Name"].dropna().astype(str).unique().tolist()),
    )
with filters[1]:
    product_filter = st.selectbox(
        "Filter by product",
        ["All products"] + sorted(df["Product_Category"].dropna().astype(str).unique().tolist()),
    )
with filters[2]:
    route_filter = st.selectbox(
        "Filter by current route",
        ["All routes"] + sorted(df["Route_Canonical"].dropna().astype(str).unique().tolist()),
    )

work = df.copy()
if customer_filter != "All customers":
    work = work[work["Customer_Name"].eq(customer_filter)]
if product_filter != "All products":
    work = work[work["Product_Category"].eq(product_filter)]
if route_filter != "All routes":
    work = work[work["Route_Canonical"].eq(route_filter)]

shipment_ids = work["Shipment_ID"].astype(str).tolist()
if not shipment_ids:
    st.warning("No shipments match the selected filters.")
    st.stop()

shipment_id = st.selectbox("Shipment ID", shipment_ids)
row = df[df["Shipment_ID"].astype(str).eq(str(shipment_id))].iloc[0]
decision = assess_shipment(
    df,
    shipment_id,
    thresholds=DecisionThresholds(service_threshold=DEFAULT_SERVICE_THRESHOLD),
)

product = str(row["Product_Category"])
current_route = str(row["Route_Canonical"])

# Product × route benchmark table used everywhere below.
route_table = product_route_metrics(df)
route_table = route_table[route_table["Product_Category"].eq(product)].copy()
route_table = route_table.sort_values("Median_Cost_per_Ton_USD")
route_table["Service_Compliant"] = route_table["DIFOT_Pct"] >= DEFAULT_SERVICE_THRESHOLD * 100
route_table["Current"] = route_table["Route_Canonical"].eq(current_route)

service_routes = route_table[route_table["Service_Compliant"]].copy()
non_direct_service_routes = service_routes[~service_routes["Route_Canonical"].isin(["Direct", "Held"])].copy()

canonical_benchmark = service_routes.iloc[0] if not service_routes.empty else None
reroute_benchmark = non_direct_service_routes.iloc[0] if not non_direct_service_routes.empty else None

# -----------------------------------------------------------------------------
# 2. SHIPMENT PROFILE
# -----------------------------------------------------------------------------
section_header("SHIPMENT PROFILE", "The operational and commercial context behind the decision.")
profile_cols = st.columns(6)
profile = [
    ("Customer", str(row["Customer_Name"]), str(row["Customer_Region"])),
    ("Product", product, str(row["Cargo_Type"])),
    ("Cargo Weight", f'{row["Cargo_Weight_Tons"]:,.1f} t', "shipment weight"),
    ("Cargo Value", fmt_money(row["Cargo_Value_USD"]), "cargo value"),
    ("Current Route", current_route, "observed route"),
    ("DIFOT", f'{row["DIFOT_Pct"]:.1f}%', "actual service result"),
]
for c, (label, value, sub) in zip(profile_cols, profile):
    with c:
        metric_card(label, value, sub)

meta = st.columns(5)
meta_values = [
    row["Departure_Date"].strftime("%d %b %Y") if pd.notna(row["Departure_Date"]) else "—",
    f'{row["Planned_Transit_Days"]:.0f} days' if pd.notna(row["Planned_Transit_Days"]) else "—",
    f'{row["Actual_Transit_Days"]:.0f} days' if pd.notna(row["Actual_Transit_Days"]) else "Held / unresolved",
    f'{row["Delay_Days"]:.0f} days' if pd.notna(row["Delay_Days"]) else "—",
    str(row["Cargo_Type"]),
]
for c, label, value in zip(meta, ["Departure", "Planned Transit", "Actual Transit", "Delay", "Cargo Type"], meta_values):
    with c:
        st.caption(label)
        st.write(value)

# -----------------------------------------------------------------------------
# 3. CURRENT ECONOMICS
# -----------------------------------------------------------------------------
section_header("CURRENT ECONOMICS", "Observed economics of the selected shipment.")
econ_cols = st.columns(6)
econ = [
    ("Cost / Ton", fmt_money(row["Cost_per_Ton_USD"], 2), "current"),
    ("Revenue / Ton", fmt_money(row["Revenue_per_Ton_USD"], 2), "contracted"),
    ("Margin / Ton", fmt_money(row["Margin_per_Ton_USD"], 2), "observed"),
    ("Total Cost", fmt_money(row["Total_Cost_to_Serve_USD"]), "observed"),
    ("Gross Margin", fmt_money(row["Gross_Margin_USD"]), "observed"),
    ("Recognised Revenue", fmt_money(row["Revenue_Recognized_USD"]), "observed"),
]
for c, (label, value, sub) in zip(econ_cols, econ):
    with c:
        metric_card(label, value, sub)

# -----------------------------------------------------------------------------
# 4. TWO-BENCHMARK FRAMEWORK
# -----------------------------------------------------------------------------
section_header(
    "TWO-BENCHMARK ROUTE LENS",
    "We separate the historical service benchmark from the observed non-Direct reroute alternative so the simulator does not automatically imply that Direct capacity is available post-blockade.",
)

bench_cols = st.columns(2)
with bench_cols[0]:
    if canonical_benchmark is not None:
        metric_card(
            "Historical / Canonical Benchmark",
            str(canonical_benchmark["Route_Canonical"]),
            f'{canonical_benchmark["DIFOT_Pct"]:.1f}% DIFOT · {fmt_money(canonical_benchmark["Median_Cost_per_Ton_USD"], 2)}/ton',
        )
        if str(canonical_benchmark["Route_Canonical"]) == "Direct":
            warning_card("Direct is a pre-blockade benchmark here — not a guaranteed post-blockade capacity commitment.")
    else:
        metric_card("Historical / Canonical Benchmark", "None observed", "No route meets the 80% DIFOT threshold.")

with bench_cols[1]:
    if reroute_benchmark is not None:
        metric_card(
            "Disruption-Era Reroute Benchmark",
            str(reroute_benchmark["Route_Canonical"]),
            f'{reroute_benchmark["DIFOT_Pct"]:.1f}% DIFOT · {fmt_money(reroute_benchmark["Median_Cost_per_Ton_USD"], 2)}/ton',
        )
    else:
        metric_card(
            "Disruption-Era Reroute Benchmark",
            "None observed",
            "No observed non-Direct route meets the 80% DIFOT threshold.",
        )

# -----------------------------------------------------------------------------
# 5. ROUTE LANDSCAPE
# -----------------------------------------------------------------------------
section_header(
    "ROUTE LANDSCAPE FOR THIS PRODUCT",
    "Each bubble is an observed product × route combination. The 80% line is the service-compliance constraint used by the benchmark framework.",
)

plot_routes = route_table.copy()
plot_routes["Route_Label"] = plot_routes["Route_Canonical"]
plot_routes["Benchmark_Type"] = "Other observed route"
plot_routes.loc[plot_routes["Current"], "Benchmark_Type"] = "Current route"
if canonical_benchmark is not None:
    plot_routes.loc[
        plot_routes["Route_Canonical"].eq(str(canonical_benchmark["Route_Canonical"])),
        "Benchmark_Type",
    ] = "Historical benchmark"
if reroute_benchmark is not None:
    plot_routes.loc[
        plot_routes["Route_Canonical"].eq(str(reroute_benchmark["Route_Canonical"])),
        "Benchmark_Type",
    ] = "Reroute benchmark"

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
        "Benchmark_Type": True,
    },
    height=420,
)
bubble.add_hline(
    y=DEFAULT_SERVICE_THRESHOLD * 100,
    line_dash="dash",
    line_width=1.5,
    annotation_text="80% service threshold",
    annotation_position="top left",
)
bubble.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="#E2E8F0")))
bubble.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=20, t=10, b=10),
    xaxis_title="Observed median cost / ton (USD)",
    yaxis_title="Observed DIFOT (%)",
    legend_title="Route lens",
)
st.plotly_chart(bubble, use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# 6. ALTERNATIVE ROUTE TABLE
# -----------------------------------------------------------------------------
section_header(
    "OBSERVED ALTERNATIVE ROUTES",
    "Audit view of the route evidence used by the simulator.",
)

route_display = route_table[
    [
        "Route_Canonical",
        "Shipments",
        "Median_Cost_per_Ton_USD",
        "Median_Margin_per_Ton_USD",
        "DIFOT_Pct",
        "Service_Compliant",
        "Current",
    ]
].copy()

route_display["Service"] = route_display["Service_Compliant"].map(
    lambda x: "✓ Compliant" if x else "Below threshold"
)

# Put the human-readable Service column where the compliance flag currently sits
route_display = route_display[
    [
        "Route_Canonical",
        "Shipments",
        "Median_Cost_per_Ton_USD",
        "Median_Margin_per_Ton_USD",
        "DIFOT_Pct",
        "Service",
        "Current",
    ]
]

route_display.columns = [
    "Route",
    "Shipments",
    "Median Cost / Ton",
    "Median Margin / Ton",
    "DIFOT",
    "Service",
    "Current",
]

st.dataframe(
    route_display.style.format(
        {
            "Median Cost / Ton": "${:,.2f}",
            "Median Margin / Ton": "${:,.2f}",
            "DIFOT": "{:.1f}%",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# 7. ACTUAL SHIPMENT SIMULATOR
# -----------------------------------------------------------------------------
section_header(
    "SIMULATE THE SHIPMENT",
    "Choose an observed route and see the modeled shipment-level economic consequence. This is a benchmark simulation, not a capacity forecast.",
)

available_sim_routes = route_table[route_table["Route_Canonical"] != "Held"]["Route_Canonical"].astype(str).tolist()
if current_route not in available_sim_routes:
    available_sim_routes = [current_route] + available_sim_routes

recommended_default = None
if reroute_benchmark is not None:
    recommended_default = str(reroute_benchmark["Route_Canonical"])
elif canonical_benchmark is not None:
    recommended_default = str(canonical_benchmark["Route_Canonical"])
elif current_route in available_sim_routes:
    recommended_default = current_route

if recommended_default not in available_sim_routes:
    recommended_default = available_sim_routes[0]

selected_index = available_sim_routes.index(recommended_default)
sim_route = st.selectbox(
    "Simulate route",
    available_sim_routes,
    index=selected_index,
    key=f"sim_route_{shipment_id}",
)

sim_row = route_table[route_table["Route_Canonical"].eq(sim_route)].iloc[0]
weight = float(row["Cargo_Weight_Tons"])
contracted_revenue = float(row["Contracted_Freight_Revenue_USD"])
current_cost = float(row["Total_Cost_to_Serve_USD"])
current_recognised_revenue = float(row["Revenue_Recognized_USD"])
current_margin = float(row["Gross_Margin_USD"])

sim_cost_per_ton = float(sim_row["Median_Cost_per_Ton_USD"])
sim_cost = sim_cost_per_ton * weight
sim_difot = float(sim_row["DIFOT_Pct"]) / 100.0
sim_recognised_revenue = contracted_revenue * sim_difot
sim_margin = sim_recognised_revenue - sim_cost
sim_margin_per_ton = sim_margin / weight if weight else np.nan

cost_change = sim_cost - current_cost
recognised_revenue_change = sim_recognised_revenue - current_recognised_revenue
margin_change = sim_margin - current_margin

sim_cols = st.columns(5)
sim_kpis = [
    ("Simulated Cost / Ton", fmt_money(sim_cost_per_ton, 2), f"vs {fmt_money(row['Cost_per_Ton_USD'], 2)} current"),
    ("Simulated DIFOT", f"{sim_difot:.1%}", f"vs {float(row['DIFOT_Flag']):.0%} current"),
    ("Simulated Recognised Revenue", fmt_money(sim_recognised_revenue), f"Δ {fmt_money(recognised_revenue_change)}"),
    ("Simulated Gross Margin", fmt_money(sim_margin), f"Δ {fmt_money(margin_change)}"),
    ("Simulated Margin / Ton", fmt_money(sim_margin_per_ton, 2), f"Δ {fmt_money(sim_margin_per_ton - float(row['Margin_per_Ton_USD']), 2)}"),
]
for c, (label, value, sub) in zip(sim_cols, sim_kpis):
    with c:
        metric_card(label, value, sub)

# Main simulation chart: one view with cost, service and margin consequences.
comparison = pd.DataFrame(
    {
        "Route": ["Current", f"Simulated · {sim_route}"],
        "Cost / Ton": [float(row["Cost_per_Ton_USD"]), sim_cost_per_ton],
        "DIFOT": [float(row["DIFOT_Pct"]), float(sim_row["DIFOT_Pct"])],
        "Gross Margin": [current_margin, sim_margin],
    }
)

chart_cols = st.columns(2)
with chart_cols[0]:
    fig_cost = go.Figure()
    fig_cost.add_trace(
        go.Bar(
            x=comparison["Route"],
            y=comparison["Cost / Ton"],
            text=[f"${v:,.2f}" for v in comparison["Cost / Ton"]],
            textposition="outside",
            name="Cost / ton",
        )
    )
    fig_cost.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=10, r=20, t=50, b=10),
        showlegend=False,
        yaxis_title="USD / ton",
        title="Unit economics",
    )
    st.plotly_chart(fig_cost, use_container_width=True, config={"displayModeBar": False})

with chart_cols[1]:
    fig_outcome = go.Figure()
    fig_outcome.add_trace(
        go.Bar(
            x=["Current", f"Simulated · {sim_route}"],
            y=[current_margin, sim_margin],
            text=[fmt_money(current_margin), fmt_money(sim_margin)],
            textposition="outside",
            name="Gross margin",
        )
    )
    fig_outcome.add_hline(y=0, line_dash="dash", line_width=1)
    fig_outcome.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=10, r=20, t=50, b=10),
        showlegend=False,
        yaxis_title="USD",
        title="Shipment gross-margin outcome",
    )
    st.plotly_chart(fig_outcome, use_container_width=True, config={"displayModeBar": False})

# Waterfall explains exactly why the scenario changed.
waterfall = go.Figure(
    go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=[
            "Current Gross Margin",
            "Recognised Revenue Δ",
            "Cost Δ",
            "Simulated Gross Margin",
        ],
        y=[current_margin, recognised_revenue_change, -cost_change, sim_margin],
        text=[fmt_money(current_margin), fmt_money(recognised_revenue_change), fmt_money(-cost_change), fmt_money(sim_margin)],
        textposition="outside",
        connector={"line": {"width": 1}},
    )
)
waterfall.add_hline(y=0, line_dash="dash", line_width=1)
waterfall.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=390,
    margin=dict(l=10, r=20, t=20, b=10),
    title="How the simulated move changes the shipment economics",
    yaxis_title="USD",
)
st.plotly_chart(waterfall, use_container_width=True, config={"displayModeBar": False})

# Concise interpretation.
if sim_route == current_route:
    info_card("Simulation is set to the current route, so the displayed economic change is expected to be approximately zero apart from benchmark-median differences.")
elif margin_change > 0:
    info_card(
        f"The observed {sim_route} benchmark improves modeled gross margin by {fmt_money(margin_change)} for this shipment. The result comes from the benchmark route's observed cost/ton and DIFOT; it is not a guarantee of future capacity or savings."
    )
elif margin_change < 0:
    warning_card(
        f"Moving this shipment to the observed {sim_route} benchmark reduces modeled gross margin by {fmt_money(abs(margin_change))}. The simulator therefore does not support that move economically under the observed benchmark data."
    )
else:
    info_card("The selected route produces approximately the same modeled gross margin as the current shipment under the observed benchmark assumptions.")

# -----------------------------------------------------------------------------
# 8. DECISION ENGINE
# -----------------------------------------------------------------------------
section_header(
    "DECISION ENGINE",
    "The recommendation below remains generated by the shared deterministic rules, not by the user-selected simulation route.",
)
decision_card(decision.primary_action, decision.explanation, decision.secondary_action)
for warning in decision.warnings:
    warning_card(warning)

# Explicit reason block so the action never appears without context.
st.markdown("### Why this action?")
reason_cols = st.columns(4)
reason_items = [
    (
        "Service",
        f"Current DIFOT: {decision.current_difot_pct:.1f}%",
        f"Threshold: {DEFAULT_SERVICE_THRESHOLD * 100:.0f}%",
    ),
    (
        "Economics",
        f"Current: {fmt_money(decision.current_cost_per_ton, 2)}/ton",
        f"Benchmark: {fmt_money(decision.best_service_compliant_cost_per_ton, 2)}/ton" if decision.best_service_compliant_cost_per_ton is not None else "No compliant benchmark",
    ),
    (
        "Exposure",
        fmt_money(decision.avoidable_exposure),
        "modeled avoidable exposure" if decision.avoidable_exposure is not None else "benchmark unavailable",
    ),
    (
        "Residual",
        fmt_money(decision.residual_exposure),
        "modeled residual exposure" if decision.residual_exposure is not None else "benchmark unavailable",
    ),
]
for c, (label, value, sub) in zip(reason_cols, reason_items):
    with c:
        metric_card(label, value, sub)

info_card(
    "Method note: the canonical counterfactual uses the lowest observed cost/ton among routes meeting DIFOT ≥ 80% for this product. The simulator additionally shows the cheapest observed non-Direct service-compliant route when available, because Direct is a pre-blockade benchmark rather than a guaranteed post-blockade option."
)
