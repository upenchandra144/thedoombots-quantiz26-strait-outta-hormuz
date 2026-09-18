from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    action_chips,
    fmt_money,
    fmt_pct,
    inject_css,
    metric_card,
    page_header,
    section_header,
)
from src.data_loader import load_data
from src.metrics import (
    concentration_metrics,
    counterfactual_metrics,
    customer_metrics,
    held_metrics,
    loss_concentration,
    portfolio_metrics,
    product_route_metrics,
    route_metrics,
)


st.set_page_config(
    page_title="Executive War Dashboard | Hormuz War Room",
    page_icon="⚓",
    layout="wide",
)

inject_css()
df = load_data()

portfolio = portfolio_metrics(df)
held = held_metrics(df)
conc = concentration_metrics(df)
loss = loss_concentration(df)
_, cf_route = counterfactual_metrics(df)
route = route_metrics(df)
customers = customer_metrics(df)
product_route = product_route_metrics(df)

# -----------------------------------------------------------------------------
# PAGE HEADER
# -----------------------------------------------------------------------------
page_header(
    "STRAIT OUTTA HORMUZ",
    "WAR ROOM CONTROL · What is happening to the business right now?",
)

# -----------------------------------------------------------------------------
# EXECUTIVE KPI ROW
# -----------------------------------------------------------------------------
kpi_cols = st.columns(4)

with kpi_cols[0]:
    metric_card(
        "Contracted Revenue",
        fmt_money(portfolio["contracted_revenue"]),
        f'{portfolio["shipment_count"]:,} shipments analysed',
    )

with kpi_cols[1]:
    metric_card(
        "Revenue Recognised",
        fmt_money(portfolio["recognized_revenue"]),
        f'{fmt_money(portfolio["unrecognized_revenue"])} remains unrecognised',
    )

with kpi_cols[2]:
    metric_card(
        "Gross Margin",
        fmt_money(portfolio["gross_margin"]),
        fmt_pct(portfolio["gross_margin_pct_of_contracted"])
        + " of contracted revenue",
    )

with kpi_cols[3]:
    metric_card(
        "Immobilised Cargo",
        fmt_money(held["immobilised_cargo_value"]),
        f'{portfolio["held_shipments"]:,} held shipments · {fmt_money(held["trapped_contracted_revenue"])} trapped revenue',
    )

st.write("")

# -----------------------------------------------------------------------------
# 1. REVENUE -> COST -> MARGIN BRIDGE
# -----------------------------------------------------------------------------
section_header(
    "HOW DID THE PORTFOLIO GET HERE?",
    "The business starts with contracted revenue, loses recognition on held cargo, then absorbs a cost-to-serve base larger than recognised revenue.",
)

waterfall = go.Figure(
    go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total", "relative", "total"],
        x=[
            "Contracted Revenue",
            "Unrecognised Revenue",
            "Revenue Recognised",
            "Total Cost to Serve",
            "Gross Margin",
        ],
        y=[
            portfolio["contracted_revenue"] / 1e6,
            -portfolio["unrecognized_revenue"] / 1e6,
            0,
            -portfolio["total_cost"] / 1e6,
            0,
        ],
        text=[
            fmt_money(portfolio["contracted_revenue"]),
            f'-{fmt_money(portfolio["unrecognized_revenue"])}',
            fmt_money(portfolio["recognized_revenue"]),
            f'-{fmt_money(portfolio["total_cost"])}',
            fmt_money(portfolio["gross_margin"]),
        ],
        textposition="outside",
        connector={"line": {"color": "#475569", "width": 1}},
        increasing={"marker": {"color": "#60A5FA"}},
        decreasing={"marker": {"color": "#E63946"}},
        totals={"marker": {"color": "#CBD5E1"}},
        hovertemplate="<b>%{x}</b><br>$%{y:.1f}M<extra></extra>",
    )
)

waterfall.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=360,
    showlegend=False,
    margin=dict(l=10, r=30, t=15, b=10),
    yaxis_title="USD millions",
    xaxis=dict(tickfont=dict(size=11)),
)
waterfall.update_yaxes(zeroline=True, zerolinecolor="#334155", gridcolor="#1E293B")
st.plotly_chart(
    waterfall,
    use_container_width=True,
    config={"displayModeBar": False},
)

# -----------------------------------------------------------------------------
# 2. ROUTE ECONOMICS × SERVICE MATRIX
# -----------------------------------------------------------------------------
section_header(
    "WHERE IS THE ROUTE DAMAGE — AND DOES SERVICE JUSTIFY IT?",
    "Each bubble is a current route. X = modeled avoidable exposure, Y = observed DIFOT, size = shipment count. The 80% line is the service-compliance benchmark.",
)

route_view = cf_route.merge(
    route[["Route_Canonical", "DIFOT_Pct", "Cargo_Value_USD", "Contracted_Revenue_USD"]],
    on="Route_Canonical",
    how="left",
    suffixes=("", "_route"),
)
route_view = route_view[route_view["Route_Canonical"].notna()].copy()
route_view["Exposure_M"] = route_view["Avoidable_Exposure_USD"] / 1e6
route_view["Bubble_Size"] = route_view["Shipments"].clip(lower=1)

fig_route = go.Figure()

fig_route.add_trace(
    go.Scatter(
        x=route_view["Exposure_M"],
        y=route_view["DIFOT_Pct"],
        mode="markers+text",
        text=route_view["Route_Canonical"],
        textposition="top center",
        marker=dict(
            size=route_view["Bubble_Size"] * 2.3 + 10,
            color=route_view["DIFOT_Pct"],
            colorscale=[
                [0.0, "#E63946"],
                [0.8, "#F59E0B"],
                [1.0, "#22C55E"],
            ],
            cmin=0,
            cmax=100,
            line=dict(color="#0F172A", width=1.5),
            colorbar=dict(title="DIFOT", ticksuffix="%"),
            opacity=0.92,
        ),
        customdata=np.stack(
            [
                route_view["Shipments"],
                route_view["Contracted_Revenue_USD"],
                route_view["Residual_Exposure_USD"],
            ],
            axis=-1,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Avoidable exposure: $%{x:.1f}M<br>"
            "DIFOT: %{y:.1f}%<br>"
            "Shipments: %{customdata[0]:.0f}<br>"
            "Contracted revenue: $%{customdata[1]:,.0f}<br>"
            "Residual exposure: $%{customdata[2]:,.0f}<extra></extra>"
        ),
    )
)

fig_route.add_hline(
    y=80,
    line_dash="dash",
    line_color="#F59E0B",
    annotation_text="80% service threshold",
    annotation_position="top left",
)

fig_route.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=410,
    showlegend=False,
    margin=dict(l=10, r=10, t=25, b=10),
    xaxis_title="Modeled avoidable exposure (USD millions)",
    yaxis_title="Observed DIFOT",
)
fig_route.update_xaxes(gridcolor="#1E293B", zeroline=False)
fig_route.update_yaxes(
    gridcolor="#1E293B",
    ticksuffix="%",
    range=[-5, 108],
)

st.plotly_chart(
    fig_route,
    use_container_width=True,
    config={"displayModeBar": False},
)

# -----------------------------------------------------------------------------
# 3. CUSTOMER REVENUE SHARE × LOSS SHARE MATRIX
# -----------------------------------------------------------------------------
left, right = st.columns([1.2, 1])

with left:
    section_header(
        "WHO CARRIES THE COMMERCIAL EXPOSURE?",
        "Points above the diagonal are carrying a larger share of margin loss than their share of contracted revenue.",
    )

    cust_view = customers.copy()
    if len(cust_view) > 5:
        tail = cust_view.iloc[5:].sum(numeric_only=True)
        cust_view = cust_view.iloc[:5].copy()
        cust_view.loc[len(cust_view)] = {
            "Customer_Name": "All other customers",
            "Shipments": int(tail.get("Shipments", 0)),
            "Contracted_Revenue_USD": float(tail.get("Contracted_Revenue_USD", 0.0)),
            "Recognized_Revenue_USD": float(tail.get("Recognized_Revenue_USD", 0.0)),
            "Gross_Margin_USD": float(tail.get("Gross_Margin_USD", 0.0)),
            "DIFOT_Pct": np.nan,
            "Held_Shipments": int(tail.get("Held_Shipments", 0)),
            "Cargo_Value_USD": float(tail.get("Cargo_Value_USD", 0.0)),
            "Revenue_Share": float(tail.get("Revenue_Share", 0.0)),
            "Margin_Loss_USD": float(tail.get("Margin_Loss_USD", 0.0)),
            "Margin_Loss_Share": float(tail.get("Margin_Loss_Share", 0.0)),
        }
    cust_view["Cargo_Value_M"] = cust_view["Cargo_Value_USD"] / 1e6
    cust_view["Bubble_Size"] = np.sqrt(cust_view["Cargo_Value_M"].clip(lower=0)) * 4 + 8

    fig_customer = go.Figure()

    fig_customer.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max(cust_view["Revenue_Share"].max(), cust_view["Margin_Loss_Share"].max()) * 1.08,
        y1=max(cust_view["Revenue_Share"].max(), cust_view["Margin_Loss_Share"].max()) * 1.08,
        line=dict(color="#64748B", dash="dash", width=1),
    )

    fig_customer.add_trace(
        go.Scatter(
            x=cust_view["Revenue_Share"],
            y=cust_view["Margin_Loss_Share"],
            mode="markers+text",
            text=cust_view["Customer_Name"].str.replace(" Energy Partners", "", regex=False).str.replace(" Crude Traders", "", regex=False).str.replace(" Petrochem", "", regex=False).str.replace(" Fuel Alliance", "", regex=False).str.replace(" Refining Group", "", regex=False),
            textposition="top center",
            marker=dict(
                size=cust_view["Bubble_Size"],
                color=cust_view["DIFOT_Pct"].fillna(50),
                colorscale=[
                    [0.0, "#E63946"],
                    [0.8, "#F59E0B"],
                    [1.0, "#22C55E"],
                ],
                cmin=0,
                cmax=100,
                line=dict(color="#0F172A", width=1.5),
                opacity=0.9,
            ),
            customdata=np.stack(
                [
                    cust_view["Customer_Name"],
                    cust_view["DIFOT_Pct"].fillna(50),
                    cust_view["Held_Shipments"],
                    cust_view["Cargo_Value_M"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Revenue share: %{x:.1%}<br>"
                "Margin-loss share: %{y:.1%}<br>"
                "DIFOT: %{customdata[1]:.1f}%<br>"
                "Held shipments: %{customdata[2]:.0f}<br>"
                "Cargo value: $%{customdata[3]:.1f}M<extra></extra>"
            ),
        )
    )

    fig_customer.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=410,
        showlegend=False,
        margin=dict(l=10, r=10, t=15, b=10),
        xaxis_title="Contracted revenue share",
        yaxis_title="Portfolio margin-loss share",
    )
    fig_customer.update_xaxes(gridcolor="#1E293B", tickformat=".0%", range=[0, max(cust_view["Revenue_Share"].max(), 0.1) * 1.12])
    fig_customer.update_yaxes(gridcolor="#1E293B", tickformat=".0%", range=[0, max(cust_view["Margin_Loss_Share"].max(), 0.1) * 1.12])

    st.plotly_chart(
        fig_customer,
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    section_header(
        "HOW CONCENTRATED IS THE LOSS?",
        f'{loss["shipments_to_threshold"]} shipments account for approximately {loss["threshold_share"]:.0%} of observed portfolio loss.',
    )

    pareto = df[["Shipment_ID", "Gross_Margin_USD"]].copy()
    pareto["Loss_USD"] = (-pareto["Gross_Margin_USD"]).clip(lower=0)
    pareto = pareto.sort_values("Loss_USD", ascending=False).reset_index(drop=True)
    pareto["Rank"] = np.arange(1, len(pareto) + 1)
    total_loss = max(float(pareto["Loss_USD"].sum()), 1.0)
    pareto["Cumulative_Loss_Share"] = pareto["Loss_USD"].cumsum() / total_loss

    fig_pareto = go.Figure()
    fig_pareto.add_trace(
        go.Scatter(
            x=pareto["Rank"],
            y=pareto["Cumulative_Loss_Share"],
            mode="lines",
            line=dict(color="#E63946", width=3),
            fill="tozeroy",
            fillcolor="rgba(230,57,70,0.10)",
            hovertemplate="Top %{x:.0f} shipments<br>Cumulative loss: %{y:.1%}<extra></extra>",
        )
    )
    fig_pareto.add_hline(
        y=0.80,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text="80% of loss",
        annotation_position="top left",
    )
    threshold_rank = loss["shipments_to_threshold"]
    threshold_share = loss["threshold_share"]
    fig_pareto.add_trace(
        go.Scatter(
            x=[threshold_rank],
            y=[threshold_share],
            mode="markers+text",
            text=[f"{threshold_rank} shipments\n{threshold_rank / len(df):.1%} of book"],
            textposition="bottom right",
            marker=dict(size=11, color="#F59E0B", line=dict(color="#0F172A", width=1.5)),
            hovertemplate=f"<b>{threshold_rank} shipments</b><br>Book share: {threshold_rank / len(df):.1%}<br>Loss share: {threshold_share:.1%}<extra></extra>",
        )
    )

    fig_pareto.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=410,
        showlegend=False,
        margin=dict(l=10, r=10, t=15, b=10),
        xaxis_title="Shipments ranked by loss exposure",
        yaxis_title="Cumulative share of portfolio loss",
    )
    fig_pareto.update_xaxes(gridcolor="#1E293B", rangemode="tozero")
    fig_pareto.update_yaxes(gridcolor="#1E293B", tickformat=".0%", range=[0, 1.05])

    st.plotly_chart(
        fig_pareto,
        use_container_width=True,
        config={"displayModeBar": False},
    )

# -----------------------------------------------------------------------------
# 4. PRODUCT × ROUTE ECONOMICS HEATMAP
# -----------------------------------------------------------------------------
section_header(
    "WHICH ROUTE / PRODUCT COMBINATIONS BREAK THE ECONOMICS?",
    "Observed median cost per ton by product and route. The annotations pair cost with observed DIFOT so expensive service does not get confused with service value.",
)

heat = product_route.copy()
heat["Cost_Label"] = heat["Median_Cost_per_Ton_USD"].map(lambda x: f"${x:,.0f}")
heat["DIFOT_Label"] = heat["DIFOT_Pct"].map(lambda x: f"{x:.0f}%")
heat["Cell_Label"] = heat["Cost_Label"] + " / " + heat["DIFOT_Label"]

pivot = heat.pivot(index="Product_Category", columns="Route_Canonical", values="Median_Cost_per_Ton_USD")
labels = heat.pivot(index="Product_Category", columns="Route_Canonical", values="Cell_Label")

route_order = ["Direct", "Overland", "Cape", "Pipeline", "Air", "Held"]
route_order = [r for r in route_order if r in pivot.columns]
pivot = pivot.reindex(columns=route_order)
labels = labels.reindex(columns=route_order)

fig_heat = go.Figure(
    go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        text=labels.values,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=[
            [0.0, "#10233B"],
            [0.45, "#8B5E34"],
            [0.72, "#C2410C"],
            [1.0, "#E63946"],
        ],
        colorbar=dict(title="Median cost / ton (USD)"),
        hovertemplate=(
            "<b>%{y}</b> × <b>%{x}</b><br>"
            "Median cost/ton: $%{z:,.2f}<extra></extra>"
        ),
        hoverongaps=False,
    )
)

fig_heat.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=430,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Observed route",
    yaxis_title="Product category",
)

st.plotly_chart(
    fig_heat,
    use_container_width=True,
    config={"displayModeBar": False},
)

# -----------------------------------------------------------------------------
# MANAGEMENT RESPONSE
# -----------------------------------------------------------------------------
st.write("")
action_chips(["RELEASE", "REROUTE", "REPRICE", "SELECTIVE PROTECT"])
st.markdown("### Protect the customer, not the uneconomic route.")
st.caption(
    "Benchmark terminology: observed · modeled · service-compliant · not a causal forecast. "
    "Direct remains a pre-blockade benchmark, not a guaranteed post-blockade option."
)
