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
    threshold_sensitivity,
)


st.set_page_config(
    page_title="Executive War Dashboard | Hormuz War Room",
    page_icon="⚓",
    layout="wide",
)

inject_css()
df = load_data()

# -----------------------------------------------------------------------------
# Shared analytical framework
# -----------------------------------------------------------------------------
SERVICE_THRESHOLD = 0.80
SERVICE_THRESHOLD_PCT = SERVICE_THRESHOLD * 100.0

portfolio = portfolio_metrics(df)
held = held_metrics(df)
conc = concentration_metrics(df)
loss = loss_concentration(df)
cf_shipments, cf_route = counterfactual_metrics(
    df,
    service_threshold=SERVICE_THRESHOLD,
)
route = route_metrics(df)
customers = customer_metrics(df)
product_route = product_route_metrics(
    df,
    service_threshold=SERVICE_THRESHOLD,
)

# Counterfactual portfolio view.  These values come from the same
# disruption-era actionable benchmark used by the decision engine and
# scenario engine: non-Direct / non-Held, >=80% DIFOT, minimum 5 observations.
actionable_avoidable = float(
    cf_shipments["Avoidable_Exposure_USD"].sum(skipna=True)
)
actionable_residual = float(
    cf_shipments["Residual_Exposure_USD"].sum(skipna=True)
)

actionable_benchmarkable_shipments = int(
    cf_shipments["Actionable_Benchmark_Available"].sum()
)

actionable_revenue_coverage = float(
    cf_shipments.loc[
        cf_shipments["Actionable_Benchmark_Available"],
        "Contracted_Freight_Revenue_USD",
    ].sum()
    / max(float(portfolio["contracted_revenue"]), 1.0)
)

# -----------------------------------------------------------------------------
# PAGE HEADER
# -----------------------------------------------------------------------------
page_header(
    "MINIMISE THE LOSS, NOT THE DISRUPTION",
    "EXECUTIVE WAR ROOM · Isolate addressable loss, protect service, and stop paying to move uneconomic cargo.",
)

st.markdown(
    "<div class='info-card'><strong>Management objective:</strong> minimise residual economic loss while maintaining an "
    f"<strong>{SERVICE_THRESHOLD_PCT:.0f}% DIFOT</strong> service constraint. "
    "The actionable benchmark excludes Direct (pre-blockade reference only) and Held cargo, and requires at least five observed shipments.</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# EXECUTIVE KPI ROW
# -----------------------------------------------------------------------------
kpi_cols = st.columns(6)

with kpi_cols[0]:
    metric_card(
        "Contracted Revenue",
        fmt_money(portfolio["contracted_revenue"]),
        f'{portfolio["shipment_count"]:,} shipments analysed',
    )

with kpi_cols[1]:
    metric_card(
        "Current Gross Margin",
        fmt_money(portfolio["gross_margin"]),
        f'{fmt_pct(portfolio["gross_margin_pct_of_contracted"])} of contracted revenue',
    )

with kpi_cols[2]:
    metric_card(
        "Modeled Avoidable Loss",
        fmt_money(actionable_avoidable),
        "disruption-era actionable benchmark",
    )

with kpi_cols[3]:
    metric_card(
        "Modeled Residual Loss",
        fmt_money(actionable_residual),
        "after applying actionable benchmark",
    )

with kpi_cols[4]:
    metric_card(
        "Immobilised Cargo",
        fmt_money(held["immobilised_cargo_value"]),
        f'{portfolio["held_shipments"]:,} held · {fmt_money(held["trapped_contracted_revenue"])} trapped revenue',
    )

with kpi_cols[5]:
    metric_card(
        "Portfolio DIFOT",
        f'{portfolio["difot_pct"]:.1f}%',
        f'{SERVICE_THRESHOLD_PCT:.0f}% management service constraint',
    )

st.caption(
    f"Actionable benchmark coverage: {actionable_benchmarkable_shipments:,} of {len(df):,} shipments "
    f"({actionable_benchmarkable_shipments / max(len(df), 1):.1%}) and "
    f"{actionable_revenue_coverage:.1%} of contracted revenue. Avoidable and residual exposure are benchmarked only where an observed actionable route exists."
)

# -----------------------------------------------------------------------------
# 1. LOSS POSITION
# -----------------------------------------------------------------------------
section_header(
    "THE LOSS HAS TWO PARTS: ADDRESSABLE AND RESIDUAL",
    "The current portfolio loss is not assumed to be fully recoverable. The benchmark framework isolates the portion that can be addressed by observed disruption-era route economics, while the remainder stays as residual modeled loss.",
)

loss_fig = go.Figure()
loss_fig.add_trace(
    go.Bar(
        x=["Current portfolio loss", "Modeled avoidable loss", "Modeled residual loss"],
        y=[
            abs(float(portfolio["gross_margin"])) / 1e6,
            actionable_avoidable / 1e6,
            actionable_residual / 1e6,
        ],
        text=[
            fmt_money(abs(float(portfolio["gross_margin"]))),
            fmt_money(actionable_avoidable),
            fmt_money(actionable_residual),
        ],
        textposition="outside",
        marker_color=["#E63946", "#F59E0B", "#64748B"],
        hovertemplate="<b>%{x}</b><br>$%{y:.1f}M<extra></extra>",
    )
)
loss_fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=350,
    margin=dict(l=10, r=20, t=15, b=40),
    showlegend=False,
    yaxis_title="USD millions",
    xaxis_title="",
)
loss_fig.update_yaxes(gridcolor="#1E293B", zeroline=False)
st.plotly_chart(
    loss_fig,
    use_container_width=True,
    config={"displayModeBar": False},
)

st.caption(
    "The historical/pre-blockade benchmark may use Direct as a reference. The actionable view is deliberately stricter because Direct is not treated as guaranteed future capacity."
)

# -----------------------------------------------------------------------------
# 2. REVENUE -> COST -> MARGIN BRIDGE
# -----------------------------------------------------------------------------
section_header(
    "HOW DID THE PORTFOLIO GET HERE?",
    "Contracted revenue is reduced by unrecognised revenue, while total cost-to-serve materially exceeds recognised revenue.",
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
    height=350,
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
# 3. ACTIONABLE ROUTE LANDSCAPE
# -----------------------------------------------------------------------------
section_header(
    "WHERE CAN WE ACTUALLY REDUCE THE LOSS?",
    "Only disruption-era transport options are plotted here. X = modeled avoidable exposure, Y = observed DIFOT, size = shipments. The 80% line is the shared service constraint.",
)

actionable_route_view = cf_route.merge(
    route[
        [
            "Route_Canonical",
            "DIFOT_Pct",
            "Cargo_Value_USD",
            "Contracted_Revenue_USD",
        ]
    ],
    on="Route_Canonical",
    how="left",
    suffixes=("", "_route"),
)
actionable_route_view = actionable_route_view[
    ~actionable_route_view["Route_Canonical"].isin(["Direct", "Held"])
].copy()
actionable_route_view = actionable_route_view[
    actionable_route_view["Avoidable_Exposure_USD"].notna()
].copy()
actionable_route_view["Exposure_M"] = (
    actionable_route_view["Avoidable_Exposure_USD"] / 1e6
)

if actionable_route_view.empty:
    st.info("No actionable route-level counterfactual is available in the observed dataset.")
else:
    fig_route = go.Figure()
    fig_route.add_trace(
        go.Scatter(
            x=actionable_route_view["Exposure_M"],
            y=actionable_route_view["DIFOT_Pct"],
            mode="markers+text",
            text=actionable_route_view["Route_Canonical"],
            textposition="top center",
            marker=dict(
                size=actionable_route_view["Shipments"].clip(lower=1) * 2.1 + 10,
                color=actionable_route_view["DIFOT_Pct"],
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
                    actionable_route_view["Shipments"],
                    actionable_route_view["Contracted_Revenue_USD"],
                    actionable_route_view["Residual_Exposure_USD"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Modeled avoidable exposure: $%{x:.1f}M<br>"
                "Observed DIFOT: %{y:.1f}%<br>"
                "Shipments: %{customdata[0]:.0f}<br>"
                "Contracted revenue: $%{customdata[1]:,.0f}<br>"
                "Residual exposure: $%{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )

    fig_route.add_hline(
        y=SERVICE_THRESHOLD_PCT,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"{SERVICE_THRESHOLD_PCT:.0f}% service constraint",
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
    fig_route.update_yaxes(gridcolor="#1E293B", ticksuffix="%", range=[-5, 108])
    st.plotly_chart(
        fig_route,
        use_container_width=True,
        config={"displayModeBar": False},
    )

st.caption(
    "Direct is intentionally absent from this action map: it is a pre-blockade historical reference, not a guaranteed disruption-era route. Held cargo is handled separately as a release priority."
)

# -----------------------------------------------------------------------------
# 4. CUSTOMER REVENUE SHARE × LOSS SHARE MATRIX
# -----------------------------------------------------------------------------
left, right = st.columns([1.2, 1])

with left:
    section_header(
        "WHO CARRIES THE COMMERCIAL EXPOSURE?",
        "Points above the diagonal carry a larger share of portfolio margin loss than their share of contracted revenue.",
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
    max_axis = max(
        float(cust_view["Revenue_Share"].max()),
        float(cust_view["Margin_Loss_Share"].max()),
        0.1,
    ) * 1.12

    fig_customer.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_axis,
        y1=max_axis,
        line=dict(color="#64748B", dash="dash", width=1),
    )

    fig_customer.add_trace(
        go.Scatter(
            x=cust_view["Revenue_Share"],
            y=cust_view["Margin_Loss_Share"],
            mode="markers+text",
            text=(
                cust_view["Customer_Name"]
                .str.replace(" Energy Partners", "", regex=False)
                .str.replace(" Crude Traders", "", regex=False)
                .str.replace(" Petrochem", "", regex=False)
                .str.replace(" Fuel Alliance", "", regex=False)
                .str.replace(" Refining Group", "", regex=False)
            ),
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
    fig_customer.update_xaxes(gridcolor="#1E293B", tickformat=".0%", range=[0, max_axis])
    fig_customer.update_yaxes(gridcolor="#1E293B", tickformat=".0%", range=[0, max_axis])
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
    threshold_rank = int(loss["shipments_to_threshold"])
    threshold_share = float(loss["threshold_share"])
    fig_pareto.add_trace(
        go.Scatter(
            x=[threshold_rank],
            y=[threshold_share],
            mode="markers+text",
            text=[f"{threshold_rank} shipments\n{threshold_rank / len(df):.1%} of book"],
            textposition="bottom right",
            marker=dict(size=11, color="#F59E0B", line=dict(color="#0F172A", width=1.5)),
            hovertemplate=(
                f"<b>{threshold_rank} shipments</b><br>"
                f"Book share: {threshold_rank / len(df):.1%}<br>"
                f"Loss share: {threshold_share:.1%}<extra></extra>"
            ),
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
# 5. WHY 80%? THRESHOLD SENSITIVITY
# -----------------------------------------------------------------------------
section_header(
    "WHY KEEP 80% AS THE SERVICE CONSTRAINT?",
    "Threshold sensitivity shows where observed disruption-era alternatives disappear. 80% is retained as the management constraint: 70% admits the 70%-DIFOT Refined→Cape option, 75% removes it while retaining Crude→Cape, and 81% removes the 80%-DIFOT Crude→Cape alternative.",
)

thresholds_to_show = [0.70, 0.75, 0.80, 0.81, 0.85, 0.90]
sensitivity = threshold_sensitivity(df, thresholds=thresholds_to_show)

left, right = st.columns([1.25, 0.9])
with left:
    fig_threshold = go.Figure()
    fig_threshold.add_trace(
        go.Scatter(
            x=sensitivity["Threshold"] * 100,
            y=sensitivity["Revenue_Coverage_Pct"],
            mode="lines+markers",
            name="Revenue coverage",
            line=dict(color="#60A5FA", width=3),
            marker=dict(size=8),
            hovertemplate="Threshold: %{x:.0f}%<br>Revenue coverage: %{y:.1f}%<extra></extra>",
        )
    )
    fig_threshold.add_trace(
        go.Scatter(
            x=sensitivity["Threshold"] * 100,
            y=sensitivity["Addressable_Loss_M"],
            mode="lines+markers",
            name="Addressable loss ($M)",
            yaxis="y2",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=8),
            hovertemplate="Threshold: %{x:.0f}%<br>Addressable loss: $%{y:.1f}M<extra></extra>",
        )
    )
    fig_threshold.add_vline(
        x=SERVICE_THRESHOLD_PCT,
        line_dash="dash",
        line_color="#FDE68A",
        annotation_text="80%",
        annotation_position="top left",
    )
    fig_threshold.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        margin=dict(l=10, r=10, t=15, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Observed DIFOT threshold",
        yaxis=dict(title="Revenue coverage (%)", ticksuffix="%", gridcolor="#1E293B"),
        yaxis2=dict(
            title="Addressable loss (USD M)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )
    st.plotly_chart(
        fig_threshold,
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    display = sensitivity.copy()
    display["Threshold"] = display["Threshold"] * 100
    display = display[
        [
            "Threshold",
            "Products_Covered",
            "Shipment_Coverage_Pct",
            "Revenue_Coverage_Pct",
            "Addressable_Loss_USD",
            "Residual_Loss_USD",
        ]
    ]
    display.columns = [
        "Threshold",
        "Products",
        "Shipments Covered",
        "Revenue Covered",
        "Addressable Loss",
        "Residual Loss",
    ]
    st.dataframe(
        display.style.format(
            {
                "Threshold": "{:.0f}%",
                "Shipments Covered": "{:.1f}%",
                "Revenue Covered": "{:.1f}%",
                "Addressable Loss": "${:,.0f}",
                "Residual Loss": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "This is sensitivity analysis, not a claim that 80% is a universal industry standard. "
        "The management constraint is retained because it preserves the key Crude→Cape alternative while excluding the 70%-DIFOT Refined→Cape option."
    )

# -----------------------------------------------------------------------------
# 6. PRODUCT × ROUTE ECONOMICS HEATMAP
# -----------------------------------------------------------------------------
section_header(
    "WHICH PRODUCT × ROUTE COMBINATIONS BREAK THE ECONOMICS?",
    "Observed median cost per ton by product and route. Cell labels pair cost with observed DIFOT; route-role notes prevent historical Direct and Held from being misread as interchangeable options.",
)

heat = product_route.copy()
heat["Cost_Label"] = heat["Median_Cost_per_Ton_USD"].map(lambda x: f"${x:,.0f}")
heat["DIFOT_Label"] = heat["DIFOT_Pct"].map(lambda x: f"{x:.0f}%")
heat["Cell_Label"] = heat["Cost_Label"] + " / " + heat["DIFOT_Label"]

pivot = heat.pivot(
    index="Product_Category",
    columns="Route_Canonical",
    values="Median_Cost_per_Ton_USD",
)
labels = heat.pivot(
    index="Product_Category",
    columns="Route_Canonical",
    values="Cell_Label",
)

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
st.markdown("### Protect the economics, then protect the service.")
st.caption(
    "Shared framework: 80% DIFOT management constraint · Direct = historical reference · Held = release priority · "
    "actionable benchmark = cheapest observed non-Direct/non-Held route meeting the constraint with at least five observations · "
    "no compliant observed alternative = reprice / negotiate rather than force an uneconomic reroute."
)
