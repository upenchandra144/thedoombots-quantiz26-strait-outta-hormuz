from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    decision_card,
    fmt_money,
    inject_css,
    metric_card,
    page_header,
    section_header,
    warning_card,
)
from src.data_loader import load_data
from src.decision_rules import decision_for_row
from src.metrics import customer_metrics, customer_product_metrics, customer_route_metrics


st.set_page_config(
    page_title="Customer Command Panel | Hormuz War Room",
    page_icon="⚓",
    layout="wide",
)
inject_css()
df = load_data()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def money_m(value: float) -> str:
    return f"${value / 1e6:.1f}M"


def add_plotly_base_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=30, t=30, b=45),
        height=height,
        font=dict(color="#CBD5E1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# -----------------------------------------------------------------------------
# Header + customer selector
# -----------------------------------------------------------------------------
page_header(
    "CUSTOMER COMMAND PANEL",
    "Turn customer exposure into a route, service and commercial action plan.",
)

customer_table = customer_metrics(df)
customers = customer_table["Customer_Name"].astype(str).tolist()

selector_col, status_col = st.columns([2.2, 1])
with selector_col:
    customer = st.selectbox(
        "Select customer",
        customers,
        index=0,
        label_visibility="collapsed",
    )
summary = customer_metrics(df, customer)

with status_col:
    exposure_label = "HIGH EXPOSURE" if (
        summary["revenue_share"] >= 0.20 or summary["margin_loss_share"] >= 0.20
    ) else "MANAGED EXPOSURE"
    st.markdown(
        f"<div class='info-card'><strong>{exposure_label}</strong><br>"
        f"{summary['shipments']} shipments · {summary['revenue_share']:.1%} of portfolio revenue · "
        f"{summary['margin_loss_share']:.1%} of portfolio margin loss</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Customer KPI strip
# -----------------------------------------------------------------------------
cols = st.columns(6)
kpis = [
    (
        "Contracted Revenue",
        fmt_money(summary["contracted_revenue"]),
        f"{summary['revenue_share']:.1%} of portfolio",
    ),
    (
        "Recognised Revenue",
        fmt_money(summary["recognized_revenue"]),
        f"{summary['contracted_revenue'] - summary['recognized_revenue']:,.0f} unrecognised",
    ),
    (
        "Gross Margin",
        fmt_money(summary["gross_margin"]),
        f"{summary['margin_loss_share']:.1%} of portfolio loss",
    ),
    ("DIFOT", f"{summary['difot_pct']:.1f}%", "service performance"),
    ("Held Shipments", f"{summary['held_shipments']}", "release exposure"),
    ("Cargo Value", fmt_money(summary["cargo_value"]), "customer cargo value"),
]
for col, (label, value, sub) in zip(cols, kpis):
    with col:
        metric_card(label, value, sub)


# -----------------------------------------------------------------------------
# 1. Commercial concentration + decision-engine action mix
# -----------------------------------------------------------------------------
st.write("")
left, right = st.columns([1.0, 1.25])

with left:
    section_header(
        "COMMERCIAL SHARE VS LOSS SHARE",
        "A customer deserves disproportionate attention when its share of margin loss materially exceeds its share of revenue.",
    )

    share_df = pd.DataFrame(
        {
            "Metric": ["Revenue share", "Margin-loss share"],
            "Share": [summary["revenue_share"], summary["margin_loss_share"]],
        }
    )
    fig_share = px.bar(
        share_df.sort_values("Share"),
        x="Share",
        y="Metric",
        orientation="h",
        text="Share",
        range_x=[0, max(1.0, summary["margin_loss_share"] * 1.25)],
    )
    fig_share.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
        marker_color=["#64748B", "#E63946"],
    )
    fig_share.update_xaxes(tickformat=".0%", title="Portfolio share", showgrid=False)
    fig_share.update_yaxes(title="")
    add_plotly_base_layout(fig_share, 250)
    st.plotly_chart(fig_share, use_container_width=True, config={"displayModeBar": False})

    imbalance = summary["margin_loss_share"] - summary["revenue_share"]
    if imbalance > 0:
        st.caption(
            f"Loss concentration is {imbalance:.1%} above revenue concentration for {customer}."
        )
    else:
        st.caption(
            f"Loss concentration is {abs(imbalance):.1%} below revenue concentration for {customer}."
        )

with right:
    section_header(
        "DECISION ENGINE SIGNAL — TOP LOSS-EXPOSED SHIPMENTS",
        "Shows the management actions produced by the shared shipment-level rules for this customer's largest loss exposures.",
    )

    customer_rows = df[df["Customer_Name"].eq(customer)].copy()
    customer_rows["Loss_Exposure_USD"] = customer_rows["Gross_Margin_USD"].clip(upper=0).abs()
    top_loss = customer_rows.sort_values(
        "Loss_Exposure_USD", ascending=False
    ).head(5).copy()

    action_records: list[dict[str, object]] = []
    for _, shipment_row in top_loss.iterrows():
        decision = decision_for_row(df, shipment_row)
        action_records.append(
            {
                "Shipment_ID": str(shipment_row["Shipment_ID"]),
                "Action": decision.primary_action,
                "Loss_Exposure_USD": float(shipment_row["Loss_Exposure_USD"]),
            }
        )

    action_df = pd.DataFrame(action_records)
    if not action_df.empty:
        action_summary = (
            action_df.groupby("Action", as_index=False)["Loss_Exposure_USD"]
            .sum()
            .sort_values("Loss_Exposure_USD", ascending=True)
        )
        fig_actions = px.bar(
            action_summary,
            x="Loss_Exposure_USD",
            y="Action",
            orientation="h",
            text="Loss_Exposure_USD",
            color="Action",
            color_discrete_map={
                "RELEASE": "#FCA5A5",
                "REROUTE": "#FDE68A",
                "REPRICE": "#FDBA74",
                "SELECTIVE PROTECT": "#93C5FD",
                "MONITOR / RETAIN": "#86EFAC",
            },
        )
        fig_actions.update_traces(texttemplate="$%{text:.2s}", textposition="outside")
        fig_actions.update_layout(showlegend=False)
        fig_actions.update_xaxes(title="Loss exposure represented", showgrid=False)
        fig_actions.update_yaxes(title="")
        add_plotly_base_layout(fig_actions, 250)
        st.plotly_chart(fig_actions, use_container_width=True, config={"displayModeBar": False})
        st.caption("Decision actions are rule-based; they are not ML predictions.")
    else:
        st.info("No loss-exposed shipments are available for this customer.")


# -----------------------------------------------------------------------------
# 2. Customer route battlefield
# -----------------------------------------------------------------------------
st.write("")
section_header(
    "WHERE IS THE CUSTOMER EXPOSURE?",
    "Route points combine financial damage and service performance. Larger bubbles represent contracted revenue.",
)

route = customer_route_metrics(df, customer).copy()
customer_work = df[df["Customer_Name"].eq(customer)].copy()
route_extra = (
    customer_work.groupby("Route_Canonical")
    .agg(
        Median_Cost_per_Ton_USD=("Cost_per_Ton_USD", "median"),
        Median_Margin_per_Ton_USD=("Margin_per_Ton_USD", "median"),
        Cargo_Value_USD=("Cargo_Value_USD", "sum"),
    )
    .reset_index()
)
route = route.merge(route_extra, on="Route_Canonical", how="left")

route_fig = px.scatter(
    route,
    x="DIFOT_Pct",
    y="Margin_Loss_USD",
    size="Contracted_Revenue_USD",
    color="Route_Canonical",
    hover_name="Route_Canonical",
    custom_data=[
        "Shipments",
        "Contracted_Revenue_USD",
        "DIFOT_Pct",
        "Held_Shipments",
        "Median_Cost_per_Ton_USD",
    ],
    size_max=42,
)
route_fig.update_traces(
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Margin loss: $%{y:,.0f}<br>"
        "DIFOT: %{x:.1f}%<br>"
        "Revenue: $%{customdata[1]:,.0f}<br>"
        "Shipments: %{customdata[0]}<br>"
        "Held: %{customdata[3]}<br>"
        "Median cost/ton: $%{customdata[4]:,.2f}<extra></extra>"
    )
)
route_fig.add_hline(
    y=0,
    line_width=1,
    line_dash="dot",
    line_color="#64748B",
)
route_fig.add_vline(
    x=80,
    line_width=1.5,
    line_dash="dash",
    line_color="#FDE68A",
    annotation_text="80% DIFOT",
    annotation_position="top right",
)
route_fig.update_xaxes(title="Observed DIFOT (%)", range=[0, 105], showgrid=False)
route_fig.update_yaxes(title="Margin loss (USD)", tickprefix="$", separatethousands=True)
add_plotly_base_layout(route_fig, 390)
st.plotly_chart(route_fig, use_container_width=True, config={"displayModeBar": False})


# -----------------------------------------------------------------------------
# 3. Product × route concentration heatmap
# -----------------------------------------------------------------------------
st.write("")
left, right = st.columns([1.15, 0.85])

with left:
    section_header(
        "PRODUCT × ROUTE LOSS CONCENTRATION",
        "Cells show modeled margin loss for this customer. This exposes whether the problem is product-specific, route-specific, or both.",
    )
    heat = route.merge(
        customer_work.groupby("Route_Canonical")
        .agg(Revenue_USD=("Contracted_Freight_Revenue_USD", "sum"))
        .reset_index(),
        on="Route_Canonical",
        how="left",
    )

    customer_product_route = (
        customer_work.groupby(["Product_Category", "Route_Canonical"], dropna=False)
        .agg(
            Margin_Loss_USD=("Gross_Margin_USD", lambda s: float(s.clip(upper=0).abs().sum())),
            Contracted_Revenue_USD=("Contracted_Freight_Revenue_USD", "sum"),
            DIFOT_Pct=("DIFOT_Flag", "mean"),
            Shipments=("Shipment_ID", "nunique"),
        )
        .reset_index()
    )
    customer_product_route["DIFOT_Pct"] *= 100

    if customer_product_route.empty:
        st.info("No product × route exposure is available.")
    else:
        pivot = customer_product_route.pivot(
            index="Product_Category",
            columns="Route_Canonical",
            values="Margin_Loss_USD",
        ).fillna(0)
        fig_heat = px.imshow(
            pivot,
            text_auto=".2s",
            aspect="auto",
            color_continuous_scale=["#111827", "#7F1D1D", "#EF4444"],
        )
        fig_heat.update_traces(
            texttemplate="%{z:$,.2s}",
            hovertemplate="Product: %{y}<br>Route: %{x}<br>Margin loss: $%{z:,.0f}<extra></extra>",
        )
        fig_heat.update_xaxes(title="Route")
        fig_heat.update_yaxes(title="Product")
        fig_heat.update_layout(coloraxis_colorbar_title="Margin loss")
        add_plotly_base_layout(fig_heat, 420)
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

with right:
    section_header(
        "CUSTOMER × PRODUCT EXPOSURE",
        "Financial exposure and service performance by product category.",
    )
    prod = customer_product_metrics(df, customer).copy()
    prod_display = prod[
        [
            "Product_Category",
            "Shipments",
            "Contracted_Revenue_USD",
            "Gross_Margin_USD",
            "DIFOT_Pct",
            "Held_Shipments",
            "Route_Concentration",
        ]
    ].copy()
    prod_display.columns = [
        "Product",
        "Shipments",
        "Revenue",
        "Gross Margin",
        "DIFOT",
        "Held",
        "Route Concentration",
    ]
    st.dataframe(
        prod_display.style.format(
            {
                "Revenue": "${:,.0f}",
                "Gross Margin": "${:,.0f}",
                "DIFOT": "{:.1f}%",
                "Route Concentration": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# 4. Route exposure table
# -----------------------------------------------------------------------------
st.write("")
section_header(
    "CUSTOMER × ROUTE BATTLEFIELD",
    "Sorted by financial loss exposure. This table is the audit trail behind the visuals above.",
)

route_display = route[
    [
        "Route_Canonical",
        "Shipments",
        "Contracted_Revenue_USD",
        "Gross_Margin_USD",
        "Margin_Loss_USD",
        "DIFOT_Pct",
        "Held_Shipments",
        "Median_Cost_per_Ton_USD",
    ]
].copy()
route_display.columns = [
    "Route",
    "Shipments",
    "Contracted Revenue",
    "Gross Margin",
    "Margin Loss",
    "DIFOT",
    "Held",
    "Median Cost/Ton",
]
st.dataframe(
    route_display.style.format(
        {
            "Contracted Revenue": "${:,.0f}",
            "Gross Margin": "${:,.0f}",
            "Margin Loss": "${:,.0f}",
            "DIFOT": "{:.1f}%",
            "Median Cost/Ton": "${:,.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# -----------------------------------------------------------------------------
# 5. Customer War Room response — explicit reasoning
# -----------------------------------------------------------------------------
st.write("")
section_header(
    "CUSTOMER WAR ROOM RESPONSE",
    "The action is anchored to the customer's highest-loss shipment and then explained using customer-level evidence.",
)

if not top_loss.empty:
    anchor = top_loss.iloc[0]
    anchor_decision = decision_for_row(df, anchor)

    # Build customer-specific evidence statements.
    dominant_route = route.iloc[0] if not route.empty else None
    primary_issue = dominant_route["Route_Canonical"] if dominant_route is not None else "—"
    primary_issue_loss = float(dominant_route["Margin_Loss_USD"]) if dominant_route is not None else 0.0
    primary_issue_difot = float(dominant_route["DIFOT_Pct"]) if dominant_route is not None else 0.0
    primary_issue_held = int(dominant_route["Held_Shipments"]) if dominant_route is not None else 0

    reasons: list[str] = []

    if anchor_decision.explanation:
        reasons.extend(anchor_decision.explanation[:3])

    reasons.append(
        f"{primary_issue} is the customer's largest route-level margin-loss exposure at {fmt_money(primary_issue_loss)}, with {primary_issue_difot:.1f}% DIFOT."
    )

    if primary_issue_held > 0:
        reasons.append(
            f"{primary_issue} includes {primary_issue_held} Held shipment(s), increasing the account's trapped-revenue and release exposure."
        )

    if summary["margin_loss_share"] > summary["revenue_share"]:
        reasons.append(
            f"The customer absorbs {summary['margin_loss_share']:.1%} of portfolio margin loss against {summary['revenue_share']:.1%} of revenue, indicating disproportionate financial exposure."
        )

    response_left, response_right = st.columns([1.0, 1.25])
    with response_left:
        decision_card(
            anchor_decision.primary_action,
            reasons[:5],
            secondary_action=anchor_decision.secondary_action,
        )
        st.caption(
            "This is an explainable rule-based action, not an ML prediction. Customer tenure is intentionally excluded from the decision logic."
        )

    with response_right:
        section_header("DECISION EVIDENCE", "Key numbers supporting the management response.")
        evidence_cols = st.columns(2)
        with evidence_cols[0]:
            metric_card(
                "Anchor Shipment",
                str(anchor_decision.shipment_id),
                f"{anchor['Product_Category']} · {anchor_decision.current_route}",
            )
            metric_card(
                "Modeled Avoidable Exposure",
                fmt_money(anchor_decision.avoidable_exposure),
                "observed benchmark framework",
            )
        with evidence_cols[1]:
            metric_card(
                "Best Service-Compliant Benchmark",
                anchor_decision.best_service_compliant_route or "None observed",
                (
                    f"{anchor_decision.best_service_compliant_difot_pct:.1f}% DIFOT · "
                    f"${anchor_decision.best_service_compliant_cost_per_ton:,.2f}/ton"
                    if anchor_decision.benchmark_available
                    else "No observed route meets 80% DIFOT"
                ),
            )
            metric_card(
                "Residual Exposure",
                fmt_money(anchor_decision.residual_exposure),
                "after applying the observed benchmark",
            )

    if anchor_decision.warnings:
        for warning in anchor_decision.warnings[:3]:
            warning_card(warning)

else:
    st.info("No loss-exposed shipment is available to anchor a customer action.")


# -----------------------------------------------------------------------------
# Held-cargo alert
# -----------------------------------------------------------------------------
if summary["held_shipments"] > 0:
    warning_card(
        f"{summary['held_shipments']} shipments are currently Held for {customer}. "
        f"That represents {fmt_money(summary['contracted_revenue'] - summary['recognized_revenue'])} of unrecognised customer revenue within this account. "
        "Release should be evaluated first for Held cargo under the shared decision rules."
    )

st.caption(
    "Benchmark terminology: observed · modeled · service-compliant · avoidable exposure · residual exposure. "
    "Direct remains a pre-blockade benchmark and is not represented as guaranteed future capacity."
)
