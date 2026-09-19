from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    action_chips,
    fmt_money,
    inject_css,
    info_card,
    metric_card,
    page_header,
    section_header,
    warning_card,
)
from src.data_loader import load_data
from src.metrics import (
    DEFAULT_SERVICE_THRESHOLD,
    counterfactual_metrics,
    held_metrics,
    portfolio_metrics,
    product_route_metrics,
)


st.set_page_config(
    page_title="Recovery Strategy | Hormuz War Room",
    page_icon="⚓",
    layout="wide",
)

inject_css()
df = load_data()

SERVICE_THRESHOLD = DEFAULT_SERVICE_THRESHOLD
portfolio = portfolio_metrics(df)
held = held_metrics(df)
shipment_cf, route_summary = counterfactual_metrics(
    df,
    service_threshold=SERVICE_THRESHOLD,
)

# -----------------------------------------------------------------------------
# Core exposure numbers
# -----------------------------------------------------------------------------
loss_exposure = abs(float(portfolio["gross_margin"]))
modeled_addressable = float(
    shipment_cf["Avoidable_Exposure_USD"]
    .dropna()
    .sum()
)
modeled_residual = float(
    shipment_cf["Residual_Exposure_USD"]
    .dropna()
    .sum()
)

# "Rerouting opportunity" deliberately excludes Held and Direct because:
# - Held is a release problem, not a route benchmark.
# - Direct is retained only as a pre-blockade historical reference.
rerouting_opportunity = float(
    route_summary.loc[
        ~route_summary["Route_Canonical"].isin(["Held", "Direct"]),
        "Avoidable_Exposure_USD",
    ].sum()
)
held_modeled_addressable = float(
    route_summary.loc[
        route_summary["Route_Canonical"].eq("Held"),
        "Avoidable_Exposure_USD",
    ].sum()
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
page_header(
    "RECOVERY STRATEGY & TARGET STATE",
    "Translate the modeled loss envelope into a concrete management playbook — without pretending benchmark exposure is guaranteed savings.",
)

action_chips(["RELEASE", "REROUTE", "REPRICE", "SELECTIVE PROTECT"])

# -----------------------------------------------------------------------------
# Executive target-state KPI strip
# -----------------------------------------------------------------------------
cols = st.columns(4)
with cols[0]:
    metric_card(
        "Portfolio Gross Margin Loss",
        fmt_money(loss_exposure),
        "current portfolio position",
    )
with cols[1]:
    metric_card(
        "Modeled Addressable",
        fmt_money(modeled_addressable),
        "under observed disruption-era benchmarks",
    )
with cols[2]:
    metric_card(
        "Modeled Residual",
        fmt_money(modeled_residual),
        "loss remaining after benchmark routing",
    )
with cols[3]:
    metric_card(
        "Rerouting Opportunity",
        fmt_money(rerouting_opportunity),
        "excludes Held and historical Direct",
    )

st.write("")

# -----------------------------------------------------------------------------
# Target-state message
# -----------------------------------------------------------------------------
section_header(
    "THE TARGET STATE",
    "The point is not to eliminate disruption. It is to reduce the portion of economic loss that the observed network gives management a credible route to address.",
)

target_cols = st.columns([1.35, 1.0])

with target_cols[0]:
    fig_target = go.Figure()
    fig_target.add_trace(
        go.Bar(
            x=["Modeled benchmark exposure"],
            y=[modeled_addressable / 1e6],
            name="Addressable",
            marker_color="#C58D2A",
            text=[fmt_money(modeled_addressable)],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Addressable: $%{y:.2f}M<extra></extra>",
        )
    )
    fig_target.add_trace(
        go.Bar(
            x=["Modeled benchmark exposure"],
            y=[modeled_residual / 1e6],
            name="Residual",
            marker_color="#B64032",
            text=[fmt_money(modeled_residual)],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Residual: $%{y:.2f}M<extra></extra>",
        )
    )
    fig_target.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=310,
        margin=dict(l=10, r=20, t=65, b=15),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.02),
        yaxis_title="USD millions",
        xaxis_title="",
        title="Modeled exposure: addressable vs residual",
    )
    fig_target.update_yaxes(gridcolor="#1E293B")
    st.plotly_chart(
        fig_target,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.caption(
        "The two values are summed across shipments for which the shared counterfactual framework produces an actionable benchmark. They are a modeled exposure decomposition, not a forecast of realised savings."
    )

with target_cols[1]:
    st.markdown(
        "### MINIMISE THE LOSS, NOT THE DISRUPTION"
    )

    addressable_share = (
        modeled_addressable / (modeled_addressable + modeled_residual)
        if (modeled_addressable + modeled_residual)
        else 0.0
    )

    st.markdown(
        f"**{fmt_money(modeled_addressable)}** of modeled benchmark exposure is addressable "
        f"under the observed disruption-era benchmark — approximately **{addressable_share:.0%}**."
    )

    st.write("")

    st.markdown(
        f"**{fmt_money(modeled_residual)}** remains after applying that benchmark. "
        "This is the portion of the loss that routing alone does not resolve."
    )

    st.write("")

    st.markdown(
        "The management response therefore splits into four jobs:"
    )

    st.markdown(
        "**RELEASE** what is trapped · "
        "**REROUTE** what has a credible service-compliant alternative · "
        "**REPRICE** what remains structurally uneconomic · "
        "**SELECTIVELY PROTECT** service where the observed premium is actually justified."
    )

# -----------------------------------------------------------------------------
# Route-level recovery map
# -----------------------------------------------------------------------------
section_header(
    "WHERE THE RECOVERY OPPORTUNITY SITS",
    "Current-route exposure decomposition using the same actionable counterfactual as the decision engine. Direct remains historical; Held is a release problem.",
)

plot_df = route_summary.copy()
plot_df["Route_Label"] = plot_df["Route_Canonical"].map(
    {
        "Direct": "Direct\n(historical)",
        "Held": "Held\n(release)",
        "Pipeline": "Pipeline",
        "Cape": "Cape",
        "Overland": "Overland",
        "Air": "Air",
    }
).fillna(plot_df["Route_Canonical"])
plot_df["Avoidable_M"] = plot_df["Avoidable_Exposure_USD"].fillna(0) / 1e6
plot_df["Residual_M"] = plot_df["Residual_Exposure_USD"].fillna(0) / 1e6
plot_df = plot_df.sort_values("Avoidable_M", ascending=True)

fig_route = go.Figure()
fig_route.add_trace(
    go.Bar(
        y=plot_df["Route_Label"],
        x=plot_df["Avoidable_M"],
        orientation="h",
        name="Addressable",
        marker_color="#C58D2A",
        text=[fmt_money(v * 1e6) if v >= 0.05 else "" for v in plot_df["Avoidable_M"]],
        textposition="inside",
        hovertemplate="%{y}<br>Addressable: $%{x:.2f}M<extra></extra>",
    )
)
fig_route.add_trace(
    go.Bar(
        y=plot_df["Route_Label"],
        x=plot_df["Residual_M"],
        orientation="h",
        name="Residual",
        marker_color="#B64032",
        text=[fmt_money(v * 1e6) if v >= 0.05 else "" for v in plot_df["Residual_M"]],
        textposition="inside",
        hovertemplate="%{y}<br>Residual: $%{x:.2f}M<extra></extra>",
    )
)
fig_route.update_layout(
    barmode="stack",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=390,
    margin=dict(l=10, r=25, t=20, b=10),
    xaxis_title="USD millions",
    yaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
fig_route.update_xaxes(gridcolor="#1E293B")

st.plotly_chart(
    fig_route,
    use_container_width=True,
    config={"displayModeBar": False},
)

# -----------------------------------------------------------------------------
# Key route insights
# -----------------------------------------------------------------------------
insight_cols = st.columns(3)

with insight_cols[0]:
    pipeline_row = route_summary.loc[
        route_summary["Route_Canonical"].eq("Pipeline")
    ]
    pipeline_avoidable = (
        float(pipeline_row["Avoidable_Exposure_USD"].iloc[0])
        if not pipeline_row.empty
        else 0.0
    )
    pipeline_residual = (
        float(pipeline_row["Residual_Exposure_USD"].iloc[0])
        if not pipeline_row.empty
        else 0.0
    )
    metric_card(
        "Pipeline = Primary Routing Lever",
        fmt_money(pipeline_avoidable),
        f"modeled addressable · {fmt_money(pipeline_residual)} residual",
    )

with insight_cols[1]:
    metric_card(
        "Held = Release Lever",
        fmt_money(held["trapped_contracted_revenue"]),
        f"trapped contracted revenue · {held["held_shipments"]} held shipments",
    )

with insight_cols[2]:
    refined_route = product_route_metrics(
        df,
        service_threshold=SERVICE_THRESHOLD,
    )
    refined_route = refined_route[
        refined_route["Product_Category"].eq("Refined Petrochemicals")
    ].copy()
    compliant_refined = refined_route[
        (~refined_route["Route_Canonical"].isin(["Direct", "Held"]))
        & (refined_route["DIFOT"] >= SERVICE_THRESHOLD - 1e-10)
        & refined_route["Shipments"].ge(5)
    ]
    metric_card(
        "Refined = Commercial Recovery",
        "No compliant reroute",
        "no observed non-Direct / non-Held route meets 80% DIFOT",
    )

# -----------------------------------------------------------------------------
# Management playbook
# -----------------------------------------------------------------------------
section_header(
    "MANAGEMENT PLAYBOOK",
    "A simple translation from modeled exposure into what management should investigate next.",
)

playbook = [
    (
        "RELEASE",
        "Unlock trapped cargo first.",
        f"{held['held_shipments']} Held shipments · {fmt_money(held['trapped_contracted_revenue'])} trapped contracted revenue.",
        f"The scenario engine treats release acceleration as a planning assumption, not a forecast of the physical release schedule.",
    ),
    (
        "REROUTE",
        "Move non-compliant / uneconomic flows to observed disruption-era alternatives.",
        f"Pipeline carries {fmt_money(pipeline_avoidable)} of modeled addressable exposure; crude's observed Cape benchmark is {80:.0f}% DIFOT.",
        "Rerouting is only treated as actionable when the observed product-route evidence clears the service constraint and evidence requirement.",
    ),
    (
        "REPRICE",
        "Recover economics where routing cannot solve the problem.",
        "Refined Petrochemicals has no observed non-Direct, non-Held route meeting 80% DIFOT with sufficient observations.",
        "Commercial recovery should be framed as repricing / renegotiation, not as a fabricated operational substitute.",
    ),
    (
        "SELECTIVE PROTECT",
        "Pay a premium only when the observed service advantage is documented.",
        "Premium service is not protected automatically; the route-level evidence must show an observable service advantage over the cheaper compliant benchmark.",
        "Where two routes both show 100% observed DIFOT, the data does not itself justify paying the premium solely for service.",
    ),
]

for action, title, evidence, note in playbook:
    left, middle, right = st.columns([0.9, 1.5, 1.7])
    with left:
        st.markdown(f"### {action}")
    with middle:
        st.markdown(f"**{title}**")
        st.caption(evidence)
    with right:
        st.caption(note)

# -----------------------------------------------------------------------------
# Commercial negotiation posture
# -----------------------------------------------------------------------------
section_header(
    "NEGOTIATION POSTURE",
    "The commercial message is different depending on whether an observed operating alternative exists.",
)

negotiation_df = pd.DataFrame(
    [
        {
            "Situation": "Compliant alternative exists",
            "Management ask": "Reroute + reset economics around observed benchmark cost/service",
            "Evidence": "Product × route median cost/ton + observed DIFOT",
        },
        {
            "Situation": "No compliant alternative",
            "Management ask": "Reprice / renegotiate the commercial terms",
            "Evidence": "No observed non-Direct / non-Held route clears 80% DIFOT",
        },
        {
            "Situation": "Premium service retained",
            "Management ask": "Recover / justify the service premium",
            "Evidence": "Premium route must show a documented observed service advantage",
        },
        {
            "Situation": "Cargo Held",
            "Management ask": "Prioritise release and contractual treatment",
            "Evidence": f"{held['held_shipments']} held shipments · {fmt_money(held['trapped_contracted_revenue'])} trapped contracted revenue",
        },
    ]
)

st.dataframe(
    negotiation_df,
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# Final target-state readout
# -----------------------------------------------------------------------------
st.write("")
info_card(
    f"TARGET STATE: minimise residual economic loss while preserving ≥ {SERVICE_THRESHOLD:.0%} DIFOT. "
    f"The current observed framework identifies {fmt_money(modeled_addressable)} of modeled addressable exposure and {fmt_money(modeled_residual)} of modeled residual exposure. "
)

st.caption(
    "Method note: Direct is a historical/pre-blockade reference, not guaranteed future capacity. The actionable benchmark excludes Direct and Held, requires observed DIFOT ≥ 80% and at least 5 observations, and selects the lowest observed median cost/ton among qualifying product-route combinations."
)
