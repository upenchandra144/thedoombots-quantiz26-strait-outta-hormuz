from __future__ import annotations

import streamlit as st

from src.components import (
    action_chips,
    fmt_money,
    inject_css,
    metric_card,
    page_header,
    section_header,
)
from src.data_loader import load_data, source_status, validate_data
from src.metrics import DEFAULT_SERVICE_THRESHOLD, held_metrics, portfolio_metrics


st.set_page_config(
    page_title="Strait Outta Hormuz | War Room",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

status = source_status()
if not status["exists"]:
    st.error("Shipment dataset not found. Please check data/raw/.")
    st.stop()

try:
    df = load_data()
except Exception as exc:
    st.error(f"Unable to load the shipment dataset: {exc}")
    st.stop()

validation = validate_data(df)
if not validation["dataset_ready"]:
    st.error("Dataset validation failed. Review data/raw/ before continuing.")
    st.stop()

portfolio = portfolio_metrics(df)
held = held_metrics(df)
service_threshold_pct = int(round(DEFAULT_SERVICE_THRESHOLD * 100))

# -----------------------------------------------------------------------------
# SIDEBAR / SOURCE CONTROL
# -----------------------------------------------------------------------------
st.sidebar.markdown("## STRAIT OUTTA HORMUZ")
st.sidebar.markdown("### WAR ROOM")
st.sidebar.caption("Quantiz'26 Round 2")
st.sidebar.divider()

st.sidebar.markdown("**Decision chain**")
st.sidebar.markdown(
    "DATA → EVIDENCE → DIAGNOSIS → COUNTERFACTUAL → DECISION"
)
st.sidebar.divider()

st.sidebar.markdown("**Management service constraint**")
st.sidebar.markdown(f"### {service_threshold_pct}% DIFOT")
st.sidebar.caption(
    "Minimum observed service level used by the disruption-era actionable benchmark."
)

st.sidebar.divider()
st.sidebar.markdown("**Source validation**")
st.sidebar.caption(
    f"{validation['rows']:,} rows · "
    f"{validation['unique_shipment_ids']:,} unique shipments · "
    f"{validation['held_shipments']:,} held"
)

# -----------------------------------------------------------------------------
# PAGE HEADER
# -----------------------------------------------------------------------------
page_header(
    "STRAIT OUTTA HORMUZ",
    "MINIMISE THE LOSS, NOT THE DISRUPTION · Interactive shipment network decision system",
)

action_chips(["RELEASE", "REROUTE", "REPRICE", "SELECTIVE PROTECT"])

# -----------------------------------------------------------------------------
# EXECUTIVE FOUNDATION KPIs
# -----------------------------------------------------------------------------
cols = st.columns(4)
with cols[0]:
    metric_card(
        "Contracted Revenue",
        fmt_money(portfolio["contracted_revenue"]),
        f"{portfolio['shipment_count']:,} shipments analysed",
    )
with cols[1]:
    metric_card(
        "Gross Margin",
        fmt_money(portfolio["gross_margin"]),
        f"{portfolio['gross_margin_pct_of_contracted'] * 100:.1f}% of contracted revenue",
    )
with cols[2]:
    metric_card(
        "Trapped Revenue",
        fmt_money(held["trapped_contracted_revenue"]),
        f"{held['held_shipments']:,} held shipments",
    )
with cols[3]:
    metric_card(
        "Immobilised Cargo",
        fmt_money(held["immobilised_cargo_value"]),
        "cargo value in held rows",
    )

st.write("")

# -----------------------------------------------------------------------------
# MANAGEMENT FRAMEWORK STRIP
# -----------------------------------------------------------------------------
section_header(
    "EXECUTIVE DECISION FRAMEWORK",
    "The War Room does not optimise for disruption elimination. It identifies the portion of economic loss that is addressable while preserving the minimum service constraint.",
)

framework = st.columns(3)
with framework[0]:
    metric_card(
        "Service Constraint",
        f"≥ {service_threshold_pct}% DIFOT",
        "management threshold",
    )
with framework[1]:
    metric_card(
        "Actionable Benchmark",
        "Non-Direct / Non-Held",
        f"observed route · ≥ {service_threshold_pct}% DIFOT · ≥5 shipments",
    )
with framework[2]:
    metric_card(
        "Decision Principle",
        "Minimise Residual Loss",
        "protect customer economics, not uneconomic routes",
    )

# -----------------------------------------------------------------------------
# WAR ROOM FLOW
# -----------------------------------------------------------------------------
st.write("")
section_header(
    "WAR ROOM FLOW",
    "Move from portfolio diagnosis to customer prioritisation, shipment action, economic what-if testing and the final recovery strategy.",
)

flow = st.columns(5)
for col, title, text in zip(
    flow,
    [
        "01 · WHAT IS HAPPENING?",
        "02 · WHO IS EXPOSED?",
        "03 · WHAT SHOULD WE DO?",
        "04 · WHAT IF WE CHANGE IT?",
        "05 · WHAT IS THE TARGET STATE?",
    ],
    [
        "Executive War Dashboard",
        "Customer Command Panel",
        "Shipment Decision Simulator",
        "What-If Simulator",
        "Recovery Strategy & Target State",
    ],
):
    with col:
        st.markdown(f"### {title}")
        st.caption(text)
        st.markdown("Use the page from the sidebar →")

# -----------------------------------------------------------------------------
# READOUT
# -----------------------------------------------------------------------------
st.divider()
st.markdown("### Protect the customer, not the uneconomic route.")
st.caption(
    f"{len(df):,} shipment records loaded from Shipment_Data. "
    "Original Excel workbook is read-only from the application's perspective. "
    f"Direct is retained as a pre-blockade reference; it is not treated as guaranteed future capacity. "
    f"The actionable route framework uses a {service_threshold_pct}% DIFOT minimum."
)
