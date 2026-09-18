from __future__ import annotations

import streamlit as st

from src.components import action_chips, fmt_money, inject_css, metric_card, page_header, section_header
from src.data_loader import load_data, source_status, validate_data
from src.metrics import held_metrics, portfolio_metrics

st.set_page_config(page_title="Strait Outta Hormuz | War Room", page_icon="⚓", layout="wide", initial_sidebar_state="expanded")
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

st.sidebar.markdown("## STRAIT OUTTA HORMUZ")
st.sidebar.markdown("### WAR ROOM")
st.sidebar.caption("Quantiz'26 Round 2")
st.sidebar.divider()
st.sidebar.markdown("**Decision chain**")
st.sidebar.markdown("DATA → EVIDENCE → DIAGNOSIS → COUNTERFACTUAL → DECISION")
st.sidebar.divider()
st.sidebar.caption(f'{len(df):,} shipments analysed')

page_header("STRAIT OUTTA HORMUZ", "WAR ROOM CONTROL · Interactive shipment network decision system")
action_chips(["RELEASE", "REROUTE", "REPRICE", "SELECTIVE PROTECT"])

cols = st.columns(4)
with cols[0]: metric_card("Contracted Revenue", fmt_money(portfolio["contracted_revenue"]), "source dataset")
with cols[1]: metric_card("Gross Margin", fmt_money(portfolio["gross_margin"]), f'{portfolio["gross_margin_pct_of_contracted"]*100:.1f}% of contracted revenue')
with cols[2]: metric_card("Trapped Revenue", fmt_money(held["trapped_contracted_revenue"]), f'{held["held_shipments"]} held shipments')
with cols[3]: metric_card("Immobilised Cargo", fmt_money(held["immobilised_cargo_value"]), "cargo value in held rows")

st.write("")
section_header("WAR ROOM FLOW", "Use the four operating views to move from portfolio diagnosis to shipment-level action.")
flow = st.columns(4)
for col, title, text in zip(
    flow,
    ["01 · WHAT IS HAPPENING?", "02 · WHO IS EXPOSED?", "03 · WHAT SHOULD WE DO?", "04 · WHAT IF WE CHANGE IT?"],
    ["Executive War Dashboard", "Customer Command Panel", "Shipment Decision Simulator", "What-If Simulator"],
):
    with col:
        st.markdown(f"### {title}")
        st.caption(text)
        st.markdown("Use the page from the sidebar →")

st.divider()
st.markdown("### Protect the customer, not the uneconomic route.")
st.caption("243 shipment records loaded from Shipment_Data. Original Excel workbook is read-only from the application's perspective.")
