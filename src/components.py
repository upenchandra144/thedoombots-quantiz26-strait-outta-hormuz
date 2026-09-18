from __future__ import annotations

import html
from typing import Iterable

import streamlit as st


ACTION_META = {
    "RELEASE": ("#FCA5A5", "#450A0A"),
    "REROUTE": ("#FDE68A", "#451A03"),
    "REPRICE": ("#FDBA74", "#431407"),
    "SELECTIVE PROTECT": ("#93C5FD", "#172554"),
    "MONITOR / RETAIN": ("#86EFAC", "#052E16"),
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1500px; padding-top: 1.7rem; padding-bottom: 3rem; }
        .war-title { font-size: 2.5rem; font-weight: 850; letter-spacing: -0.04em; line-height: 1.05; }
        .war-subtitle { color: #94A3B8; font-size: 1rem; margin-top: .25rem; margin-bottom: 1.2rem; }
        .eyebrow { color: #94A3B8; font-size: .73rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .25rem; }
        .section-title { font-size: 1.22rem; font-weight: 800; letter-spacing: -.02em; margin: .4rem 0 .7rem; }
        .muted { color: #94A3B8; }
        .tiny { color: #64748B; font-size: .75rem; }
        .metric-card { background: #111827; border: 1px solid #263449; border-radius: 14px; padding: 1rem 1.05rem; min-height: 118px; }
        .metric-label { color: #94A3B8; font-size: .73rem; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }
        .metric-value { color: #F8FAFC; font-size: 1.65rem; font-weight: 850; letter-spacing: -.03em; margin-top: .22rem; }
        .metric-sub { color: #64748B; font-size: .77rem; margin-top: .15rem; }
        .decision-card { background: linear-gradient(135deg,#121B2D 0%,#0F172A 100%); border: 1px solid #334155; border-radius: 16px; padding: 1.25rem 1.3rem; }
        .decision-label { color: #94A3B8; font-size: .75rem; letter-spacing: .12em; font-weight: 750; }
        .decision-action { font-size: 2rem; font-weight: 900; margin-top: .35rem; letter-spacing: -.04em; }
        .chip-row { display:flex; flex-wrap:wrap; gap:.55rem; margin:.8rem 0 0; }
        .action-chip { display:inline-block; border:1px solid #334155; background:#111827; color:#E2E8F0; padding:.42rem .68rem; border-radius:999px; font-size:.74rem; font-weight:750; letter-spacing:.04em; }
        .warning-card { background:#23180B; border:1px solid #704B12; border-radius:12px; padding:.8rem 1rem; color:#FDE68A; margin-top:.5rem; }
        .info-card { background:#0F1A2C; border:1px solid #263A59; border-radius:12px; padding:.8rem 1rem; color:#CBD5E1; }
        .action-box { border-radius:12px; padding:.85rem 1rem; font-weight:800; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, eyebrow: str = "Quantiz'26 · Round 2") -> None:
    st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="war-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="war-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def section_header(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def metric_card(label: str, value: str, sub: str | None = None) -> None:
    sub_html = f'<div class="metric-sub">{html.escape(sub)}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def decision_card(action: str, explanation: Iterable[str], secondary_action: str | None = None) -> None:
    accent, bg = ACTION_META.get(action, ("#CBD5E1", "#0F172A"))
    st.markdown(
        f'<div class="decision-card"><div class="decision-label">RECOMMENDED MANAGEMENT ACTION</div>'
        f'<div class="decision-action" style="color:{accent}">{html.escape(action)}</div>',
        unsafe_allow_html=True,
    )
    if secondary_action:
        st.markdown(
            f'<div class="action-box" style="background:{bg};color:{accent};margin-top:.65rem">'
            f'FOLLOW-ON: {html.escape(secondary_action)}</div>',
            unsafe_allow_html=True,
        )
    for idx, reason in enumerate(explanation, 1):
        st.markdown(f"**{idx}.** {reason}")
    st.markdown("</div>", unsafe_allow_html=True)


def warning_card(text: str) -> None:
    st.markdown(f'<div class="warning-card">{html.escape(text)}</div>', unsafe_allow_html=True)


def info_card(text: str) -> None:
    st.markdown(f'<div class="info-card">{html.escape(text)}</div>', unsafe_allow_html=True)


def action_chips(actions: Iterable[str]) -> None:
    chips = "".join(f'<span class="action-chip">{html.escape(a)}</span>' for a in actions)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def fmt_money(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}${value/1e9:.{digits}f}B"
    if value >= 1e6:
        return f"{sign}${value/1e6:.{digits}f}M"
    if value >= 1e3:
        return f"{sign}${value/1e3:.{digits}f}K"
    return f"{sign}${value:,.0f}"


def fmt_pct(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"
