"""
SHADOWCAT SOC Cockpit - Page 4: Interaction Graph
Shows the flow topology of the active input (nodes, directed edges, flow and byte counts)
as built by backend.predict. No model scores individual hosts: the deployed models produce
one window-level probability, so this page shows no per-host risk or compromise state.
"""

import html

import pandas as pd
import streamlit as st
from styles import TOKENS, render_html
from data_provider import get_forecast_trajectory, get_graph_topology, get_mitre_data


def _dot(nodes, edges):
    lines = ['digraph G {', 'rankdir=LR; bgcolor="transparent";',
             'node [shape=box, style="rounded", fontname="Helvetica", fontsize=9, color="#7E8B9B", fontcolor="#A0AEC0"];',
             'edge [color="#4b5563", fontname="Helvetica", fontsize=8, fontcolor="#7E8B9B"];']
    for n in nodes:
        lines.append(f'"{n["id"]}" [label="{n["id"]}\\n{n.get("role", "")}"];')
    for e in edges:
        lines.append(f'"{e["source"]}" -> "{e["target"]}" [label="{e.get("flow_count", "")}"];')
    lines.append("}")
    return "\n".join(lines)


def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    topo = get_graph_topology() or {}
    nodes = topo.get("graph_nodes", [])
    edges = topo.get("graph_edges", [])
    fc = get_forecast_trajectory()

    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">
            Interaction Graph
        </span>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: {t['text_secondary']}; margin-top: 0.3rem; line-height: 1.5;">
            {len(nodes)} nodes and {len(edges)} directed edges (first 45 edges) for window
            <code>{html.escape(str(fc.get('window_id', '—')))}</code>. Edge label = flow count.
            For CSE-CIC-IDS2018 window input the nodes are pseudo-endpoints from the dataset's canonical edge lists
            (service ports and client buckets), because the CICFlowMeter CSVs carry no IP addresses.
            This is input topology only: no model in the deployed pipeline scores hosts, so no host is marked as
            compromised or at risk. The window-level onset probability is on the Forecast page.
        </div>
    </div>
    """)

    if not nodes:
        st.info("The active input produced no graph (no IP columns and no canonical edge list for its windows).")
        return

    st.graphviz_chart(_dot(nodes, edges), width="stretch")

    with st.expander("Edges"):
        st.dataframe(pd.DataFrame(edges), hide_index=True, width="stretch")
    with st.expander("Nodes"):
        st.dataframe(pd.DataFrame(nodes), hide_index=True, width="stretch")

    mitre = get_mitre_data()
    ms = mitre[0] if mitre else {}
    stage = (f"{html.escape(ms['tactic_name'])} ({ms['tactic_id']}) / {ms['technique_id']} {html.escape(ms['technique_full_name'])}, "
             f"stage head p = {ms['confidence']:.2f}") if ms.get("technique_id") else "No stage determined"
    render_html(f"""
    <div class="soc-card" style="margin-top: 1rem;">
        <div class="soc-section-title">ATT&amp;CK stage for this window (not per host)</div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: {t['text_secondary']}; margin-top: 0.3rem;">{stage}</div>
    </div>
    """)


if __name__ == "__main__":
    render_page()
