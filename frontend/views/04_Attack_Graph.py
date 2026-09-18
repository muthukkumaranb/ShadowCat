"""
SHADOWCAT SOC Cockpit - Page 4: Attack Graph with Dynamic K-Step Rollout
Direct implementation of Stitch folder shadowcat_soc_attack_graph_with_dynamic_k_step_rollout.
Features Cytoscape.js interactive topology, GNN experimental caution banner, and K-step scrubber.
"""

import streamlit as st
from styles import TOKENS, render_html
import importlib
import components.cytoscape_attack_graph
importlib.reload(components.cytoscape_attack_graph)
from components.cytoscape_attack_graph import render_cytoscape_graph
from data_provider import get_host_risk_graph, get_fusion_experimental

def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    host_graph = get_host_risk_graph()
    fusion = get_fusion_experimental()

    if "attack_k_step" not in st.session_state:
        st.session_state.attack_k_step = 0
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = "svc-auth-master"

    curr_k = st.session_state.attack_k_step

    # TOP CONTROL BAR
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: {t['primary']}; font-size: 1.25rem;">🕸</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.125rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">
                    Attack Topology & Lateral Propagation Graph
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                <span class="soc-badge badge-neutral">Cluster: Alpha-01 [VPC-8812]</span>
                <span class="soc-badge badge-nominal">Force-Directed</span>
                <span class="soc-badge badge-neutral">Flow Vol</span>
            </div>
        </div>
    </div>
    """)

    # MANDATORY CAUTION BANNER (Experimental GNN Fusion)
    render_html(f"""
    <div class="soc-caution-banner">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.1rem; color: {t['tertiary']};">⚡</span>
            <div>
                <span class="soc-caution-title">EXPERIMENTAL — GRAPH FUSION SIGNAL (HELD BACK FROM PRIMARY FORECAST)</span>
                <span style="color: {t['text_muted']}; margin: 0 0.35rem;">//</span>
                <span style="color: {t['text_secondary']};">Secondary topological GNN signal calibrated for analyst review only. Not factored into automated containment playbooks.</span>
            </div>
        </div>
        <div class="soc-badge badge-caution" style="shrink: 0;">
            <span class="soc-pulse-dot" style="background:{t['tertiary']};"></span>
            EPISTEMIC SIGMA: 0.28
        </div>
    </div>
    """)

    # K-STEP SCRUBBER PANEL
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 0.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="soc-badge badge-nominal">HORIZON SCRUBBER [k = 0..5]</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_high']}; font-weight: 600;">
                    Temporal Graph Diffusion Simulation (Δt = 15m)
                </span>
            </div>
            <div class="soc-badge badge-critical">
                Mitigation Window: 34m 12s [TIGHTENING]
            </div>
        </div>
    </div>
    """)

    # Step buttons for Attack Graph
    k_steps_info = [
        {"k": 0, "title": "k = 0 [NOW]", "phase": "Initial Breach (1 Comp)"},
        {"k": 1, "title": "k = 1 [+15m]", "phase": "RPC Probe (1 Comp / 2 Targets)"},
        {"k": 2, "title": "k = 2 [+30m]", "phase": "Worker Fall (2 Comp)"},
        {"k": 3, "title": "k = 3 [+45m]", "phase": "IAM Siphon (3 Comp)"},
        {"k": 4, "title": "k = 4 [+60m]", "phase": "Audit Vault (4 Comp)"},
        {"k": 5, "title": "k = 5 [+75m]", "phase": "Cascade Fall (6+ Hosts)"},
    ]

    s_cols = st.columns(6)
    for i, s_col in enumerate(s_cols):
        with s_col:
            is_cur = (i == curr_k)
            if st.button(f"{k_steps_info[i]['title']}\n{k_steps_info[i]['phase']}", key=f"att_k_{i}", type="primary" if is_cur else "secondary", use_container_width=True):
                st.session_state.attack_k_step = i
                st.rerun()

    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem; padding: 0.6rem 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">
            <div>
                Active Step: <b style="color:{t['primary']}">k = {curr_k} (+{curr_k*15}m)</b> • 
                Diffusion P(t): <b style="color:{t['secondary']}">0.962</b> • 
                Compromised: <b style="color:{t['secondary']}">{(curr_k + 1)} / 19</b> • 
                Surge: <b style="color:{t['secondary']}">14.8 GB/s</b>
            </div>
            <span style="color: {t['tertiary']}; font-weight: 600;">PROPAGATION HORIZON: Single Enclave Pivot Contained</span>
        </div>
    </div>
    """)

    # 2-COLUMN VIEWPORT: Left (8 cols Cytoscape) | Right (4 cols Inspector)
    c_graph, c_inspector = st.columns([8, 4])

    with c_graph:
        render_cytoscape_graph(
            k_step=curr_k,
            selected_node_id=st.session_state.selected_node,
            height=660,
            theme=st.session_state.get("theme", "dark"),
        )

    with c_inspector:
        sel_node = st.selectbox(
            "Inspect Host Node",
            ["svc-auth-master", "ip-10-0-14-88", "analytics-agg-02", "audit-vault", "iam-sync-daemon", "db-shard-01", "edge-gw-02", "ext-asn4837-c2"],
            index=0,
        )
        st.session_state.selected_node = sel_node

        is_critical = sel_node in ["svc-auth-master", "ext-asn4837-c2", "ip-10-0-14-88"]
        status_text = "CRITICAL COMPROMISE" if is_critical else "NOMINAL MONITORING"
        badge_type = "badge-critical" if is_critical else "badge-nominal"

        render_html(f"""
        <div class="soc-card" style="padding: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">
                    Node Telemetry & Attribution
                </span>
                <span class="soc-badge {badge_type}">{status_text}</span>
            </div>
            <div class="soc-card-nested" style="margin-bottom: 0.75rem;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; color: {t['text_high']};">
                    {sel_node}
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_secondary']}; margin-top: 0.25rem;">
                    IP: <b style="color:{t['text_high']}">10.0.14.88</b> • VPC: <b style="color:{t['text_high']}">vpc-8812</b>
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: {t['text_muted']}; margin-top: 0.25rem;">
                    Role: Authentication Master Daemon
                </div>
                <div style="margin-top: 0.5rem; padding-top: 0.25rem; border-top: 1px solid {t['border']}; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['secondary']}; font-weight: 700;">
                    Attack State: Active Beaconing (T1071.001)
                </div>
            </div>

            <!-- Flow stats -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.75rem;">
                <div class="soc-card-nested">
                    <span class="soc-stat-label">Inbound Rate</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: {t['text_high']};">1.2 GB/s</div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.625rem; color: {t['primary']};">Nominal</span>
                </div>
                <div class="soc-card-nested">
                    <span class="soc-stat-label">Outbound Surge</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: {t['secondary']};">14.8 GB/s</div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.625rem; color: {t['secondary']};">+410% Surge</span>
                </div>
            </div>

            <!-- Causal Attribution Graph Weights -->
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; text-transform: uppercase; margin-bottom: 0.35rem; display: flex; justify-content: space-between;">
                <span>Causal Graph Weights</span>
                <span>SHAPLEY VAL</span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 0.35rem; margin-bottom: 0.75rem;">
                <div>
                    <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem;">
                        <span style="color:{t['text_secondary']}">Egress Burst Contribution</span>
                        <span style="color:{t['secondary']}; font-weight: 700;">64.2%</span>
                    </div>
                    <div style="width: 100%; height: 5px; background: {t['surface_highest']}; border-radius: 2px;">
                        <div style="width: 64.2%; height: 100%; background: {t['secondary']}; border-radius: 2px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem;">
                        <span style="color:{t['text_secondary']}">Unauthenticated Micro-RPC</span>
                        <span style="color:{t['tertiary']}; font-weight: 700;">21.8%</span>
                    </div>
                    <div style="width: 100%; height: 5px; background: {t['surface_highest']}; border-radius: 2px;">
                        <div style="width: 21.8%; height: 100%; background: {t['tertiary']}; border-radius: 2px;"></div>
                    </div>
                </div>
            </div>

            <div class="soc-card-nested" style="margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="soc-stat-label">MITRE Technique</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {t['primary']};">
                        T1071.001 - Web Protocols
                    </div>
                </div>
                <span class="soc-badge badge-critical" style="font-size: 0.625rem;">P(conf) = 0.94</span>
            </div>
        </div>
        """)

        if st.button("Sever Host Connections & Isolate", type="primary", use_container_width=True):
            st.success(f"SDN Isolation Policy applied: {sel_node} blocked on all VPC interfaces.")
        if st.button("Download PCAP Trace (4.2 MB)", use_container_width=True):
            st.info(f"Generated cryptographic PCAP forensic export for {sel_node}.")

if __name__ == "__main__":
    render_page()
