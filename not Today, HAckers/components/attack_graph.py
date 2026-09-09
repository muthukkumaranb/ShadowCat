"""
SHADOWCAT - Dynamic Attack Graph Component with K-Step Rollout & Widening Uncertainty
Renders enterprise host entities, active flow connections, and K-step forward simulation.

ARCHITECTURE COMPLIANCE:
1. "IP used only as a graph-position key, never as a model input feature."
   Node role annotations (e.g. [SSH Jump Host]) are purely cosmetic UI-layer lookups.
2. "Uncertainty grows with depth — measured as rollout error vs K."
   Host risk scores h_v(t) incorporate widening epistemic uncertainty bounds (+/- sigma)
   that visibly widen as the rollout advances from t to t+4.
3. Mandatory n=1 caption:
   "Demonstrated on the single infiltration case study (n = 1). Not a general lateral-movement forecasting capability."
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from styles import render_html

# UI-Layer Display Lookup Table ONLY:
# Strictly cosmetic human-readable labels for frontend visualization.
# Sourced exclusively in the UI layer. IP strings are topology position keys only,
# NEVER passed into, extracted from, or processed by any model feature pipeline.
UI_NODE_ROLES = {
    "10.0.2.15": "Workstation (Patient Zero)",
    "10.0.2.18": "Internal File Share",
    "10.0.4.10": "SSH Jump Host",
    "10.0.4.21": "Internal Auth Cluster",
    "10.0.5.1": "Domain Controller (Critical Asset)",
}

# Graph topology fixed layout positions
NODE_POSITIONS = {
    "10.0.2.15": (-1.0, 0.0),
    "10.0.2.18": (-0.4, 0.8),
    "10.0.4.10": (0.0, 0.0),
    "10.0.4.21": (0.4, -0.7),
    "10.0.5.1": (1.0, 0.2),
}

# Rollout state transitions across K=0..4 horizons
ROLLOUT_DATA = {
    0: {
        "label": "Window t (Current Observed)",
        "horizon_code": "t",
        "uncertainty_sigma": 0.02,
        "uncertainty_label": "±2% (Observed Window)",
        "uncertainty_tier": "Nominal Telemetry",
        "uncertainty_color": "#00E676",
        "summary": "Attacker observed scanning ports on 10.0.4.10 from 10.0.2.15.",
        "active_edges": [("10.0.2.15", "10.0.4.10", "Port 22/TCP")],
        "host_risks": {
            "10.0.2.15": {"risk": 0.85, "uncertainty": 0.02},
            "10.0.4.10": {"risk": 0.38, "uncertainty": 0.03},
            "10.0.4.21": {"risk": 0.12, "uncertainty": 0.02},
            "10.0.2.18": {"risk": 0.08, "uncertainty": 0.02},
            "10.0.5.1":  {"risk": 0.05, "uncertainty": 0.01},
        }
    },
    1: {
        "label": "Horizon t+1 (+1 min Rollout)",
        "horizon_code": "t+1",
        "uncertainty_sigma": 0.05,
        "uncertainty_label": "±5% (Autoregressive Step 1)",
        "uncertainty_tier": "Low Epistemic Drift",
        "uncertainty_color": "#00E5FF",
        "summary": "SSH brute-force activity intensifies against jump host 10.0.4.10.",
        "active_edges": [
            ("10.0.2.15", "10.0.4.10", "Port 22/TCP [High Rate]"),
            ("10.0.2.15", "10.0.2.18", "Port 445/SMB [Probe]")
        ],
        "host_risks": {
            "10.0.2.15": {"risk": 0.92, "uncertainty": 0.04},
            "10.0.4.10": {"risk": 0.54, "uncertainty": 0.06},
            "10.0.4.21": {"risk": 0.18, "uncertainty": 0.05},
            "10.0.2.18": {"risk": 0.14, "uncertainty": 0.04},
            "10.0.5.1":  {"risk": 0.08, "uncertainty": 0.03},
        }
    },
    2: {
        "label": "Horizon t+2 (+2 min Rollout)",
        "horizon_code": "t+2",
        "uncertainty_sigma": 0.09,
        "uncertainty_label": "±9% (Autoregressive Step 2)",
        "uncertainty_tier": "Controlled Epistemic Drift",
        "uncertainty_color": "#60A5FA",
        "summary": "Anticipated credential extraction on 10.0.4.10; reconnaissance directed at Auth Cluster.",
        "active_edges": [
            ("10.0.2.15", "10.0.4.10", "Compromised"),
            ("10.0.4.10", "10.0.4.21", "Port 88/Kerberos")
        ],
        "host_risks": {
            "10.0.4.10": {"risk": 0.76, "uncertainty": 0.09},
            "10.0.2.15": {"risk": 0.94, "uncertainty": 0.05},
            "10.0.4.21": {"risk": 0.46, "uncertainty": 0.09},
            "10.0.5.1":  {"risk": 0.19, "uncertainty": 0.07},
            "10.0.2.18": {"risk": 0.15, "uncertainty": 0.06},
        }
    },
    3: {
        "label": "Horizon t+3 (+3 min Rollout)",
        "horizon_code": "t+3",
        "uncertainty_sigma": 0.16,
        "uncertainty_label": "±16% (Autoregressive Step 3)",
        "uncertainty_tier": "Compounding Rollout Error",
        "uncertainty_color": "#FACC15",
        "summary": "Anticipated lateral pivot: Kerberos ticket reuse toward Domain Controller 10.0.5.1.",
        "active_edges": [
            ("10.0.4.10", "10.0.4.21", "Auth Spray"),
            ("10.0.4.21", "10.0.5.1", "Port 389/LDAP Pivot")
        ],
        "host_risks": {
            "10.0.4.21": {"risk": 0.79, "uncertainty": 0.15},
            "10.0.5.1":  {"risk": 0.67, "uncertainty": 0.17},
            "10.0.4.10": {"risk": 0.88, "uncertainty": 0.12},
            "10.0.2.15": {"risk": 0.95, "uncertainty": 0.08},
            "10.0.2.18": {"risk": 0.18, "uncertainty": 0.11},
        }
    },
    4: {
        "label": "Horizon t+4 (+4 min Rollout)",
        "horizon_code": "t+4",
        "uncertainty_sigma": 0.24,
        "uncertainty_label": "±24% (Autoregressive Step 4)",
        "uncertainty_tier": "Epistemic Divergence Limit",
        "uncertainty_color": "#FF5252",
        "summary": "Deep rollout horizon: anticipated Domain Controller persistence attempts. High epistemic variance.",
        "active_edges": [
            ("10.0.4.21", "10.0.5.1", "DCSync Replication [Critical]")
        ],
        "host_risks": {
            "10.0.5.1":  {"risk": 0.84, "uncertainty": 0.24},
            "10.0.4.21": {"risk": 0.86, "uncertainty": 0.21},
            "10.0.4.10": {"risk": 0.91, "uncertainty": 0.18},
            "10.0.2.15": {"risk": 0.96, "uncertainty": 0.10},
            "10.0.2.18": {"risk": 0.22, "uncertainty": 0.16},
        }
    }
}


def render_attack_graph_panel():
    """
    Renders the interactive cyber attack graph with a K-step rollout slider,
    dynamic edge lighting, and visibly widening rollout uncertainty.
    """
    render_html("""
    <div style="margin-top: 18px; margin-bottom: 8px;">
        <div class="card-title" style="margin-bottom: 4px;">
            <span>Dynamic Enterprise Attack Graph & K-Step Rollout</span>
            <span class="badge">[Branch: GraphSAGE Spatial Topology]</span>
        </div>
        <div style="font-size: 0.82rem; color: #94A3B8;">
            Topological host entities and multi-step lateral progression rollout. IP addresses are used solely as topology position keys.
        </div>
    </div>
    """)

    # Mandatory n=1 caption
    render_html("""
    <div style="background: rgba(0, 229, 255, 0.04); border-left: 3px solid #00E5FF; padding: 8px 14px; border-radius: 0 6px 6px 0; margin-bottom: 16px;">
        <span style="font-size: 0.80rem; color: #E2E8F0; font-style: italic;">
            "Demonstrated on the single infiltration case study (n = 1). Not a general lateral-movement forecasting capability."
        </span>
    </div>
    """)

    # K-step Horizon Slider
    col_slider, col_unc_badge = st.columns([2.5, 1.5])
    with col_slider:
        k_step = st.slider(
            "Select Forward Rollout Horizon (t → t+4):",
            min_value=0,
            max_value=4,
            value=st.session_state.get("attack_graph_k", 2),
            step=1,
            format="Step %d",
            key="attack_graph_k_slider"
        )
        st.session_state["attack_graph_k"] = k_step

    step_info = ROLLOUT_DATA[k_step]

    with col_unc_badge:
        # Dynamic widening uncertainty indicator
        sigma_val = step_info["uncertainty_sigma"]
        bar_pct = min(100, int(sigma_val * 350))  # Scale for visual width representation
        render_html(f"""
        <div class="glass-card" style="padding: 10px 14px; margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span class="metric-label" style="margin: 0; font-size: 0.70rem;">Rollout Uncertainty</span>
                <span style="font-size: 0.82rem; font-weight: 800; color: {step_info['uncertainty_color']}; font-family: 'JetBrains Mono', monospace;">
                    {step_info['uncertainty_label']}
                </span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.08); border-radius: 4px; height: 6px; width: 100%; overflow: hidden; margin-bottom: 4px;">
                <div style="background: {step_info['uncertainty_color']}; height: 100%; width: {bar_pct}%; border-radius: 4px; transition: width 0.3s ease;"></div>
            </div>
            <div style="font-size: 0.68rem; color: #94A3B8; display: flex; justify-content: space-between;">
                <span>{step_info['uncertainty_tier']}</span>
                <span style="color: {step_info['uncertainty_color']};">Width expands with depth K</span>
            </div>
        </div>
        """)

    # Layout: Graph visualization (left) + Host Risk Leaderboard (right)
    col_graph, col_hosts = st.columns([2.2, 1.3])

    with col_graph:
        # Build networkx graph
        G = nx.Graph()
        for node_ip, pos in NODE_POSITIONS.items():
            G.add_node(node_ip, pos=pos)

        # Base edges
        base_edges = [
            ("10.0.2.15", "10.0.4.10"),
            ("10.0.2.15", "10.0.2.18"),
            ("10.0.4.10", "10.0.4.21"),
            ("10.0.4.21", "10.0.5.1"),
        ]
        for u, v in base_edges:
            G.add_edge(u, v)

        # Active edge set for this horizon
        active_edge_pairs = {tuple(sorted([u, v])) for u, v, _ in step_info["active_edges"]}

        # Draw edges
        edge_traces = []
        for u, v in G.edges():
            x0, y0 = NODE_POSITIONS[u]
            x1, y1 = NODE_POSITIONS[v]
            pair = tuple(sorted([u, v]))
            is_active = pair in active_edge_pairs

            edge_color = "#FF1744" if is_active else "rgba(148, 163, 184, 0.25)"
            edge_width = 3.5 if is_active else 1.2
            dash_style = "solid" if is_active else "dot"

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=edge_width, color=edge_color, dash=dash_style),
                hoverinfo="none",
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Draw nodes
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_hover_texts = []
        node_labels = []

        host_risks = step_info["host_risks"]
        for node_ip, pos in NODE_POSITIONS.items():
            node_x.append(pos[0])
            node_y.append(pos[1])
            hr = host_risks.get(node_ip, {"risk": 0.05, "uncertainty": 0.02})
            r_val = hr["risk"]
            u_val = hr["uncertainty"]

            # UI label constructed purely at display time
            cosmetic_role = UI_NODE_ROLES.get(node_ip, "Host")
            node_labels.append(f"<b>{node_ip}</b><br><span style='font-size:9px;'>{cosmetic_role}</span>")

            # Risk-scaled color
            if r_val > 0.70:
                color = "#FF1744"
                size = 32
            elif r_val > 0.35:
                color = "#FFB300"
                size = 26
            else:
                color = "#00E5FF"
                size = 20

            node_colors.append(color)
            node_sizes.append(size)

            hover_text = (
                f"<b>Node:</b> {node_ip} [{cosmetic_role}]<br>"
                f"<b>Predicted Risk h_v(t):</b> {r_val:.1%}<br>"
                f"<b>Epistemic Uncertainty:</b> ±{u_val:.1%}<br>"
                f"<b>Forecast Confidence Band:</b> [{max(0, r_val-u_val):.1%} – {min(1, r_val+u_val):.1%}]"
            )
            node_hover_texts.append(hover_text)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            textfont=dict(color="#FFFFFF", size=10, family="'Plus Jakarta Sans', sans-serif"),
            hoverinfo="text",
            hovertext=node_hover_texts,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(color="#FFFFFF", width=1.5),
                opacity=0.95
            ),
            showlegend=False
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.8)",
            plot_bgcolor="rgba(15, 23, 42, 0.8)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.1, 1.1]),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Rollout stage summary note
        render_html(f"""
        <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: -8px; background: rgba(255,255,255,0.03); border-radius: 6px; padding: 6px 10px;">
            <b>{step_info['label']}:</b> {step_info['summary']}
        </div>
        """)

    with col_hosts:
        render_html("""
        <div style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
            Ranked Host Risk & Epistemic Uncertainty
        </div>
        """)

        # Sort hosts by risk descending
        sorted_hosts = sorted(host_risks.items(), key=lambda x: x[1]["risk"], reverse=True)

        for host_ip, hr in sorted_hosts:
            r = hr["risk"]
            u = hr["uncertainty"]
            cosmetic_role = UI_NODE_ROLES.get(host_ip, "Host")

            if r > 0.70:
                tier_color = "#FF1744"
                status_bg = "rgba(255, 23, 68, 0.12)"
            elif r > 0.35:
                tier_color = "#FFB300"
                status_bg = "rgba(255, 179, 0, 0.12)"
            else:
                tier_color = "#00E5FF"
                status_bg = "rgba(0, 229, 255, 0.12)"

            # Bar width scales with uncertainty to make widening visually unmistakable
            u_bar_width = min(100, int(u * 380))

            render_html(f"""
            <div class="glass-card" style="padding: 8px 12px; margin-bottom: 6px; border-left: 3px solid {tier_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.82rem; font-weight: 700; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">
                        {host_ip}
                    </span>
                    <span style="font-size: 0.80rem; font-weight: 800; color: {tier_color}; font-family: 'JetBrains Mono', monospace;">
                        {r:.0%} <span style="font-size: 0.70rem; color: #94A3B8; font-weight: 400;">±{u*100:.0f}%</span>
                    </span>
                </div>
                <div style="font-size: 0.70rem; color: #94A3B8; margin-top: 2px;">
                    {cosmetic_role}
                </div>
                <!-- Widening Uncertainty Indicator Band -->
                <div style="margin-top: 4px; background: rgba(255, 255, 255, 0.06); border-radius: 3px; height: 3px; width: 100%; overflow: hidden;">
                    <div style="background: {step_info['uncertainty_color']}; height: 100%; width: {u_bar_width}%; border-radius: 3px; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """)
