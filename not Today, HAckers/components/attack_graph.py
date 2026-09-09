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
from styles import render_html, COLORS
from data_provider import get_host_risk_graph, is_using_mock_data, get_mock_badge_html

try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except Exception:
    AGRAPH_AVAILABLE = False
    import plotly.graph_objects as go

# Graph topology fixed layout positions
NODE_POSITIONS = {
    "10.0.2.15": (-1.0, 0.0),
    "10.0.2.18": (-0.4, 0.8),
    "10.0.4.10": (0.0, 0.0),
    "10.0.4.21": (0.4, -0.7),
    "10.0.5.1": (1.0, 0.2),
}


def render_attack_graph_panel():
    """Renders the interactive force-directed attack graph and ranked host risk leaderboard."""
    mock_badge = get_mock_badge_html("host_risk_graph")

    render_html(f"""
    <div style="margin-top: 26px; margin-bottom: 12px;">
        <div class="card-title">
            <span>Dynamic Enterprise Attack Graph & Lateral Rollout {mock_badge}</span>
        </div>
        <div style="font-size: 0.80rem; color: #9AA7BD;">
            Interactive force-directed topology tracking multi-step lateral progression rollout across hosts.
        </div>
    </div>
    """)

    # Mandatory n=1 caption (preserved verbatim for compliance)
    render_html("""
    <div style="background: rgba(56, 189, 248, 0.05); border-left: 3px solid #38BDF8; padding: 8px 14px; border-radius: 0 6px 6px 0; margin-bottom: 14px;">
        <span style="font-size: 0.78rem; color: #E8EDF5; font-style: italic;">
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

    # Ingest data dynamically from data_provider
    graph_data = get_host_risk_graph(k_step=k_step)
    step_info = graph_data["rollout_steps"][k_step]
    UI_NODE_ROLES = graph_data["node_roles"]

    with col_unc_badge:
        sigma_val = step_info["uncertainty_sigma"]
        bar_pct = min(100, int(sigma_val * 350))
        render_html(f"""
        <div class="glass-card" style="padding: 10px 14px; margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span class="metric-label" style="margin: 0; font-size: 0.70rem;">Rollout Uncertainty {mock_badge}</span>
                <span style="font-size: 0.82rem; font-weight: 800; color: {step_info['uncertainty_color']}; font-family: 'JetBrains Mono', monospace;">
                    {step_info['uncertainty_label']}
                </span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.08); border-radius: 4px; height: 5px; width: 100%; overflow: hidden; margin-bottom: 4px;">
                <div style="background: {step_info['uncertainty_color']}; height: 100%; width: {bar_pct}%; border-radius: 4px; transition: width 0.3s ease;"></div>
            </div>
            <div style="font-size: 0.68rem; color: #9AA7BD; display: flex; justify-content: space-between;">
                <span>{step_info['uncertainty_tier']}</span>
                <span style="color: {step_info['uncertainty_color']};">Width expands with depth K</span>
            </div>
        </div>
        """)

    # Layout: Graph visualization (left) + Host Risk Leaderboard (right)
    col_graph, col_hosts = st.columns([2.2, 1.3])
    host_risks = step_info["host_risks"]
    selected_host_id = st.session_state.get("selected_graph_host", "10.0.4.10")

    with col_graph:
        if AGRAPH_AVAILABLE:
            nodes = []
            for node_ip in NODE_POSITIONS.keys():
                hr = host_risks.get(node_ip, {"risk": 0.05, "uncertainty": 0.02})
                r_val = hr["risk"]
                u_val = hr["uncertainty"]
                role = UI_NODE_ROLES.get(node_ip, "Host")
                is_selected = (node_ip == selected_host_id)

                if r_val > 0.70:
                    node_color = "#FF453A"  # High-visibility Crimson
                    border_color = "#FFFFFF" if is_selected else "#FF8A80"
                    node_size = 36
                elif r_val > 0.35:
                    node_color = "#FF9F0A"  # High-visibility Amber
                    border_color = "#FFFFFF" if is_selected else "#FFD54F"
                    node_size = 30
                else:
                    node_color = "#30D158"  # High-visibility Emerald
                    border_color = "#FFFFFF" if is_selected else "#81C784"
                    node_size = 24

                # Crisp, high-contrast label with dark halo outline
                clean_role = role.split('(')[0].strip()
                node_label = f"{node_ip}\n{clean_role}"

                nodes.append(Node(
                    id=node_ip,
                    label=node_label,
                    size=node_size,
                    color={
                        "background": node_color,
                        "border": border_color,
                        "highlight": {"background": node_color, "border": "#38BDF8"},
                        "hover": {"background": node_color, "border": "#FFFFFF"},
                    },
                    borderWidth=4 if is_selected else 2,
                    font={
                        "color": "#FFFFFF",
                        "size": 13,
                        "face": "Inter, -apple-system, sans-serif",
                        "strokeWidth": 4,
                        "strokeColor": "#0A0E17",
                        "bold": True,
                    },
                    title=f"{node_ip} [{role}]\nPredicted Risk: {r_val:.0%} (±{u_val:.0%})\nStatus: {'CRITICAL' if r_val > 0.70 else ('ELEVATED' if r_val > 0.35 else 'NORMAL')}"
                ))

            edges = []
            base_edges = [
                ("10.0.2.15", "10.0.4.10", "Port 22/TCP"),
                ("10.0.2.15", "10.0.2.18", "Port 445/SMB"),
                ("10.0.4.10", "10.0.4.21", "Port 88/Kerberos"),
                ("10.0.4.21", "10.0.5.1", "Port 389/LDAP"),
            ]
            active_pairs = {tuple(sorted([u, v])): lbl for u, v, lbl in step_info["active_edges"]}

            for u, v, default_lbl in base_edges:
                pair = tuple(sorted([u, v]))
                is_active = pair in active_pairs
                edge_lbl = active_pairs.get(pair, default_lbl)

                edge_color = "#FF453A" if is_active else "#334766"
                edge_width = 3.5 if is_active else 1.8

                edges.append(Edge(
                    source=u,
                    target=v,
                    label=edge_lbl if is_active else "",
                    color={"color": edge_color, "highlight": "#38BDF8"},
                    width=edge_width,
                    font={
                        "color": "#FFFFFF",
                        "size": 11,
                        "face": "JetBrains Mono, monospace",
                        "strokeWidth": 4,
                        "strokeColor": "#0A0E17",
                        "align": "horizontal",
                    }
                ))

            # Spacious viewport with smooth centering physics
            config = Config(
                width=720,
                height=480,
                directed=True,
                physics=True,
                nodeHighlightBehavior=True,
                highlightColor="#38BDF8",
                collapsible=False,
                minVelocity=0.75,
                maxVelocity=30,
            )

            clicked_node = agraph(nodes=nodes, edges=edges, config=config)
            if clicked_node:
                st.session_state["selected_graph_host"] = clicked_node
                selected_host_id = clicked_node

        else:
            G = nx.Graph()
            for node_ip, pos in NODE_POSITIONS.items():
                G.add_node(node_ip, pos=pos)
            for u, v in [("10.0.2.15", "10.0.4.10"), ("10.0.2.15", "10.0.2.18"), ("10.0.4.10", "10.0.4.21"), ("10.0.4.21", "10.0.5.1")]:
                G.add_edge(u, v)

            edge_traces = []
            active_pairs = {tuple(sorted([u, v])) for u, v, _ in step_info["active_edges"]}
            for u, v in G.edges():
                x0, y0 = NODE_POSITIONS[u]
                x1, y1 = NODE_POSITIONS[v]
                is_active = tuple(sorted([u, v])) in active_pairs
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                    line=dict(width=3 if is_active else 1.2, color="#E5484D" if is_active else "#22304A"),
                    hoverinfo="none", showlegend=False
                ))

            node_x = [pos[0] for pos in NODE_POSITIONS.values()]
            node_y = [pos[1] for pos in NODE_POSITIONS.values()]
            node_colors = ["#FF453A" if host_risks[ip]["risk"] > 0.70 else "#FF9F0A" if host_risks[ip]["risk"] > 0.35 else "#30D158" for ip in NODE_POSITIONS.keys()]
            node_sizes = [34 if host_risks[ip]["risk"] > 0.70 else 28 if host_risks[ip]["risk"] > 0.35 else 22 for ip in NODE_POSITIONS.keys()]
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                text=[f"<b>{ip}</b><br>{UI_NODE_ROLES[ip].split('(')[0].strip()}" for ip in NODE_POSITIONS.keys()],
                textposition="top center",
                textfont=dict(color="#FFFFFF", size=11, family="Inter, -apple-system, sans-serif"),
                marker=dict(size=node_sizes, color=node_colors, line=dict(color="#FFFFFF", width=2)),
                showlegend=False
            )
            fig = go.Figure(data=edge_traces + [node_trace])
            fig.update_layout(paper_bgcolor="#131b2c", plot_bgcolor="#131b2c", height=480, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        render_html(f"""
        <div style="font-size: 0.76rem; color: #9AA7BD; margin-top: 4px;">
            <b>{step_info['label']}:</b> {step_info['summary']}
        </div>
        """)

    with col_hosts:
        render_html(f"""
        <div style="font-size: 0.82rem; font-weight: 700; color: #E8EDF5; margin-bottom: 8px;">
            Host Threat Distribution {mock_badge}
        </div>
        """)

        sorted_hosts = sorted(host_risks.items(), key=lambda x: x[1]["risk"], reverse=True)

        for host_ip, hr in sorted_hosts:
            r = hr["risk"]
            u = hr["uncertainty"]
            role = UI_NODE_ROLES.get(host_ip, "Host")
            is_active_sel = (host_ip == selected_host_id)

            status_color = "#E5484D" if r > 0.70 else "#E0982B" if r > 0.35 else "#2FB872"
            border_left = f"3px solid {status_color}"
            bg_card = "#182338" if is_active_sel else "#131b2c"

            u_bar_width = min(100, int(u * 380))

            render_html(f"""
            <div style="background: {bg_card}; border: 1px solid #22304a; border-left: {border_left}; border-radius: 6px; padding: 7px 10px; margin-bottom: 5px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: #E8EDF5; font-family: 'JetBrains Mono', monospace;">
                        {host_ip}
                    </span>
                    <span style="font-size: 0.78rem; font-weight: 800; color: {status_color}; font-family: 'JetBrains Mono', monospace;">
                        {r:.0%} <span style="font-size: 0.68rem; color: #64708A; font-weight: 400;">±{u*100:.0f}%</span>
                    </span>
                </div>
                <div style="font-size: 0.68rem; color: #9AA7BD; margin-top: 1px;">
                    {role}
                </div>
                <div style="margin-top: 3px; background: rgba(255, 255, 255, 0.05); border-radius: 2px; height: 3px; width: 100%; overflow: hidden;">
                    <div style="background: {status_color}; height: 100%; width: {u_bar_width}%; border-radius: 2px;"></div>
                </div>
            </div>
            """)

    # Host Inspection Panel (if selected)
    if selected_host_id:
        sh_hr = host_risks.get(selected_host_id, {"risk": 0.50, "uncertainty": 0.05})
        sh_role = UI_NODE_ROLES.get(selected_host_id, "Host")
        render_html(f"""
        <div style="background: #131b2c; border: 1px solid #22304a; border-radius: 8px; padding: 12px 16px; margin-top: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div>
                    <span style="font-size: 0.85rem; font-weight: 700; color: #E8EDF5; font-family: 'JetBrains Mono', monospace;">
                        HOST TELEMETRY: {selected_host_id}
                    </span>
                    {mock_badge}
                    <span style="font-size: 0.74rem; color: #9AA7BD; margin-left: 8px;">
                        [{sh_role}]
                    </span>
                </div>
                <div style="font-size: 0.82rem; font-weight: 800; color: #38BDF8; font-family: 'JetBrains Mono', monospace;">
                    Risk h_v({step_info['horizon_code']}): {sh_hr['risk']:.0%} (±{sh_hr['uncertainty']*100:.0f}%)
                </div>
            </div>
            <div style="font-size: 0.74rem; color: #9AA7BD; line-height: 1.4;">
                <b>Driving Flow Indicators:</b> Asymmetric SYN packets, active lateral socket connections, repeated auth failure bursts.<br>
                <b>Containment Stance:</b> Illustrative analyst guidance — not a system recommendation: Consider applying pre-emptive access control filters on ingress interfaces to {selected_host_id}.
            </div>
        </div>
        """)
