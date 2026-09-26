"""
Cytoscape.js Interactive Attack Graph Component for Streamlit
Matches the visual spec in Stitch folder shadowcat_soc_attack_graph_with_dynamic_k_step_rollout.
Features dynamic K-step rollout diffusion, visual host isolation, live progression arrows, and node inspection.
"""

import json
from typing import Dict, Any, Optional, List
import streamlit.components.v1 as components


def render_cytoscape_graph(
    k_step: int = 0,
    selected_node_id: Optional[str] = None,
    isolated_nodes: Optional[List[str]] = None,
    custom_nodes: Optional[List[Dict[str, Any]]] = None,
    custom_edges: Optional[List[Dict[str, Any]]] = None,
    traversal_data: Optional[Dict[str, Any]] = None,
    step_risk: float = 0.5,
    height: int = 680,
    theme: str = "dark",
) -> None:
    """
    Renders an interactive Cytoscape.js attack topology graph driven by real graph-propagation traversal.
    k_step drives dynamic propagation state across k=0..5 horizons.
    step_risk modulates empirical compromise status (no fabrication when nominal).
    isolated_nodes renders visual SDN isolation and severed edge cuts.
    custom_nodes & custom_edges dynamically render live ingested hosts.
    traversal_data provides explainable, weight-based frontier expansion over real edges.
    """
    if isolated_nodes is None:
        isolated_nodes = []

    is_light = (theme == "light")
    bg_color = "#f8fafc" if is_light else "#0b0e13"
    grid_color = "rgba(148, 163, 184, 0.25)" if is_light else "rgba(59, 74, 61, 0.15)"
    panel_bg = "rgba(255, 255, 255, 0.96)" if is_light else "rgba(25, 28, 33, 0.94)"
    panel_border = "#cbd5e1" if is_light else "#1E2633"
    btn_bg = "#ffffff" if is_light else "#272a30"
    btn_border = "#cbd5e1" if is_light else "#3b4a3d"
    btn_color = "#0f172a" if is_light else "#f5fff2"
    btn_hover_bg = "#00A84D" if is_light else "#39ff88"
    btn_hover_color = "#ffffff" if is_light else "#003918"
    text_high = "#0f172a" if is_light else "#F3F4F6"
    text_muted = "#64748b" if is_light else "#A0AEC0"
    text_subtle = "#94a3b8" if is_light else "#7E8B9B"
    text_secondary = "#334155" if is_light else "#e1e2ea"
    primary_sig = "#00A84D" if is_light else "#39FF88"
    secondary_sig = "#D61B3C" if is_light else "#FF3B5C"
    tertiary_sig = "#D97706" if is_light else "#FFB84D"

    # Fail visibly if no real graph topology exists for this input
    if not custom_nodes or len(custom_nodes) == 0:
        empty_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    background: {bg_color};
                    color: {text_high};
                    font-family: 'Inter', -apple-system, sans-serif;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: {height}px;
                }}
                .empty-card {{
                    background: {panel_bg};
                    border: 1px dashed {panel_border};
                    border-radius: 8px;
                    padding: 2.5rem 2rem;
                    text-align: center;
                    max-width: 520px;
                }}
            </style>
        </head>
        <body>
            <div class="empty-card">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 700; color: {secondary_sig}; margin-bottom: 0.75rem;">
                    [TOPOLOGY TRAVERSAL UNAVAILABLE]
                </div>
                <div style="font-size: 0.8125rem; color: {text_secondary}; line-height: 1.5; margin-bottom: 1rem;">
                    Active telemetry input does not contain IP communication endpoints or graph edges. Real graph-propagation traversal requires verified flow interactions.
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; color: {text_muted}; background: {bg_color}; padding: 0.5rem; border-radius: 4px; border: 1px solid {panel_border};">
                    Fail Visibly: 0 Discovered Endpoints • 0 Directional Flows
                </div>
            </div>
        </body>
        </html>
        """
        components.html(empty_html, height=height)
        return

    # Derive real graph traversal if not passed
    if traversal_data is None:
        from backend.graph_traversal import compute_graph_traversal
        traversal_data = compute_graph_traversal(
            graph_nodes=custom_nodes,
            graph_edges=custom_edges or [],
            max_k=5,
        )

    from backend.graph_traversal import format_bytes

    is_attack_active = (step_risk >= 0.35)
    steps = traversal_data.get("steps", [])
    step_info = steps[min(k_step, len(steps) - 1)] if steps else {}
    frontier_set = set(step_info.get("frontier", [])) if is_attack_active else set()
    added_node = step_info.get("added_node") if is_attack_active else None
    start_node = traversal_data.get("start_node") if is_attack_active else None
    walked_edges_at_k = traversal_data.get("walked_edges", [])[:k_step] if is_attack_active else []
    walked_pairs = {(str(e.get("source")), str(e.get("target"))): e for e in walked_edges_at_k}

    elements = []
    layout_name = "cose"

    for i, n in enumerate(custom_nodes):
        nid = str(n.get("id", f"node-{i}"))
        name = str(n.get("name", nid))
        ip = str(n.get("ip", nid))
        role = str(n.get("role", "Internal Enclave Host"))
        is_iso = (nid in isolated_nodes) or (name in isolated_nodes) or (ip in isolated_nodes)

        if is_iso:
            ntype = "isolated"
            lbl = f"[ISOLATED]\n{name}"
            node_risk = 0.08
        elif not is_attack_active:
            ntype = "nominal"
            lbl = f"{name}\n{ip}" if name != ip else name
            node_risk = round(step_risk * 0.4, 3)
        elif nid in frontier_set:
            ntype = "compromised"
            if nid == added_node and k_step > 0:
                src_hop = step_info.get("source_node", "")
                w_bytes = step_info.get("edge_byte_count", 0.0)
                lbl = f"[COMP k={k_step}]\n{name}\n← {src_hop} ({format_bytes(w_bytes)})"
            elif nid == start_node:
                lbl = f"[PATIENT ZERO]\n{name}"
            else:
                lbl = f"[COMPROMISED]\n{name}"
            node_risk = round(step_risk, 3)
        elif any(str(e.get("source")) in frontier_set and str(e.get("target")) == nid for e in (custom_edges or [])):
            ntype = "target"
            lbl = f"[TARGET]\n{name}"
            node_risk = round(step_risk * 0.7, 3)
        else:
            ntype = "nominal"
            lbl = f"{name}\n{ip}" if name != ip else name
            node_risk = round(min(0.12, step_risk * 0.15), 3)

        elements.append({
            "data": {
                "id": nid,
                "label": lbl,
                "name": name,
                "ip": ip,
                "role": role,
                "type": ntype,
                "risk": node_risk,
                "sigma": 0.04,
                "size": 52 if (ntype == "compromised" and nid == added_node) else (48 if ntype == "compromised" else (40 if ntype == "target" else 34)),
            }
        })

    edge_list = custom_edges if custom_edges else []
    for j, e in enumerate(edge_list):
        src = str(e.get("source", ""))
        tgt = str(e.get("target", ""))
        src_iso = (src in isolated_nodes)
        tgt_iso = (tgt in isolated_nodes)
        cnt = e.get("flow_count", 1)
        bc = float(e.get("byte_count", 0.0))

        if src_iso or tgt_iso:
            edge_type = "severed"
            edge_lbl = "[SEVERED]"
        elif not is_attack_active:
            edge_type = "nominal"
            edge_lbl = f"{cnt} flows" if j < 5 else ""
        elif (src, tgt) in walked_pairs:
            edge_type = "lateral_hop"
            edge_lbl = f"[HOP] {cnt} flw ({format_bytes(bc)})"
        elif src in frontier_set and tgt not in frontier_set:
            edge_type = "suspicious"
            edge_lbl = f"{cnt} flw ({format_bytes(bc)})"
        else:
            edge_type = "nominal"
            edge_lbl = ""

        elements.append({
            "data": {
                "id": f"custom-edge-{j}",
                "source": src,
                "target": tgt,
                "type": edge_type,
                "label": edge_lbl,
            }
        })

    elements_json = json.dumps(elements)

    # Dynamic styling tokens based on theme
    is_light = (theme == "light")

    bg_color = "#f8fafc" if is_light else "#0b0e13"
    grid_color = "rgba(148, 163, 184, 0.25)" if is_light else "rgba(59, 74, 61, 0.15)"
    panel_bg = "rgba(255, 255, 255, 0.96)" if is_light else "rgba(25, 28, 33, 0.94)"
    panel_border = "#cbd5e1" if is_light else "#1E2633"
    btn_bg = "#ffffff" if is_light else "#272a30"
    btn_border = "#cbd5e1" if is_light else "#3b4a3d"
    btn_color = "#0f172a" if is_light else "#f5fff2"
    btn_hover_bg = "#00A84D" if is_light else "#39ff88"
    btn_hover_color = "#ffffff" if is_light else "#003918"
    text_high = "#0f172a" if is_light else "#F3F4F6"
    text_muted = "#64748b" if is_light else "#A0AEC0"
    text_subtle = "#94a3b8" if is_light else "#7E8B9B"
    text_secondary = "#334155" if is_light else "#e1e2ea"
    
    primary_sig = "#00A84D" if is_light else "#39FF88"
    secondary_sig = "#D61B3C" if is_light else "#FF3B5C"
    tertiary_sig = "#D97706" if is_light else "#FFB84D"
    
    node_base_bg = "#e2e8f0" if is_light else "#1d2025"
    node_base_border = "#94a3b8" if is_light else "#3b4a3d"
    node_label_color = "#0f172a" if is_light else "#bacbb9"
    nominal_node_bg = "#ffffff" if is_light else "#191c21"
    nominal_border = "#cbd5e1" if is_light else "#3b4a3d"
    edge_default = "#94a3b8" if is_light else "#3b4a3d"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                background-color: {bg_color};
                color: {text_high};
                font-family: 'Inter', -apple-system, sans-serif;
                overflow: hidden;
            }}
            #cy {{
                width: 100vw;
                height: 100vh;
                position: absolute;
                top: 0;
                left: 0;
                background-color: {bg_color};
                background-image: 
                    linear-gradient(to right, {grid_color} 1px, transparent 1px),
                    linear-gradient(to bottom, {grid_color} 1px, transparent 1px);
                background-size: 32px 32px;
            }}
            /* Overlaid Control Pill */
            .controls-panel {{
                position: absolute;
                top: 12px;
                left: 12px;
                z-index: 100;
                background: {panel_bg};
                border: 1px solid {panel_border};
                border-radius: 4px;
                padding: 4px 6px;
                display: flex;
                flex-direction: column;
                gap: 4px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.12);
            }}
            .ctrl-btn {{
                background: {btn_bg};
                border: 1px solid {btn_border};
                color: {btn_color};
                min-width: 38px;
                height: 26px;
                border-radius: 3px;
                cursor: pointer;
                font-family: 'JetBrains Mono', monospace;
                font-size: 10px;
                font-weight: 700;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.15s;
                text-transform: uppercase;
            }}
            .ctrl-btn:hover {{
                background: {btn_hover_bg};
                color: {btn_hover_color};
                border-color: {btn_hover_bg};
            }}
            /* Status readout overlay */
            .status-overlay {{
                position: absolute;
                top: 12px;
                right: 12px;
                z-index: 100;
                background: {panel_bg};
                border: 1px solid {panel_border};
                border-radius: 4px;
                padding: 6px 12px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
                color: {text_muted};
                display: flex;
                align-items: center;
                gap: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.12);
            }}
            .legend-panel {{
                position: absolute;
                bottom: 12px;
                left: 12px;
                z-index: 100;
                background: {panel_bg};
                border: 1px solid {panel_border};
                border-radius: 4px;
                padding: 8px 12px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 10px;
                color: {text_secondary};
                max-width: 340px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.12);
            }}
            .legend-title {{
                color: {text_subtle};
                text-transform: uppercase;
                font-weight: 700;
                margin-bottom: 6px;
                display: flex;
                justify-content: space-between;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 4px;
            }}
            .dot-red {{ width: 10px; height: 10px; border-radius: 50%; background: {secondary_sig}; box-shadow: 0 0 8px {secondary_sig}; }}
            .dot-amber {{ width: 10px; height: 10px; border-radius: 50%; background: {tertiary_sig}; }}
            .dot-green {{ width: 10px; height: 10px; border-radius: 50%; background: {primary_sig}; }}
            .dot-iso {{ width: 10px; height: 10px; border-radius: 2px; background: #334155; border: 1.5px dashed {secondary_sig}; }}
            @keyframes pulseGlow {{
                0%, 100% {{ box-shadow: 0 0 4px {secondary_sig}; }}
                50% {{ box-shadow: 0 0 14px {secondary_sig}; }}
            }}
            .dot-red {{ animation: pulseGlow 2s ease-in-out infinite; }}
        </style>
    </head>
    <body>
        <div id="cy"></div>

        <div class="controls-panel">
            <button class="ctrl-btn" id="btn-zoom-in" title="Zoom In">+</button>
            <button class="ctrl-btn" id="btn-zoom-out" title="Zoom Out">-</button>
            <button class="ctrl-btn" id="btn-fit" title="Fit Topology">FIT</button>
            <button class="ctrl-btn" id="btn-reset" title="Reset Layout">RESET</button>
        </div>

        <div class="status-overlay">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:{primary_sig};"></span>
            <span>CYTOSCAPE DYNAMIC ENGINE</span>
            <span style="color:{panel_border};">|</span>
            <span>STEP: <b style="color:{primary_sig};">k={k_step} (+{k_step*15}m)</b></span>
            <span style="color:{panel_border};">|</span>
            <span>ISOLATED: <b style="color:{secondary_sig};">{len(isolated_nodes)}</b></span>
        </div>

        <div class="legend-panel">
            <div class="legend-title">
                <span>DYNAMIC PROPAGATION TOPOLOGY</span>
                <span style="color:{primary_sig};">ACTIVE</span>
            </div>
            <div class="legend-item"><span class="dot-red"></span><span>Compromised Node (Active Lateral / C2 Pivot)</span></div>
            <div class="legend-item"><span class="dot-amber"></span><span>Infection Target / Elevated Risk Horizon</span></div>
            <div class="legend-item"><span class="dot-green"></span><span>Verified Nominal Microservice</span></div>
            <div class="legend-item"><span class="dot-iso"></span><span>SDN Quarantined / Severed Host</span></div>
            <div style="border-top:1px solid {panel_border}; margin-top:6px; padding-top:4px; color:{text_muted};">
                Arrows indicate live directional attack trajectory.
            </div>
        </div>

        <script>
            const elements = {elements_json};

            const cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: elements,
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'label': 'data(label)',
                            'color': '{node_label_color}',
                            'font-family': 'JetBrains Mono, monospace',
                            'font-size': '9px',
                            'text-wrap': 'wrap',
                            'text-valign': 'bottom',
                            'text-margin-y': 6,
                            'background-color': '{node_base_bg}',
                            'border-width': 1.5,
                            'border-color': '{node_base_border}',
                            'width': 'data(size)',
                            'height': 'data(size)',
                            'transition-property': 'background-color, border-color, width, height',
                            'transition-duration': '0.25s'
                        }}
                    }},
                    // Compromised Node style
                    {{
                        selector: 'node[type = "compromised"]',
                        style: {{
                            'background-color': '{secondary_sig}',
                            'border-color': '{primary_sig}',
                            'border-width': 3,
                            'color': '#ffffff',
                            'font-weight': 'bold',
                            'font-size': '10px'
                        }}
                    }},
                    {{
                        selector: 'node[type = "external_threat"]',
                        style: {{
                            'background-color': '{secondary_sig}',
                            'border-color': '{secondary_sig}',
                            'border-width': 2,
                            'color': '{secondary_sig}',
                            'font-weight': 'bold'
                        }}
                    }},
                    // Target node style
                    {{
                        selector: 'node[type = "target"], node[type = "dispersion"]',
                        style: {{
                            'background-color': '{"#fef3c7" if is_light else "#272a30"}',
                            'border-color': '{tertiary_sig}',
                            'border-width': 2,
                            'border-style': 'dashed',
                            'color': '{"#92400e" if is_light else "#ffddb3"}'
                        }}
                    }},
                    // Gateways
                    {{
                        selector: 'node[type = "gateway"]',
                        style: {{
                            'shape': 'diamond',
                            'border-color': '{primary_sig}',
                            'border-width': 1.5,
                            'background-color': '{nominal_node_bg}',
                            'color': '{node_label_color}'
                        }}
                    }},
                    // Nominal nodes
                    {{
                        selector: 'node[type = "nominal"]',
                        style: {{
                            'border-color': '{nominal_border}',
                            'background-color': '{nominal_node_bg}',
                            'color': '{node_label_color}'
                        }}
                    }},
                    // Isolated / Quarantined node style
                    {{
                        selector: 'node[type = "isolated"]',
                        style: {{
                            'shape': 'hexagon',
                            'background-color': '#1e293b',
                            'border-color': '{secondary_sig}',
                            'border-width': 3.5,
                            'border-style': 'dashed',
                            'color': '{tertiary_sig}',
                            'font-weight': 'bold',
                            'font-size': '10px'
                        }}
                    }},
                    // Selection state
                    {{
                        selector: 'node:selected, node[id = "{selected_node_id}"]',
                        style: {{
                            'border-color': '{primary_sig}',
                            'border-width': 4,
                            'color': '{primary_sig}'
                        }}
                    }},
                    // Standard Edges with Directional Arrows
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 1.5,
                            'line-color': '{edge_default}',
                            'target-arrow-color': '{edge_default}',
                            'target-arrow-shape': 'triangle',
                            'arrow-scale': 1.4,
                            'curve-style': 'bezier',
                            'opacity': 0.65
                        }}
                    }},
                    // Active C2 exfil arrow
                    {{
                        selector: 'edge[type = "c2_exfil"]',
                        style: {{
                            'width': 3.5,
                            'line-color': '{secondary_sig}',
                            'target-arrow-color': '{secondary_sig}',
                            'target-arrow-shape': 'triangle',
                            'arrow-scale': 1.8,
                            'line-style': 'dashed',
                            'line-dash-pattern': [10, 5],
                            'opacity': 1.0,
                            'label': 'data(label)',
                            'font-family': 'JetBrains Mono',
                            'font-size': '9px',
                            'color': '{secondary_sig}',
                            'text-rotation': 'autorotate',
                            'text-margin-y': -8
                        }}
                    }},
                    // Active lateral hop arrow
                    {{
                        selector: 'edge[type = "lateral_hop"]',
                        style: {{
                            'width': 2.5,
                            'line-color': '{secondary_sig}',
                            'target-arrow-color': '{secondary_sig}',
                            'target-arrow-shape': 'triangle',
                            'arrow-scale': 1.6,
                            'line-style': 'dashed',
                            'line-dash-pattern': [6, 3],
                            'opacity': 0.95,
                            'label': 'data(label)',
                            'font-family': 'JetBrains Mono',
                            'font-size': '8.5px',
                            'color': '{secondary_sig}',
                            'text-rotation': 'autorotate',
                            'text-margin-y': -8
                        }}
                    }},
                    // Suspicious hop arrow
                    {{
                        selector: 'edge[type = "suspicious"]',
                        style: {{
                            'width': 2,
                            'line-color': '{tertiary_sig}',
                            'target-arrow-color': '{tertiary_sig}',
                            'target-arrow-shape': 'triangle',
                            'arrow-scale': 1.4,
                            'line-style': 'dashed',
                            'line-dash-pattern': [8, 4],
                            'opacity': 0.8
                        }}
                    }},
                    // Severed / Blocked Edge
                    {{
                        selector: 'edge[type = "severed"]',
                        style: {{
                            'width': 1.5,
                            'line-color': '#475569',
                            'target-arrow-color': '#475569',
                            'target-arrow-shape': 'tee',
                            'line-style': 'dotted',
                            'opacity': 0.35,
                            'label': 'data(label)',
                            'font-family': 'JetBrains Mono',
                            'font-size': '8px',
                            'color': '#ef4444'
                        }}
                    }},
                    // Dormant edges in future horizons
                    {{
                        selector: 'edge[type = "dormant"]',
                        style: {{
                            'display': 'none'
                        }}
                    }}
                ],
                layout: {{
                    name: '{layout_name}',
                    fit: true,
                    padding: 35,
                    animate: false
                }},
                userZoomingEnabled: true,
                userPanningEnabled: true,
                boxSelectionEnabled: false
            }});

            // Controls
            document.getElementById('btn-zoom-in').addEventListener('click', () => {{
                cy.zoom(cy.zoom() * 1.25);
            }});
            document.getElementById('btn-zoom-out').addEventListener('click', () => {{
                cy.zoom(cy.zoom() * 0.8);
            }});
            document.getElementById('btn-fit').addEventListener('click', () => {{
                cy.fit(null, 30);
            }});
            document.getElementById('btn-reset').addEventListener('click', () => {{
                cy.reset();
                cy.fit(null, 30);
            }});

            cy.on('tap', 'node', function(evt) {{
                const node = evt.target;
                cy.elements().removeClass('highlighted').style('opacity', 0.25);
                node.closedNeighborhood().style('opacity', 1.0);
                node.style('opacity', 1.0);
            }});

            cy.on('tap', function(evt) {{
                if (evt.target === cy) {{
                    cy.elements().style('opacity', 1.0);
                }}
            }});

            cy.fit(null, 30);

            // ── Live Animated Flowing Arrows ──
            // Creates moving dash effect on active attack edges
            let dashOffset = 0;
            function animateEdges() {{
                dashOffset += 0.6;
                cy.edges('[type = "c2_exfil"]').style('line-dash-offset', -dashOffset * 1.4);
                cy.edges('[type = "lateral_hop"]').style('line-dash-offset', -dashOffset);
                cy.edges('[type = "suspicious"]').style('line-dash-offset', -dashOffset * 0.7);
                requestAnimationFrame(animateEdges);
            }}
            animateEdges();

            // ── Compromised Node Pulse Animation ──
            // Alternates border glow on compromised nodes
            let pulsePhase = 0;
            function pulseCompromised() {{
                pulsePhase += 0.03;
                const scale = 0.6 + 0.4 * Math.abs(Math.sin(pulsePhase));
                cy.nodes('[type = "compromised"]').style('border-width', 2 + scale * 3);
                requestAnimationFrame(pulseCompromised);
            }}
            pulseCompromised();
        </script>
    </body>
    </html>
    """

    components.html(html_content, height=height, scrolling=False)
