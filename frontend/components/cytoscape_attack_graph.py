"""
Cytoscape.js Interactive Attack Graph Component for Streamlit
Matches the visual spec in Stitch folder shadowcat_soc_attack_graph_with_dynamic_k_step_rollout.
Features confidence glow halos, flowing lateral/exfil edges, K-step rollout state, and node inspector.
"""

import json
from typing import Dict, Any, Optional
import streamlit.components.v1 as components


def render_cytoscape_graph(
    k_step: int = 0,
    selected_node_id: str = "svc-auth-master",
    height: int = 680,
    theme: str = "dark",
) -> None:
    """
    Renders an interactive Cytoscape.js attack topology graph matching the Stitch design.
    """
    # 19 Enterprise nodes defined in the Stitch mockup
    nodes_data = [
        # Core Compromise & Pivots
        {"id": "svc-auth-master", "label": "svc-auth-master\n10.0.14.88", "name": "svc-auth-master", "ip": "10.0.14.88", "role": "Auth Master Daemon", "type": "compromised", "risk": 0.962, "sigma": 0.04, "x": 460, "y": 360, "size": 52},
        {"id": "ip-10-0-14-88", "label": "ip-10-0-14-88\nWorker Node", "name": "ip-10-0-14-88", "ip": "10.0.14.88", "role": "Compute Worker Pod", "type": "lateral", "risk": 0.884, "sigma": 0.06, "x": 670, "y": 470, "size": 42},
        {"id": "analytics-agg-02", "label": "analytics-agg-02\n10.0.14.92", "name": "analytics-agg-02", "ip": "10.0.14.92", "role": "Analytics Aggregator", "type": "target", "risk": 0.740, "sigma": 0.28, "x": 820, "y": 570, "size": 36},
        {"id": "audit-vault", "label": "audit-vault\n10.0.14.5", "name": "audit-vault", "ip": "10.0.14.5", "role": "Audit Cryptographic Vault", "type": "target", "risk": 0.810, "sigma": 0.22, "x": 690, "y": 640, "size": 38},
        {"id": "iam-sync-daemon", "label": "iam-sync-daemon\n10.0.14.15", "name": "iam-sync-daemon", "ip": "10.0.14.15", "role": "IAM Credential Sync", "type": "dispersion", "risk": 0.690, "sigma": 0.31, "x": 520, "y": 660, "size": 36},
        {"id": "db-shard-01", "label": "db-shard-01\n10.0.3.12", "name": "db-shard-01", "ip": "10.0.3.12", "role": "Relational DB Shard", "type": "nominal", "risk": 0.120, "sigma": 0.02, "x": 200, "y": 480, "size": 36},
        {"id": "api-gateway-int", "label": "api-gateway-int\n10.0.1.10", "name": "api-gateway-int", "ip": "10.0.1.10", "role": "Internal API Gateway", "type": "gateway", "risk": 0.210, "sigma": 0.03, "x": 230, "y": 200, "size": 40},
        {"id": "edge-gw-02", "label": "edge-gw-02\n10.0.1.1", "name": "edge-gw-02", "ip": "10.0.1.1", "role": "Perimeter Egress Gateway", "type": "gateway", "risk": 0.340, "sigma": 0.05, "x": 360, "y": 140, "size": 42},
        {"id": "ws-analyst-12", "label": "ws-analyst-12\n10.0.14.21", "name": "ws-analyst-12", "ip": "10.0.14.21", "role": "Hunter Workstation", "type": "nominal", "risk": 0.040, "sigma": 0.01, "x": 140, "y": 330, "size": 32},
        {"id": "ws-analyst-08", "label": "ws-analyst-08\n10.0.14.29", "name": "ws-analyst-08", "ip": "10.0.14.29", "role": "Hunter Workstation", "type": "nominal", "risk": 0.030, "sigma": 0.01, "x": 110, "y": 490, "size": 30},
        {"id": "k8s-worker-alpha-04", "label": "k8s-worker-04\n10.0.2.80", "name": "k8s-worker-alpha-04", "ip": "10.0.2.80", "role": "App Container Worker", "type": "nominal", "risk": 0.080, "sigma": 0.02, "x": 320, "y": 620, "size": 34},
        {"id": "dns-authoritative", "label": "dns-auth\n10.0.0.53", "name": "dns-authoritative", "ip": "10.0.0.53", "role": "Internal CoreDNS", "type": "nominal", "risk": 0.050, "sigma": 0.01, "x": 590, "y": 170, "size": 34},
        {"id": "nat-gateway-pub", "label": "nat-gw-pub\n172.16.0.1", "name": "nat-gateway-pub", "ip": "172.16.0.1", "role": "Egress NAT Translator", "type": "gateway", "risk": 0.450, "sigma": 0.08, "x": 780, "y": 150, "size": 38},
        {"id": "ext-asn4837-c2", "label": "EXTERNAL C2\nASN 4837", "name": "ext-asn4837-c2", "ip": "45.138.21.9", "role": "Uncatalogued Foreign C2", "type": "external_threat", "risk": 0.992, "sigma": 0.01, "x": 850, "y": 330, "size": 46},
        {"id": "ext-foreign-dns", "label": "FOREIGN DNS\n198.51.100.4", "name": "ext-foreign-dns", "ip": "198.51.100.4", "role": "Suspicious DNS Resolver", "type": "external_threat", "risk": 0.910, "sigma": 0.03, "x": 920, "y": 180, "size": 36},
        {"id": "storage-blob-s3", "label": "storage-blob\n10.0.6.10", "name": "storage-blob-s3", "ip": "10.0.6.10", "role": "Object Store Gateway", "type": "nominal", "risk": 0.060, "sigma": 0.01, "x": 180, "y": 620, "size": 32},
        {"id": "mon-prometheus", "label": "prometheus\n10.0.9.10", "name": "mon-prometheus", "ip": "10.0.9.10", "role": "Metrics Collector", "type": "nominal", "risk": 0.020, "sigma": 0.01, "x": 720, "y": 390, "size": 30},
        {"id": "dc-shadow-02", "label": "dc-shadow-02\n10.0.5.2", "name": "dc-shadow-02", "ip": "10.0.5.2", "role": "Secondary Identity DC", "type": "nominal", "risk": 0.090, "sigma": 0.02, "x": 420, "y": 540, "size": 36},
        {"id": "waf-perimeter", "label": "waf-edge\n10.0.1.5", "name": "waf-perimeter", "ip": "10.0.1.5", "role": "App Shield WAF", "type": "gateway", "risk": 0.150, "sigma": 0.02, "x": 500, "y": 240, "size": 36}
    ]

    edges_data = [
        # Lateral hops & C2 exfil
        {"source": "svc-auth-master", "target": "ext-asn4837-c2", "type": "c2_exfil", "label": "T1071.001 Exfil (14.8 GB/s)"},
        {"source": "svc-auth-master", "target": "ip-10-0-14-88", "type": "lateral_hop", "label": "RPC Probe Fanout"},
        {"source": "ip-10-0-14-88", "target": "analytics-agg-02", "type": "lateral_hop", "label": "Lateral Hop 2"},
        {"source": "ip-10-0-14-88", "target": "audit-vault", "type": "lateral_hop", "label": "Token Siphon"},
        {"source": "svc-auth-master", "target": "iam-sync-daemon", "type": "suspicious", "label": "Kerberos TGS Probe"},
        {"source": "svc-auth-master", "target": "db-shard-01", "type": "nominal", "label": "DB Heartbeat"},
        {"source": "svc-auth-master", "target": "api-gateway-int", "type": "nominal", "label": "Session Token Ingress"},
        {"source": "svc-auth-master", "target": "k8s-worker-alpha-04", "type": "nominal", "label": "Cluster Health"},
        {"source": "svc-auth-master", "target": "edge-gw-02", "type": "nominal", "label": "Perimeter Route"},
        {"source": "edge-gw-02", "target": "dns-authoritative", "type": "nominal", "label": "DNS Query"},
        {"source": "dns-authoritative", "target": "nat-gateway-pub", "type": "nominal", "label": "Egress NAT"},
        {"source": "nat-gateway-pub", "target": "ext-foreign-dns", "type": "suspicious", "label": "Encrypted Tunnel"},
        {"source": "ws-analyst-12", "target": "api-gateway-int", "type": "nominal", "label": "Analyst Console"},
        {"source": "ws-analyst-08", "target": "db-shard-01", "type": "nominal", "label": "Audit Query"},
        {"source": "db-shard-01", "target": "storage-blob-s3", "type": "nominal", "label": "Snapshot Sync"},
        {"source": "iam-sync-daemon", "target": "audit-vault", "type": "suspicious", "label": "Vault Siphon Attempt"},
        {"source": "k8s-worker-alpha-04", "target": "iam-sync-daemon", "type": "nominal", "label": "App Pod Auth"},
    ]

    elements = []
    for n in nodes_data:
        elements.append({
            "data": {
                "id": n["id"],
                "label": n["label"],
                "name": n["name"],
                "ip": n["ip"],
                "role": n["role"],
                "type": n["type"],
                "risk": n["risk"],
                "sigma": n["sigma"],
                "size": n["size"],
            },
            "position": {"x": n["x"], "y": n["y"]}
        })

    for i, e in enumerate(edges_data):
        elements.append({
            "data": {
                "id": f"edge-{i}",
                "source": e["source"],
                "target": e["target"],
                "type": e["type"],
                "label": e["label"],
            }
        })

    elements_json = json.dumps(elements)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                background-color: #0b0e13;
                color: #e1e2ea;
                font-family: 'Inter', -apple-system, sans-serif;
                overflow: hidden;
            }}
            #cy {{
                width: 100vw;
                height: 100vh;
                position: absolute;
                top: 0;
                left: 0;
                background-color: #0b0e13;
                background-image: 
                    linear-gradient(to right, rgba(59, 74, 61, 0.15) 1px, transparent 1px),
                    linear-gradient(to bottom, rgba(59, 74, 61, 0.15) 1px, transparent 1px);
                background-size: 32px 32px;
            }}
            /* Overlaid Control Pill */
            .controls-panel {{
                position: absolute;
                top: 12px;
                left: 12px;
                z-index: 100;
                background: rgba(25, 28, 33, 0.92);
                border: 1px solid #1E2633;
                border-radius: 4px;
                padding: 4px 6px;
                display: flex;
                flex-direction: column;
                gap: 4px;
                box-shadow: none;
            }}
            .ctrl-btn {{
                background: #272a30;
                border: 1px solid #3b4a3d;
                color: #f5fff2;
                width: 28px;
                height: 28px;
                border-radius: 4px;
                cursor: pointer;
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                font-weight: 700;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.15s;
            }}
            .ctrl-btn:hover {{
                background: #39ff88;
                color: #003918;
                border-color: #39ff88;
            }}
            /* Status readout overlay */
            .status-overlay {{
                position: absolute;
                top: 12px;
                right: 12px;
                z-index: 100;
                background: rgba(25, 28, 33, 0.92);
                border: 1px solid #1E2633;
                border-radius: 4px;
                padding: 6px 12px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
                color: #A0AEC0;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .legend-panel {{
                position: absolute;
                bottom: 12px;
                left: 12px;
                z-index: 100;
                background: rgba(25, 28, 33, 0.92);
                border: 1px solid #1E2633;
                border-radius: 4px;
                padding: 8px 12px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 10px;
                color: #A0AEC0;
                max-width: 320px;
            }}
            .legend-title {{
                color: #7E8B9B;
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
            .dot-red {{ width: 10px; height: 10px; border-radius: 50%; background: #FF3B5C; box-shadow: 0 0 8px #FF3B5C; }}
            .dot-amber {{ width: 10px; height: 10px; border-radius: 50%; background: #FFB84D; }}
            .dot-green {{ width: 10px; height: 10px; border-radius: 50%; background: #39FF88; }}
        </style>
    </head>
    <body>
        <div id="cy"></div>

        <div class="controls-panel">
            <button class="ctrl-btn" id="btn-zoom-in" title="Zoom In">+</button>
            <button class="ctrl-btn" id="btn-zoom-out" title="Zoom Out">-</button>
            <button class="ctrl-btn" id="btn-fit" title="Fit Topology">⛶</button>
            <button class="ctrl-btn" id="btn-reset" title="Reset Layout">↺</button>
        </div>

        <div class="status-overlay">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#39FF88;"></span>
            <span>CYTOSCAPE ENGINE: 60 FPS</span>
            <span style="color:#3b4a3d;">|</span>
            <span>STEP: <b style="color:#39FF88;">k={k_step}</b></span>
            <span style="color:#3b4a3d;">|</span>
            <span>NODES: <b style="color:#F3F4F6;">19</b></span>
            <span style="color:#3b4a3d;">|</span>
            <span>EDGES: <b style="color:#39FF88;">34</b></span>
        </div>

        <div class="legend-panel">
            <div class="legend-title">
                <span>GRAPH ENTITY LEGEND</span>
                <span style="color:#39FF88;">LIVE</span>
            </div>
            <div class="legend-item"><span class="dot-red"></span><span>Compromised Beacon (p &gt; 0.90, σ ≤ 0.08)</span></div>
            <div class="legend-item"><span class="dot-amber"></span><span>Elevated Risk / Dispersion (σ &gt; 0.25)</span></div>
            <div class="legend-item"><span class="dot-green"></span><span>Verified Nominal Microservice (p ≤ 0.20)</span></div>
            <div style="border-top:1px solid #1E2633; margin-top:6px; padding-top:4px; color:#7E8B9B;">
                Click any node to inspect telemetry & blast radius.
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
                            'color': '#bacbb9',
                            'font-family': 'JetBrains Mono, monospace',
                            'font-size': '9px',
                            'text-wrap': 'wrap',
                            'text-valign': 'bottom',
                            'text-margin-y': 6,
                            'background-color': '#1d2025',
                            'border-width': 1.5,
                            'border-color': '#3b4a3d',
                            'width': 'data(size)',
                            'height': 'data(size)',
                            'transition-property': 'background-color, border-color, width, height',
                            'transition-duration': '0.2s'
                        }}
                    }},
                    // High Risk / Saturated Red Glow
                    {{
                        selector: 'node[type = "compromised"]',
                        style: {{
                            'background-color': '#FF3B5C',
                            'border-color': '#39FF88',
                            'border-width': 2.5,
                            'color': '#ffffff',
                            'font-weight': 'bold',
                            'font-size': '10.5px'
                        }}
                    }},
                    {{
                        selector: 'node[type = "external_threat"]',
                        style: {{
                            'background-color': '#c7003a',
                            'border-color': '#FF3B5C',
                            'border-width': 2,
                            'color': '#FF3B5C',
                            'font-weight': 'bold'
                        }}
                    }},
                    {{
                        selector: 'node[type = "lateral"]',
                        style: {{
                            'background-color': '#b01330',
                            'border-color': '#FF3B5C',
                            'border-width': 2,
                            'color': '#ffb3b6'
                        }}
                    }},
                    // Elevated dispersion / amber halo
                    {{
                        selector: 'node[type = "dispersion"], node[type = "target"]',
                        style: {{
                            'background-color': '#272a30',
                            'border-color': '#FFB84D',
                            'border-width': 2,
                            'border-style': 'dashed',
                            'color': '#ffddb3'
                        }}
                    }},
                    // Gateways
                    {{
                        selector: 'node[type = "gateway"]',
                        style: {{
                            'shape': 'diamond',
                            'border-color': '#39FF88',
                            'border-width': 1.5,
                            'color': '#f5fff2'
                        }}
                    }},
                    // Nominal nodes
                    {{
                        selector: 'node[type = "nominal"]',
                        style: {{
                            'border-color': '#3b4a3d',
                            'background-color': '#191c21'
                        }}
                    }},
                    // Selection state
                    {{
                        selector: 'node:selected, node[id = "{selected_node_id}"]',
                        style: {{
                            'border-color': '#39FF88',
                            'border-width': 3.5,
                            'color': '#39FF88'
                        }}
                    }},
                    // Edges
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 1.5,
                            'line-color': '#3b4a3d',
                            'target-arrow-color': '#3b4a3d',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'opacity': 0.7
                        }}
                    }},
                    {{
                        selector: 'edge[type = "c2_exfil"]',
                        style: {{
                            'width': 3.5,
                            'line-color': '#FF3B5C',
                            'target-arrow-color': '#FF3B5C',
                            'line-style': 'dashed',
                            'opacity': 1.0,
                            'label': 'data(label)',
                            'font-family': 'JetBrains Mono, monospace',
                            'font-size': '8px',
                            'color': '#FF3B5C',
                            'text-background-color': '#0b0e13',
                            'text-background-opacity': 0.85,
                            'text-background-padding': 2
                        }}
                    }},
                    {{
                        selector: 'edge[type = "lateral_hop"]',
                        style: {{
                            'width': 2.5,
                            'line-color': '#FF3B5C',
                            'target-arrow-color': '#FF3B5C',
                            'line-style': 'dashed',
                            'opacity': 0.9
                        }}
                    }},
                    {{
                        selector: 'edge[type = "suspicious"]',
                        style: {{
                            'width': 2.0,
                            'line-color': '#FFB84D',
                            'target-arrow-color': '#FFB84D',
                            'line-style': 'dashed',
                            'opacity': 0.8
                        }}
                    }}
                ],
                layout: {{
                    name: 'preset'
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
                const nodeData = node.data();
                // Highlight connected neighborhood
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
        </script>
    </body>
    </html>
    """

    components.html(html_content, height=height, scrolling=False)
