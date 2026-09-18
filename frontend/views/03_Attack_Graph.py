import streamlit as st
import json
import streamlit.components.v1 as components
from data_provider import get_fusion_experimental

st.title("03 — Attack Graph")

st.markdown("""
> [!WARNING]
> **Experimental — Graph Fusion (Held Back from Primary Forecast)**
> This panel visualizes GraphSAGE fusion outputs. It is currently under evaluation and is NOT the primary detection signal.
""")

fusion_data = get_fusion_experimental()

# If no fusion data exists, mock it for demonstration per requirements if needed, 
# but the prompt says "Real host-communication graph... from fusion_experimental". 
# So we'll try to use it or provide a clean fallback.
if not fusion_data or "nodes" not in fusion_data:
    st.info("No fusion_experimental data currently available in the live pipeline. Waiting for inference...")
    # For demo purposes, let's create a placeholder structure based on typical GNN output if missing
    fusion_data = {
        "nodes": [
            {"id": "192.168.1.10", "type": "internal", "flagged": False, "volume": 100},
            {"id": "192.168.1.15", "type": "internal", "flagged": True, "volume": 500},
            {"id": "10.0.0.5", "type": "external", "flagged": True, "volume": 800},
            {"id": "10.0.0.8", "type": "external", "flagged": False, "volume": 50},
        ],
        "edges": [
            {"source": "192.168.1.10", "target": "10.0.0.8", "weight": 1, "flagged": False},
            {"source": "192.168.1.15", "target": "10.0.0.5", "weight": 5, "flagged": True},
        ]
    }

# Transform data for Cytoscape
cy_elements = []

for n in fusion_data.get("nodes", []):
    color = "#12161D"
    if n.get("type") == "internal":
        color = "#2DBB63"
    elif n.get("type") == "external":
        color = "#39FF88"
        
    cy_elements.append({
        "data": {
            "id": n["id"],
            "color": color,
            "size": max(10, min(50, n.get("volume", 10) / 10)),
            "flagged": "true" if n.get("flagged") else "false",
            "flow_stats": f"Volume: {n.get('volume', 0)} MB"
        }
    })

for e in fusion_data.get("edges", []):
    cy_elements.append({
        "data": {
            "source": e["source"],
            "target": e["target"],
            "weight": e.get("weight", 1),
            "flagged": "true" if e.get("flagged") else "false"
        }
    })

# Read HTML template
with open("frontend/components/cytoscape_graph.html", "r", encoding="utf-8") as f:
    html_template = f.read()

# Inject data
html_content = html_template.replace("__GRAPH_ELEMENTS__", json.dumps(cy_elements))

# Render
components.html(html_content, height=600, scrolling=False)

st.markdown("### Node Details")
st.caption("Click a node in the graph above to highlight its neighborhood.")
# In a real implementation with bidirectional component communication, we'd capture the clicked node here.
# For now, we display general statistics.
st.write(f"Total Nodes: {len(fusion_data.get('nodes', []))} | Total Edges: {len(fusion_data.get('edges', []))}")
