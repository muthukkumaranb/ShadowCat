import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from backend.predict import predict
from backend.graph_traversal import compute_graph_traversal

print("=" * 80)
print("TEST CASE 1: SAMPLE WINDOW W_14-02-2018_20180214_014400 TRAVERSAL RESULT")
print("=" * 80)
df = pd.read_parquet("data-engineering/data/ucs/ucs_windows.parquet")
w_df = df[df["window_id"] == "W_14-02-2018_20180214_014400"]
out = predict(w_df, source_type="flows")
gt = out.get("graph_traversal", {})

print("Status:      ", gt.get("status"))
print("Start Node:  ", gt.get("start_node"))
print("Start Reason:", gt.get("start_reason"))
print("-" * 80)
for s in gt.get("steps", []):
    k = s["k"]
    added = s["added_node"]
    src = s["source_node"]
    b = s["edge_byte_count"]
    p = s.get("edge_packet_count", 0.0)
    role = s.get("candidate_role")
    deg = s["candidate_degree"]
    expl = s["explanation"]
    print(f"k={k}: added={added} | src={src} | bytes={b} | pkts={p} | role={role} | deg={deg}")
    print(f"     explanation: {expl}")
print("=" * 80)

# TEST CASE 2: Role Priority Deciding Factor Verification
print("\n[TEST CASE 2] Role Priority Deciding Factor on Tied Volume (0 B):")
test_nodes = [
    {"id": "Workstation-P0", "role": "Enclave Workstation / Internal Host"},
    {"id": "DomainController-Auth", "role": "Identity / Auth Cluster"},
    {"id": "EphemeralScanTarget", "role": "External Service Port"},
]
test_edges = [
    {"source": "Workstation-P0", "target": "DomainController-Auth", "byte_count": 0.0, "packet_count": 1.0, "flow_count": 1},
    {"source": "Workstation-P0", "target": "EphemeralScanTarget", "byte_count": 0.0, "packet_count": 1.0, "flow_count": 1},
]
trav_role = compute_graph_traversal(test_nodes, test_edges, max_k=2)
print("Candidates tied on volume (0 B, 1 pkt, 1 flow, degree 1):")
for s in trav_role["steps"]:
    print(f"k={s['k']}: added={s['added_node']} | role={s.get('candidate_role')}")
    print(f"     explanation: {s['explanation']}")

assert trav_role["steps"][1]["added_node"] == "DomainController-Auth", "Role priority should select Identity / Auth Cluster first"
assert "selected for higher target value (Identity / Auth Cluster)" in trav_role["steps"][1]["explanation"]
print("[PASS] Role priority successfully broke tie and generated exact explanation!")

# TEST CASE 3: Sparse Graph Early Halt
print("\n[TEST CASE 3] Sparse Graph Early Halt:")
sparse_nodes = [{"id": "Node-A"}, {"id": "Node-B"}, {"id": "Node-C"}]
sparse_edges = [{"source": "Node-A", "target": "Node-B", "byte_count": 100.0, "flow_count": 1, "packet_count": 2.0}]
trav_sparse = compute_graph_traversal(sparse_nodes, sparse_edges, max_k=4)
print(f"k=1 added: {trav_sparse['steps'][1]['added_node']}, stopped: {trav_sparse['steps'][1]['stopped_early']}")
print(f"k=2 added: {trav_sparse['steps'][2]['added_node']}, stopped: {trav_sparse['steps'][2]['stopped_early']}, expl: {trav_sparse['steps'][2]['explanation']}")
assert trav_sparse["steps"][1]["stopped_early"] is False
assert trav_sparse["steps"][2]["stopped_early"] is True
assert trav_sparse["steps"][2]["added_node"] is None
assert "Propagation halted: No outgoing edges" in trav_sparse["steps"][2]["explanation"]
print("[PASS] Sparse graph early halt works as before.")

# TEST CASE 4: Empty Graph Visible Failure
print("\n[TEST CASE 4] Empty Graph Visible Failure:")
trav_empty = compute_graph_traversal([], [])
assert trav_empty["status"] == "no_data"
print(f"Empty graph status: {trav_empty['status']} | msg: {trav_empty['status_message']}")
print("[PASS] Empty graph visible failure works as before.")
