import sys
from pathlib import Path
import pandas as pd
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def format_bytes(b: float) -> str:
    if b < 1024:
        return f"{b:.0f} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    elif b < 1024**3:
        return f"{b/(1024**2):.1f} MB"
    else:
        return f"{b/(1024**3):.2f} GB"

def compute_graph_traversal(
    graph_nodes: list[dict],
    graph_edges: list[dict],
    flagged_flows: list[dict] = None,
    max_k: int = 5,
    primary_metric: str = "byte_count",
) -> dict:
    if not graph_nodes or not graph_edges:
        return {
            "status": "no_data",
            "status_message": "No graph topology available for this telemetry window.",
            "start_node": None,
            "start_reason": None,
            "total_nodes": 0,
            "total_edges": 0,
            "steps": [],
            "walked_edges": [],
        }

    flagged_flows = flagged_flows or []
    node_id_set = {str(n.get("id", "")) for n in graph_nodes}
    node_id_list = [str(n.get("id", "")) for n in graph_nodes]

    # Precompute in-degree, out-degree, total-degree
    in_degrees = {nid: 0 for nid in node_id_set}
    out_degrees = {nid: 0 for nid in node_id_set}
    out_bytes = {nid: 0.0 for nid in node_id_set}
    out_flows = {nid: 0 for nid in node_id_set}

    for e in graph_edges:
        src = str(e.get("source", ""))
        tgt = str(e.get("target", ""))
        bc = float(e.get("byte_count", 0.0))
        fc = int(e.get("flow_count", 1))
        if src in out_degrees:
            out_degrees[src] += 1
            out_bytes[src] += bc
            out_flows[src] += fc
        if tgt in in_degrees:
            in_degrees[tgt] += 1

    total_degrees = {nid: in_degrees[nid] + out_degrees[nid] for nid in node_id_set}

    # Step 1: Honest Starting Node Determination
    start_node = None
    start_reason = None
    start_derivation = None

    # Check flagged flows first
    for f in flagged_flows:
        src_cand = str(f.get("source", ""))
        dst_cand = str(f.get("destination", ""))
        fid = f.get("id", "FLW")
        freason = f.get("reason", "Anomalous flow")
        if src_cand in node_id_set:
            start_node = src_cand
            start_reason = f"Alert driver: Source endpoint of flagged flow {fid} ({freason})"
            start_derivation = "flagged_flow_src"
            break
        elif dst_cand in node_id_set:
            start_node = dst_cand
            start_reason = f"Alert driver: Destination endpoint of flagged flow {fid} ({freason})"
            start_derivation = "flagged_flow_dst"
            break

    # If no match in flagged flows, derive from highest-weighted node in window graph
    if not start_node:
        sorted_nodes = sorted(
            node_id_list,
            key=lambda nid: (out_bytes[nid], out_flows[nid], out_degrees[nid], total_degrees[nid]),
            reverse=True,
        )
        top_nid = sorted_nodes[0]
        start_node = top_nid
        if out_bytes[top_nid] > 0 or out_flows[top_nid] > 0:
            start_reason = (
                f"Highest egress volume in window graph: {format_bytes(out_bytes[top_nid])} "
                f"({out_flows[top_nid]} flows across {out_degrees[top_nid]} outbound edges)"
            )
            start_derivation = "highest_egress_traffic"
        else:
            start_reason = f"Most connected endpoint in window graph ({total_degrees[top_nid]} total connections)"
            start_derivation = "highest_graph_degree"

    # Step 2: Propagation Walk
    visited = [start_node]
    visited_set = {start_node}
    walked_edges = []
    steps = []

    # Step k = 0
    steps.append({
        "k": 0,
        "frontier": list(visited),
        "frontier_size": len(visited),
        "added_node": start_node,
        "source_node": None,
        "justifying_edge": None,
        "edge_byte_count": 0.0,
        "edge_flow_count": 0,
        "candidate_degree": total_degrees.get(start_node, 0),
        "step_surge_bytes": 0.0,
        "cumulative_surge_bytes": 0.0,
        "step_surge_flows": 0,
        "cumulative_surge_flows": 0,
        "stopped_early": False,
        "stop_reason": None,
        "explanation": f"Patient Zero origin: {start_node} ({start_reason})",
    })

    cum_bytes = 0.0
    cum_flows = 0
    stopped = False
    stop_msg = None

    for k in range(1, max_k + 1):
        if stopped:
            steps.append({
                "k": k,
                "frontier": list(visited),
                "frontier_size": len(visited),
                "added_node": None,
                "source_node": None,
                "justifying_edge": None,
                "edge_byte_count": 0.0,
                "edge_flow_count": 0,
                "candidate_degree": 0,
                "step_surge_bytes": 0.0,
                "cumulative_surge_bytes": cum_bytes,
                "step_surge_flows": 0,
                "cumulative_surge_flows": cum_flows,
                "stopped_early": True,
                "stop_reason": stop_msg,
                "explanation": f"Propagation halted: {stop_msg}",
            })
            continue

        cand_edges = [
            e for e in graph_edges
            if str(e.get("source", "")) in visited_set and str(e.get("target", "")) not in visited_set
        ]

        if not cand_edges:
            stopped = True
            stop_msg = f"No outgoing edges from frontier to unvisited endpoints (frontier size = {len(visited)})"
            steps.append({
                "k": k,
                "frontier": list(visited),
                "frontier_size": len(visited),
                "added_node": None,
                "source_node": None,
                "justifying_edge": None,
                "edge_byte_count": 0.0,
                "edge_flow_count": 0,
                "candidate_degree": 0,
                "step_surge_bytes": 0.0,
                "cumulative_surge_bytes": cum_bytes,
                "step_surge_flows": 0,
                "cumulative_surge_flows": cum_flows,
                "stopped_early": True,
                "stop_reason": stop_msg,
                "explanation": f"Propagation halted: {stop_msg}",
            })
            continue

        # Group by candidate target
        cand_targets = {}
        for e in cand_edges:
            tgt = str(e.get("target", ""))
            cand_targets.setdefault(tgt, []).append(e)

        def rank_candidate(tgt: str):
            edges_to_tgt = cand_targets[tgt]
            tot_bytes = sum(float(e.get("byte_count", 0.0)) for e in edges_to_tgt)
            tot_flows = sum(int(e.get("flow_count", 1)) for e in edges_to_tgt)
            deg = total_degrees.get(tgt, 0)
            best_edge = max(
                edges_to_tgt,
                key=lambda e: (float(e.get("byte_count", 0.0)), int(e.get("flow_count", 1)))
            )
            return (tot_bytes, deg, tot_flows, best_edge)

        ranked = sorted(
            cand_targets.keys(),
            key=lambda tgt: (rank_candidate(tgt)[0], rank_candidate(tgt)[1], rank_candidate(tgt)[2]),
            reverse=True,
        )

        top_cand = ranked[0]
        step_bytes, cand_deg, step_flows, best_e = rank_candidate(top_cand)

        visited.append(top_cand)
        visited_set.add(top_cand)
        cum_bytes += step_bytes
        cum_flows += step_flows
        walked_edges.append(best_e)

        src_node = str(best_e.get("source", ""))
        expl = (
            f"Compromised {top_cand} from {src_node}: "
            f"{format_bytes(step_bytes)} ({step_flows} flows) on connecting edge; "
            f"candidate degree = {cand_deg}"
        )

        steps.append({
            "k": k,
            "frontier": list(visited),
            "frontier_size": len(visited),
            "added_node": top_cand,
            "source_node": src_node,
            "justifying_edge": best_e,
            "edge_byte_count": step_bytes,
            "edge_flow_count": step_flows,
            "candidate_degree": cand_deg,
            "step_surge_bytes": step_bytes,
            "cumulative_surge_bytes": cum_bytes,
            "step_surge_flows": step_flows,
            "cumulative_surge_flows": cum_flows,
            "stopped_early": False,
            "stop_reason": None,
            "explanation": expl,
        })

    return {
        "status": "success",
        "status_message": "Real graph-propagation walk computed successfully.",
        "start_node": start_node,
        "start_reason": start_reason,
        "start_derivation": start_derivation,
        "total_nodes": len(node_id_set),
        "total_edges": len(graph_edges),
        "steps": steps,
        "walked_edges": walked_edges,
        "walked_edge_keys": {(str(e.get("source")), str(e.get("target"))) for e in walked_edges},
    }

if __name__ == "__main__":
    from backend.predict import predict
    df = pd.read_parquet("data-engineering/data/ucs/ucs_windows.parquet").head(45)
    res = predict(df, source_type="flows")
    gt = res["graph_topology"]
    ff = res.get("flagged_flows", [])

    print("\n--- SAMPLE WINDOW 1 ---")
    trav1 = compute_graph_traversal(gt["graph_nodes"], gt["graph_edges"], ff, max_k=5)
    print("Status:", trav1["status"])
    print("Start node:", trav1["start_node"])
    print("Start reason:", trav1["start_reason"])
    for s in trav1["steps"]:
        print(f"k={s['k']}: frontier={s['frontier_size']} nodes (+{s['added_node']}), surge={format_bytes(s['cumulative_surge_bytes'])} ({s['cumulative_surge_flows']} flows) | {s['explanation']}")

    print("\n--- SPARSE GRAPH TEST (1 edge, early stop) ---")
    sparse_nodes = [{"id": "Node-A"}, {"id": "Node-B"}, {"id": "Node-C"}]
    sparse_edges = [{"source": "Node-A", "target": "Node-B", "byte_count": 5120.0, "flow_count": 3}]
    trav_sparse = compute_graph_traversal(sparse_nodes, sparse_edges, max_k=5)
    print("Sparse start node:", trav_sparse["start_node"])
    for s in trav_sparse["steps"]:
        print(f"k={s['k']}: frontier={s['frontier_size']} stopped={s['stopped_early']} | {s['explanation']}")

    print("\n--- EMPTY GRAPH TEST (visible failure) ---")
    trav_empty = compute_graph_traversal([], [])
    print("Empty status:", trav_empty["status"], trav_empty["status_message"])
