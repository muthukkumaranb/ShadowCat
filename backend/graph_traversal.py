"""
SHADOWCAT Backend - Real Explainable Graph-Propagation Traversal
Deterministic, weight-based graph diffusion simulation across real network topology edges.
Replaces black-box or hardcoded heuristic node reveal formulas with transparent, defensible edge ranking:
- Primary metric: Real traffic volume on connecting edge (byte_count_sum)
- Secondary metric: Candidate node centrality tie-breaker (total graph degree)
"""

from typing import Any, Dict, List, Optional, Set, Tuple


def format_bytes(b: float) -> str:
    """Format byte count into human-readable engineering units."""
    if b < 1024:
        return f"{b:.0f} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    elif b < 1024**3:
        return f"{b/(1024**2):.1f} MB"
    else:
        return f"{b/(1024**3):.2f} GB"


def compute_graph_traversal(
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    flagged_flows: Optional[List[Dict[str, Any]]] = None,
    max_k: int = 5,
    primary_metric: str = "byte_count",
) -> Dict[str, Any]:
    """
    Computes an explainable, weight-based attack propagation walk over real per-window edge data.

    Algorithm:
    1. Determine starting node ("Patient Zero"):
       - Check if source/destination endpoint of alert-driving flagged flows exists in graph nodes.
       - Otherwise, pick the node with highest egress traffic volume (outbound byte_count_sum)
         in the current window's real graph.
       - If no real graph data exists, fail visibly with status='no_data'.
    2. Propagation walk across horizons k=1..max_k:
       - Maintain active 'frontier' of compromised nodes.
       - Identify all real outgoing edges from frontier to non-frontier nodes.
       - Rank candidate next-hop nodes primarily by total traffic volume on connecting edges
         (sum of byte_count across connecting edges from frontier).
       - Tie-break solely by candidate node's total degree in the graph (connections = centrality).
       - Add the top candidate to frontier.
       - If no outgoing edges exist, halt early and freeze frontier.
    3. Compute real volume moved into the frontier at each step (sum of justifying edge weights).
    """
    if not graph_nodes or not graph_edges:
        return {
            "status": "no_data",
            "status_message": "No graph topology or telemetry flows available for this window.",
            "start_node": None,
            "start_reason": "No graph data available",
            "start_derivation": "none",
            "total_nodes": 0,
            "total_edges": 0,
            "steps": [],
            "walked_edges": [],
            "walked_edge_keys": set(),
        }

    flagged_flows = flagged_flows or []
    node_id_set: Set[str] = {str(n.get("id", "")) for n in graph_nodes}
    node_id_list: List[str] = [str(n.get("id", "")) for n in graph_nodes]

    # Precompute degrees and egress volumes per node
    in_degrees: Dict[str, int] = {nid: 0 for nid in node_id_set}
    out_degrees: Dict[str, int] = {nid: 0 for nid in node_id_set}
    out_bytes: Dict[str, float] = {nid: 0.0 for nid in node_id_set}
    out_flows: Dict[str, int] = {nid: 0 for nid in node_id_set}

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

    total_degrees: Dict[str, int] = {nid: in_degrees[nid] + out_degrees[nid] for nid in node_id_set}

    # Step 1: Honest Starting Node Determination
    start_node: Optional[str] = None
    start_reason: Optional[str] = None
    start_derivation: Optional[str] = None

    # Priority 1: Check alert-driving flagged flows
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

    # Priority 2: Highest egress traffic volume in real window graph
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
            start_reason = f"Highest graph degree in window ({total_degrees[top_nid]} total connections)"
            start_derivation = "highest_graph_degree"

    # Step 2: Propagation Walk
    visited: List[str] = [start_node]
    visited_set: Set[str] = {start_node}
    walked_edges: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []

    # Horizon k = 0 (Patient Zero baseline)
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

        # Group edges by candidate target
        cand_targets: Dict[str, List[Dict[str, Any]]] = {}
        for e in cand_edges:
            tgt = str(e.get("target", ""))
            cand_targets.setdefault(tgt, []).append(e)

        def rank_candidate(tgt: str) -> Tuple[float, int, int, Dict[str, Any]]:
            edges_to_tgt = cand_targets[tgt]
            tot_bytes = sum(float(e.get("byte_count", 0.0)) for e in edges_to_tgt)
            tot_flows = sum(int(e.get("flow_count", 1)) for e in edges_to_tgt)
            deg = total_degrees.get(tgt, 0)
            best_e = max(
                edges_to_tgt,
                key=lambda e: (float(e.get("byte_count", 0.0)), int(e.get("flow_count", 1)))
            )
            return (tot_bytes, deg, tot_flows, best_e)

        # Primary metric: real connecting traffic volume (tot_bytes).
        # Tie-breaker only: candidate node's own graph degree (deg).
        # Deterministic secondary tie-breaker: flow count.
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
