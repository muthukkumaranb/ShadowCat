"""
SHADOWCAT Backend - Real Explainable Graph-Propagation Traversal
Deterministic, weight-based graph diffusion simulation across real network topology edges.
Replaces black-box or hardcoded heuristic node reveal formulas with transparent, defensible edge ranking:
- Primary metric: Real traffic volume on connecting edge (byte_count_sum)
- Secondary metric: Candidate node target role priority (Identity/Auth > SSH Jump > Web Ingress > Core DNS > Host > Generic)
- Tertiary metric: Real packet volume on connecting edge (packet_count_sum)
- Quaternary metric: Candidate node graph centrality (total graph degree)
- Quinary metric: Total flow count on connecting edge (flow_count)
- True-tie disclosure: When all real signals are identical, disclose honestly rather than inventing reasons.
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


def classify_endpoint_role(ep_str: str) -> str:
    """
    Classify a network endpoint into a deterministic structural role based on
    real port numbers and IP subnet conventions.
    """
    ep_lower = str(ep_str).lower()

    # 1. Enclave client workstation or private subnet IP
    if (
        ep_lower.startswith("client_")
        or "enclave" in ep_lower
        or "workstation" in ep_lower
        or ep_lower.startswith("10.")
        or ep_lower.startswith("192.168.")
        or ep_lower.startswith("172.")
    ):
        return "Enclave Workstation / Internal Host"

    tokens = set(ep_lower.replace(":", "_").replace("-", "_").split("_"))

    # 2. Identity / Auth Cluster (Kerberos 88, LDAP 389, LDAPS 636)
    if any(p in tokens for p in ("88", "389", "636")) or any(k in ep_lower for k in ("auth", "kerberos", "ldap")):
        return "Identity / Auth Cluster"

    # 3. SSH Jump Host / Remote Management Gateway (SSH 22, RDP 3389)
    if any(p in tokens for p in ("22", "3389")) or any(k in ep_lower for k in ("ssh", "rdp", "jump")):
        return "SSH Jump Host"

    # 4. Web / Ingress Gateway (HTTP 80, HTTPS 443, Alt 8080, 8443)
    if any(p in tokens for p in ("80", "443", "8080", "8443")) or any(k in ep_lower for k in ("web", "http", "ingress")):
        return "Web / Ingress Gateway"

    # 5. Core DNS Resolver (DNS 53, LLMNR 5355, NetBIOS 137, 138)
    if any(p in tokens for p in ("53", "5355", "137", "138")) or any(k in ep_lower for k in ("dns", "resolver")):
        return "Core DNS Resolver"

    # 6. Generic service ports (e.g. SvcPort_52152_P6 ephemeral probe port)
    if "svcport" in ep_lower:
        return "External Service Port"

    return "External / Remote Endpoint"


def get_role_priority(role: str) -> int:
    """
    Defines the value-based target priority for lateral attack propagation.
    Higher value targets are prioritized by an adversary over generic/ephemeral endpoints:
      6: Identity / Auth Cluster (controls credential infrastructure)
      5: SSH Jump Host (pivot point to internal segments)
      4: Web / Ingress Gateway (perimeter boundary & high connectivity)
      3: Core DNS Resolver (internal network name resolution & redirection)
      2: Enclave Workstation / Internal Host (user client node)
      1: External / Remote Endpoint (outside target, minimal lateral movement value)
      0: External Service Port / Unclassified (generic/ephemeral scan ports)
    """
    r = (role or "").lower()
    if any(k in r for k in ("identity", "auth", "domain controller", "kerberos", "ldap")):
        return 6
    if any(k in r for k in ("ssh", "jump", "rdp", "remote access")):
        return 5
    if any(k in r for k in ("web", "ingress", "http", "gateway")):
        return 4
    if any(k in r for k in ("dns", "resolver", "netbios")):
        return 3
    if any(k in r for k in ("enclave", "workstation", "internal host", "host node")):
        return 2
    if any(k in r for k in ("external / remote", "remote endpoint")):
        return 1
    return 0


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
       - Rank candidate next-hop nodes:
         1) Primary metric: Total traffic volume on connecting edges (sum of byte_count).
         2) Secondary tie-breaker: Target role priority (Identity/Auth > SSH Jump > Web > DNS > Host > Generic).
         3) Tertiary tie-breaker: Connecting packet volume (sum of packet_count).
         4) Quaternary tie-breaker: Candidate node's total degree in the graph (centrality).
         5) Quinary tie-breaker: Connecting flow count.
         6) True tie: If all metrics are identical, select deterministically and disclose the tie plainly.
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

    # Map node IDs to their classified roles
    node_roles: Dict[str, str] = {}
    for n in graph_nodes:
        nid = str(n.get("id", ""))
        role = str(n.get("role", "")).strip()
        if not role or role == "Host Node":
            role = classify_endpoint_role(nid)
        node_roles[nid] = role

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
    start_role = node_roles.get(start_node) or classify_endpoint_role(start_node)
    steps.append({
        "k": 0,
        "frontier": list(visited),
        "frontier_size": len(visited),
        "added_node": start_node,
        "source_node": None,
        "justifying_edge": None,
        "edge_byte_count": 0.0,
        "edge_flow_count": 0,
        "edge_packet_count": 0.0,
        "candidate_degree": total_degrees.get(start_node, 0),
        "candidate_role": start_role,
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
                "edge_packet_count": 0.0,
                "candidate_degree": 0,
                "candidate_role": None,
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
                "edge_packet_count": 0.0,
                "candidate_degree": 0,
                "candidate_role": None,
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

        # Precompute metrics for all active candidate targets
        cand_metrics: Dict[str, Dict[str, Any]] = {}
        for tgt, edges_to_tgt in cand_targets.items():
            tot_bytes = sum(float(e.get("byte_count", 0.0)) for e in edges_to_tgt)
            tot_packets = sum(float(e.get("packet_count", 0.0)) for e in edges_to_tgt)
            tot_flows = sum(int(e.get("flow_count", 1)) for e in edges_to_tgt)
            deg = total_degrees.get(tgt, 0)
            cand_role = node_roles.get(tgt) or classify_endpoint_role(tgt)
            role_prio = get_role_priority(cand_role)
            stable_idx = node_id_list.index(tgt) if tgt in node_id_list else 999999
            best_e = max(
                edges_to_tgt,
                key=lambda e: (
                    float(e.get("byte_count", 0.0)),
                    float(e.get("packet_count", 0.0)),
                    int(e.get("flow_count", 1)),
                ),
            )
            cand_metrics[tgt] = {
                "bytes": tot_bytes,
                "role": cand_role,
                "role_prio": role_prio,
                "packets": tot_packets,
                "deg": deg,
                "flows": tot_flows,
                "stable_idx": stable_idx,
                "best_e": best_e,
            }

        # Hierarchy of real signals:
        # 1. Real connecting traffic volume (tot_bytes)
        # 2. Target role priority (role_prio)
        # 3. Secondary numeric traffic signal: packet count (tot_packets)
        # 4. Graph centrality: candidate node total degree (deg)
        # 5. Flow count: flow count on connecting edges (tot_flows)
        # 6. Stable order: deterministic preservation
        ranked = sorted(
            cand_targets.keys(),
            key=lambda tgt: (
                cand_metrics[tgt]["bytes"],
                cand_metrics[tgt]["role_prio"],
                cand_metrics[tgt]["packets"],
                cand_metrics[tgt]["deg"],
                cand_metrics[tgt]["flows"],
                -cand_metrics[tgt]["stable_idx"],
            ),
            reverse=True,
        )

        top_cand = ranked[0]
        top_info = cand_metrics[top_cand]
        step_bytes = top_info["bytes"]
        top_role = top_info["role"]
        top_role_prio = top_info["role_prio"]
        step_packets = top_info["packets"]
        cand_deg = top_info["deg"]
        step_flows = top_info["flows"]
        best_e = top_info["best_e"]

        # Track tie levels against the top candidate across all metrics
        tied_bytes = [t for t, m in cand_metrics.items() if m["bytes"] == step_bytes]
        tied_role = [t for t in tied_bytes if cand_metrics[t]["role_prio"] == top_role_prio]
        tied_packets = [t for t in tied_role if cand_metrics[t]["packets"] == step_packets]
        tied_deg = [t for t in tied_packets if cand_metrics[t]["deg"] == cand_deg]
        tied_all = [t for t in tied_deg if cand_metrics[t]["flows"] == step_flows]

        visited.append(top_cand)
        visited_set.add(top_cand)
        cum_bytes += step_bytes
        cum_flows += step_flows
        walked_edges.append(best_e)

        src_node = str(best_e.get("source", ""))

        # Honest explanation generation grounded in actual signals
        if len(cand_targets) == 1:
            expl = (
                f"Compromised {top_cand} from {src_node}: "
                f"{format_bytes(step_bytes)} ({step_flows} flows) on connecting edge; "
                f"sole remaining frontier candidate"
            )
        elif len(tied_bytes) == 1:
            expl = (
                f"Compromised {top_cand} from {src_node}: "
                f"{format_bytes(step_bytes)} ({step_flows} flows) on connecting edge; "
                f"dominant traffic volume"
            )
        elif len(tied_role) < len(tied_bytes):
            num_other = len(tied_bytes) - 1
            other_s = "candidate" if num_other == 1 else "candidates"
            expl = (
                f"Compromised {top_cand} from {src_node}: "
                f"tied on traffic volume ({format_bytes(step_bytes)}, {step_flows} flows) with {num_other} other {other_s}; "
                f"selected for higher target value ({top_role}) over generic service ports."
            )
        elif len(tied_packets) < len(tied_role):
            num_other = len(tied_role) - 1
            other_s = "candidate" if num_other == 1 else "candidates"
            expl = (
                f"Compromised {top_cand} from {src_node}: "
                f"tied on traffic volume ({format_bytes(step_bytes)}) and target value ({top_role}) with {num_other} other {other_s}; "
                f"selected for higher packet activity ({int(step_packets)} pkts)."
            )
        elif len(tied_deg) < len(tied_packets):
            num_other = len(tied_packets) - 1
            other_s = "candidate" if num_other == 1 else "candidates"
            expl = (
                f"Compromised {top_cand} from {src_node}: "
                f"tied on traffic volume, role, and packet activity with {num_other} other {other_s}; "
                f"selected for higher graph centrality (degree = {cand_deg})."
            )
        elif len(tied_all) < len(tied_deg):
            num_other = len(tied_deg) - 1
            other_s = "candidate" if num_other == 1 else "candidates"
            expl = (
                f"Compromised {top_cand} from {src_node}: "
                f"tied on volume, role, packets, and degree with {num_other} other {other_s}; "
                f"selected for higher flow count ({step_flows} flows)."
            )
        else:
            # True tie on all available real signals: disclose honestly!
            expl = (
                f"Compromised {top_cand} from {src_node}: "
                f"{len(tied_all)} candidates were equally weighted on all available real signals "
                f"({format_bytes(step_bytes)}, {top_role}, {int(step_packets)} pkts, degree {cand_deg}); "
                f"this one was selected deterministically (stable order)."
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
            "edge_packet_count": step_packets,
            "candidate_degree": cand_deg,
            "candidate_role": top_role,
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
