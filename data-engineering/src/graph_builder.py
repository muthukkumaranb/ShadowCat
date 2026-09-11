"""
Stage 5: Graph Construction per Window Module
Constructs per-window directed interaction graphs from flow topology.
Produces edge lists with aggregated attributes and integer node lookup mappings.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class GraphTopologyBuilder:
    """
    Manages integer node ID assignments and builds per-window graph edge lists.
    """
    def __init__(self):
        self.node_to_id: Dict[str, int] = {}
        self.id_to_node: Dict[int, str] = {}
        self._next_id: int = 1

    def get_or_create_node_id(self, node_key: str) -> int:
        """Returns stable integer ID for a given node endpoint string."""
        if node_key not in self.node_to_id:
            node_id = self._next_id
            self.node_to_id[node_key] = node_id
            self.id_to_node[node_id] = node_key
            self._next_id += 1
            return node_id
        return self.node_to_id[node_key]

    def build_window_edge_lists(
        self,
        df_flows: pd.DataFrame,
        interval_sec: int = 60,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, any]]:
        """
        Builds graph edge list per 1-minute window from flow records.
        """
        if "timestamp_utc" not in df_flows.columns:
            raise ValueError("DataFrame must contain 'timestamp_utc'.")

        freq_str = f"{interval_sec}s"
        df_flows["window_start_utc"] = df_flows["timestamp_utc"].dt.floor(freq_str)
        source_day = df_flows["source_day"].iloc[0] if "source_day" in df_flows.columns and len(df_flows) > 0 else "unknown"

        # Determine endpoint representations from flow attributes
        # Since raw AWS CSVs have destination_port and protocol, construct stable endpoint signatures:
        # Source client cluster: 'ClientGroup_{protocol}_{fwd_header_len_or_win}'
        # Destination service: 'SvcPort_{destination_port}_{protocol}'
        
        has_src_ip = "source_id" in df_flows.columns or "src_ip" in df_flows.columns
        has_dst_ip = "destination_id" in df_flows.columns or "dst_ip" in df_flows.columns

        if has_src_ip and has_dst_ip:
            src_col = "source_id" if "source_id" in df_flows.columns else "src_ip"
            dst_col = "destination_id" if "destination_id" in df_flows.columns else "dst_ip"
            df_flows["_src_endpoint"] = df_flows[src_col].astype(str)
            df_flows["_dst_endpoint"] = df_flows[dst_col].astype(str)
        else:
            # Flow-derived topology with protocol/port clustering
            df_flows["_src_endpoint"] = "Client_Proto" + df_flows["protocol"].astype(str) + "_Win" + df_flows["window_size_fwd"].fillna(0).astype(int).astype(str)
            df_flows["_dst_endpoint"] = "SvcPort_" + df_flows["destination_port"].astype(str) + "_P" + df_flows["protocol"].astype(str)

        # Map to integer node IDs
        unique_endpoints = set(df_flows["_src_endpoint"]).union(set(df_flows["_dst_endpoint"]))
        for ep in unique_endpoints:
            self.get_or_create_node_id(ep)

        df_flows["src_node_id"] = df_flows["_src_endpoint"].map(self.node_to_id)
        df_flows["dst_node_id"] = df_flows["_dst_endpoint"].map(self.node_to_id)

        # Aggregate directed edges per window
        edge_groups = df_flows.groupby(["window_start_utc", "src_node_id", "dst_node_id"])

        edge_df = edge_groups.agg(
            flow_count=("protocol", "count"),
            byte_count_sum=("byte_count_fwd", lambda s: float(s.sum()) if "byte_count_fwd" in df_flows.columns else 0.0),
            packet_count_sum=("packet_count_fwd", lambda s: float(s.sum()) if "packet_count_fwd" in df_flows.columns else 0.0),
            duration_mean_sec=("duration_sec", lambda s: float(s.mean()) if "duration_sec" in df_flows.columns else 0.0),
            protocol_mode=("protocol", lambda s: int(s.mode().iloc[0]) if not s.empty else 6),
        ).reset_index()

        # Add window_id
        edge_df["window_id"] = [
            f"W_{source_day}_{ts.strftime('%Y%m%d_%H%M%S')}"
            for ts in edge_df["window_start_utc"]
        ]
        edge_df["source_day"] = source_day

        # Build node lookup table DataFrame
        node_lookup_df = pd.DataFrame([
            {"node_id": nid, "endpoint_identifier": name}
            for nid, name in self.id_to_node.items()
        ]).sort_values("node_id").reset_index(drop=True)

        audit = {
            "source_day": source_day,
            "total_edges": len(edge_df),
            "total_unique_nodes": len(self.node_to_id),
            "avg_edges_per_window": float(len(edge_df) / df_flows["window_start_utc"].nunique()) if df_flows["window_start_utc"].nunique() > 0 else 0,
        }

        # Cleanup internal temporary columns
        df_flows.drop(columns=["_src_endpoint", "_dst_endpoint", "src_node_id", "dst_node_id"], errors="ignore", inplace=True)

        return edge_df, node_lookup_df, audit
