"""
Node mapping: IP string ↔ integer index for PyTorch Geometric graphs.
Uses per-snapshot mapping for memory efficiency (4.3M global IPs would be wasteful).
"""
import numpy as np
from typing import Dict, List, Set, Tuple


class NodeMapper:
    """Maps IP address strings to contiguous integer indices within a single snapshot."""

    def __init__(self):
        self._ip_to_idx: Dict[str, int] = {}
        self._idx_to_ip: List[str] = []

    def fit(self, source_ips: np.ndarray, dest_ips: np.ndarray) -> "NodeMapper":
        """Build mapping from all unique IPs in a snapshot."""
        all_ips = np.unique(np.concatenate([source_ips, dest_ips]))
        self._idx_to_ip = list(all_ips)
        self._ip_to_idx = {ip: idx for idx, ip in enumerate(self._idx_to_ip)}
        return self

    def transform(self, ips: np.ndarray) -> np.ndarray:
        """Convert IP strings to integer indices."""
        return np.array([self._ip_to_idx[ip] for ip in ips], dtype=np.int64)

    @property
    def num_nodes(self) -> int:
        return len(self._idx_to_ip)

    @property
    def ip_to_idx(self) -> Dict[str, int]:
        return self._ip_to_idx

    @property
    def idx_to_ip(self) -> List[str]:
        return self._idx_to_ip

    def get_ip(self, idx: int) -> str:
        return self._idx_to_ip[idx]

    def get_idx(self, ip: str) -> int:
        return self._ip_to_idx[ip]


class GlobalNodeMapper:
    """
    Optional: maintains a global IP→index mapping across all snapshots.
    Useful for tracking node persistence over time.
    """

    def __init__(self):
        self._ip_to_idx: Dict[str, int] = {}
        self._next_idx = 0

    def register(self, ips: np.ndarray) -> None:
        """Register new IPs, assigning indices to unseen ones."""
        for ip in ips:
            if ip not in self._ip_to_idx:
                self._ip_to_idx[ip] = self._next_idx
                self._next_idx += 1

    def transform(self, ips: np.ndarray) -> np.ndarray:
        return np.array([self._ip_to_idx[ip] for ip in ips], dtype=np.int64)

    @property
    def num_nodes(self) -> int:
        return self._next_idx
