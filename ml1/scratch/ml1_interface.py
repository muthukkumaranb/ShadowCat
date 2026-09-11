"""
ML1 Interface: adapter that loads ML1's per-host novelty scores and injects
them into graph node features.

The key temporal constraint: for graph at window t, we inject ML1 novelty
from window t-1 (the most recent *past* signal). This prevents information
leakage from the current or future windows.
"""

import numpy as np
import os
import json
from typing import Dict, Optional, List
from pathlib import Path
# from torch_geometric.data import Data
import torch
import logging

logger = logging.getLogger(__name__)


class ML1Adapter:
    """
    Loads ML1's per-host novelty scores and injects them into graph node features.

    Interface contract:
        - ML1 produces a directory of per-window JSON files:
              {output_dir}/{window_id}.json
          Each file maps host_id → novelty_score (float).
        - Alternatively, ML1 can produce a single .npz file with keys
          'timestamps' (array of window IDs) and 'novelty' (2D array [T, max_hosts]).

    When ML1 outputs are not yet available, this adapter operates in mock mode,
    injecting zeros or random noise for development/testing.
    """

    def __init__(
        self,
        ml1_output_dir: Optional[str] = None,
        z_dim: int = 64,
        mock_mode: bool = True,
        mock_noise_std: float = 0.0,
    ):
        """
        Args:
            ml1_output_dir: Path to ML1's output directory.
            z_dim: Dimensionality of ML1's temporal embedding (for validation).
            mock_mode: If True, generates synthetic novelty scores instead of loading.
            mock_noise_std: Std of Gaussian noise in mock mode (0 = all zeros).
        """
        self.ml1_output_dir = ml1_output_dir
        self.z_dim = z_dim
        self.mock_mode = mock_mode
        self.mock_noise_std = mock_noise_std

        # Cache: window_id → {host_id: novelty_score}
        self._cache: Dict[str, Dict[str, float]] = {}

        if ml1_output_dir is not None and os.path.isdir(ml1_output_dir):
            self.mock_mode = False
            self._preload(ml1_output_dir)
        elif not mock_mode:
            logger.warning(
                f"ML1 output dir '{ml1_output_dir}' not found. Falling back to mock mode."
            )
            self.mock_mode = True

    def _preload(self, output_dir: str) -> None:
        """Preload all ML1 novelty files from the output directory."""
        output_path = Path(output_dir)

        # Try .npz format first
        npz_files = list(output_path.glob("*.npz"))
        if npz_files:
            data = np.load(npz_files[0], allow_pickle=True)
            timestamps = data.get("timestamps", data.get("window_ids", []))
            novelty = data.get("novelty", data.get("scores", None))
            hosts = data.get("hosts", data.get("host_ids", []))

            if novelty is not None and len(timestamps) > 0:
                for i, ts in enumerate(timestamps):
                    ts_key = str(ts)
                    self._cache[ts_key] = {}
                    for j, host in enumerate(hosts):
                        if j < novelty.shape[1]:
                            self._cache[ts_key][str(host)] = float(novelty[i, j])
                logger.info(f"Loaded ML1 novelty from .npz: {len(self._cache)} windows")
                return

        # Try per-window JSON format
        json_files = sorted(output_path.glob("*.json"))
        for jf in json_files:
            window_id = jf.stem
            with open(jf, "r") as f:
                self._cache[window_id] = json.load(f)
        if self._cache:
            logger.info(f"Loaded ML1 novelty from JSON: {len(self._cache)} windows")

    def get_novelty_tminus1(
        self,
        window_id: str,
        active_hosts: list,
        host_to_idx: dict,
        previous_window_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns the novelty vector for window t-1, aligned to the current
        window's host ordering.

        Args:
            window_id: Current window identifier (used for logging).
            active_hosts: List of host IDs active in the current window.
            host_to_idx: Mapping from host_id to node index.
            previous_window_id: Window t-1 identifier. If None, returns zeros.

        Returns:
            np.ndarray of shape [N] with novelty scores for each host.
        """
        N = len(active_hosts)
        novelty = np.zeros(N, dtype=np.float32)

        if self.mock_mode:
            if self.mock_noise_std > 0:
                novelty = np.random.normal(0, self.mock_noise_std, size=N).astype(np.float32)
            return novelty

        if previous_window_id is None:
            return novelty

        prev_key = str(previous_window_id)
        if prev_key not in self._cache:
            logger.debug(f"No ML1 novelty for window {prev_key}")
            return novelty

        prev_scores = self._cache[prev_key]
        for host in active_hosts:
            idx = host_to_idx[host]
            host_key = str(host)
            if host_key in prev_scores:
                novelty[idx] = prev_scores[host_key]

        return novelty

    def inject_novelty_into_graphs(
        self,
        graphs: list,
        window_ids: list,
        host_lists: Optional[list] = None,
    ) -> list:
        """
        Batch-injects ML1 novelty into slot 10 of each graph's node features.

        For graph at index i (window t), injects novelty from window t-1
        (i.e., window_ids[i-1]).

        Args:
            graphs: List of PyG Data objects with x of shape [N, 11].
            window_ids: Corresponding window identifiers, same length as graphs.
            host_lists: Optional list of (active_hosts, host_to_idx) per graph.
                        If None, novelty injection uses mock mode with matching shape.

        Returns:
            The same list of graphs with slot 10 updated in-place.
        """
        for i, graph in enumerate(graphs):
            N = graph.x.shape[0]

            if i == 0:
                # First window has no t-1 → leave novelty as zero
                continue

            prev_window_id = str(window_ids[i - 1])

            if self.mock_mode:
                if self.mock_noise_std > 0:
                    novelty = np.random.normal(0, self.mock_noise_std, size=N).astype(np.float32)
                else:
                    novelty = np.zeros(N, dtype=np.float32)
            elif host_lists is not None:
                active_hosts, host_to_idx = host_lists[i]
                novelty = self.get_novelty_tminus1(
                    window_id=str(window_ids[i]),
                    active_hosts=active_hosts,
                    host_to_idx=host_to_idx,
                    previous_window_id=prev_window_id,
                )
            else:
                novelty = np.zeros(N, dtype=np.float32)

            graph.x[:, 10] = torch.tensor(novelty, dtype=torch.float32)

        return graphs
