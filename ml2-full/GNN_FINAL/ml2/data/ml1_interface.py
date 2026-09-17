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
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path
from torch_geometric.data import Data
import torch
import logging

logger = logging.getLogger(__name__)


class ML1Adapter:
    """
    Loads ML1's per-window novelty scores and injects them into graph node features.
    Reads from deviation_predictions.csv which contains window-level deviation_score.
    """

    def __init__(
        self,
        ml1_output_file: str = None,
        z_dim: int = 64,
        mock_mode: bool = True,
        mock_noise_std: float = 0.0,
    ):
        self.ml1_output_file = ml1_output_file
        self.z_dim = z_dim
        self.mock_mode = mock_mode
        self.mock_noise_std = mock_noise_std

        # Cache: window_id (timestamp str) → deviation_score
        self._cache: Dict[str, float] = {}

        if ml1_output_file is not None and os.path.isfile(ml1_output_file):
            self.mock_mode = False
            self._preload(ml1_output_file)
        elif not mock_mode:
            logger.warning(
                f"ML1 output file '{ml1_output_file}' not found. Falling back to mock mode."
            )
            self.mock_mode = True

    def _preload(self, output_file: str) -> None:
        """Preload ML1 novelty from CSV."""
        df = pd.read_csv(output_file)
        timestamp_column = (
            "window_start_utc" if "window_start_utc" in df.columns else "timestamp"
        )
        score_column = (
            "normalized_deviation_score"
            if "normalized_deviation_score" in df.columns
            else "deviation_score"
        )
        missing = {timestamp_column, score_column} - set(df.columns)
        if missing:
            raise ValueError(
                f"ML1 deviation CSV is missing required columns: {sorted(missing)}"
            )
        df["timestamp"] = pd.to_datetime(df[timestamp_column], utc=True)
        
        for _, row in df.iterrows():
            ts = row['timestamp']
            score = row[score_column]
            self._cache[str(ts)] = float(score)
            
        logger.info(f"Loaded ML1 novelty from CSV: {len(self._cache)} windows")

    def get_novelty_tminus1(
        self,
        window_id: str,
        N: int,
        previous_window_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns the novelty vector for window t-1.
        Since ML1 output is window-level, we broadcast the score to all N nodes.
        """
        novelty = np.zeros(N, dtype=np.float32)

        if self.mock_mode:
            if self.mock_noise_std > 0:
                novelty = np.random.normal(0, self.mock_noise_std, size=N).astype(np.float32)
            return novelty

        if previous_window_id is None:
            return novelty

        prev_key = str(previous_window_id)
        if prev_key not in self._cache:
            return novelty

        score = self._cache[prev_key]
        novelty[:] = score

        return novelty

    def inject_novelty_into_graphs(
        self,
        graphs: list,
        window_ids: list,
        host_lists: Optional[list] = None,
    ) -> list:
        """
        Batch-injects ML1 novelty into slot 10 of each graph's node features.
        """
        for i, graph in enumerate(graphs):
            N = graph.x.shape[0]

            if i == 0:
                continue

            prev_window_id = str(pd.to_datetime(window_ids[i - 1]))

            if self.mock_mode:
                if self.mock_noise_std > 0:
                    novelty = np.random.normal(0, self.mock_noise_std, size=N).astype(np.float32)
                else:
                    novelty = np.zeros(N, dtype=np.float32)
            else:
                novelty = self.get_novelty_tminus1(
                    window_id=str(pd.to_datetime(window_ids[i])),
                    N=N,
                    previous_window_id=prev_window_id,
                )

            graph.x[:, 10] = torch.tensor(novelty, dtype=torch.float32)

        return graphs
