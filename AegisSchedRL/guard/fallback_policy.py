from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from env.state_representation import NodeSnapshot, TaskSnapshot


@dataclass
class FallbackPolicyConfig:
    """
    Standalone fallback configuration (if you want to use fallback outside HybridGuard).
    """
    w_util: float = 1.0
    w_lat: float = 1.0


class FallbackPolicy:
    """
    Deterministic fallback selector: choose feasible node that minimizes a weighted score
    based on utilization and latency.
    """
    def __init__(self, cfg: FallbackPolicyConfig):
        self.cfg = cfg

    @staticmethod
    def utilization(ns: NodeSnapshot) -> float:
        if ns.cpu_max <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (ns.cpu_avail / ns.cpu_max)))

    def select(self, feasible_nodes: List[NodeSnapshot]) -> NodeSnapshot:
        if not feasible_nodes:
            raise ValueError("FallbackPolicy.select requires a non-empty feasible node list.")
        lat_max = max(n.lat_ms for n in feasible_nodes)
        lat_max = max(lat_max, 1e-9)

        def score(n: NodeSnapshot) -> float:
            util = self.utilization(n)
            lat = n.lat_ms / lat_max
            return self.cfg.w_util * util + self.cfg.w_lat * lat

        return min(feasible_nodes, key=score)
