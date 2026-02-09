from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.normalizer import FeatureNormalizer


@dataclass
class NodeSnapshot:
    node_id: int
    node_type: str                 # "edge" or "cloud"
    cpu_avail: float               # C_i(t)
    cpu_max: float                 # C_i^max
    mem_avail: float               # M_i(t)
    mem_max: float                 # M_i^max
    energy_avail: Optional[float]  # E_i(t) (may be None for cloud)
    energy_max: Optional[float]    # E_i^max
    queue_len: float               # Q_i(t)
    queue_max: float               # Q_i^max
    lat_ms: float                  # L_i(t)
    bw_mbps: float                 # B_i(t)
    bw_max: float                  # B_i^max


@dataclass
class TaskSnapshot:
    task_id: int
    workload: float     # w_j
    priority: float     # p_j
    deadline: float     # d_j
    workload_max: float # w^max
    priority_max: float # p^max
    deadline_max: float # d^max


class StateBuilder:
    """
    Builds the normalized state vector s_t.

    Eq. (7): s_t = [R(t), N(t), T(t)]
    Eq. (8): per-node resource features (normalized)
    Eq. (9): per-node network features (normalized)
    Eq. (10): task features (normalized)

    Output: flat list[float] in a deterministic node order.
    """
    def __init__(self, normalizer: FeatureNormalizer):
        self.norm = normalizer

    def _node_features_raw(self, ns: NodeSnapshot) -> Dict[str, float]:
        cpu_ratio = ns.cpu_avail / ns.cpu_max if ns.cpu_max > 0 else 0.0
        mem_ratio = ns.mem_avail / ns.mem_max if ns.mem_max > 0 else 0.0

        if ns.energy_avail is None or ns.energy_max is None or ns.energy_max <= 0:
            energy_ratio = 1.0
        else:
            energy_ratio = ns.energy_avail / ns.energy_max

        queue_ratio = ns.queue_len / ns.queue_max if ns.queue_max > 0 else 0.0

        # lat_ms is normalized via config bounds (min/max)
        bw_ratio = ns.bw_mbps / ns.bw_max if ns.bw_max > 0 else 0.0

        return {
            "cpu_ratio": cpu_ratio,
            "mem_ratio": mem_ratio,
            "energy_ratio": energy_ratio,
            "queue_ratio": queue_ratio,
            "lat_ms": ns.lat_ms,
            "bw_ratio": bw_ratio,
            "is_edge": 1.0 if ns.node_type.lower() == "edge" else 0.0,
        }

    def _task_features_raw(self, ts: TaskSnapshot) -> Dict[str, float]:
        w = ts.workload / ts.workload_max if ts.workload_max > 0 else 0.0
        p = ts.priority / ts.priority_max if ts.priority_max > 0 else 0.0
        d = ts.deadline / ts.deadline_max if ts.deadline_max > 0 else 0.0
        return {"task_workload": w, "task_priority": p, "task_deadline": d}

    def build(self, nodes: List[NodeSnapshot], task: TaskSnapshot) -> List[float]:
        if not nodes:
            raise ValueError("StateBuilder.build requires at least one node snapshot.")

        def sort_key(ns: NodeSnapshot) -> Tuple[int, int]:
            is_cloud = 1 if ns.node_type.lower() == "cloud" else 0
            return (is_cloud, ns.node_id)

        nodes_sorted = sorted(nodes, key=sort_key)

        state_vec: List[float] = []
        node_feature_keys = [
            "cpu_ratio", "mem_ratio", "energy_ratio", "queue_ratio",
            "lat_ms", "bw_ratio", "is_edge"
        ]

        for ns in nodes_sorted:
            raw = self._node_features_raw(ns)
            normed = self.norm.normalize(raw)
            state_vec.extend([normed[k] for k in node_feature_keys])

        task_feature_keys = ["task_workload", "task_priority", "task_deadline"]
        raw_t = self._task_features_raw(task)
        normed_t = self.norm.normalize(raw_t)
        state_vec.extend([normed_t[k] for k in task_feature_keys])

        return state_vec
