from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from env.state_representation import NodeSnapshot


def node_utilization(ns: NodeSnapshot) -> float:
    """u_i(t) = 1 - C_i(t)/C_i^max"""
    if ns.cpu_max <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (ns.cpu_avail / ns.cpu_max)))


def load_imbalance(nodes: List[NodeSnapshot]) -> float:
    """
    σ_u(t) = sqrt((1/N) Σ (u_i - ū)^2)
    """
    if not nodes:
        return 0.0
    utils = [node_utilization(n) for n in nodes]
    mean_u = sum(utils) / len(utils)
    var = sum((u - mean_u) ** 2 for u in utils) / len(utils)
    return math.sqrt(var)


@dataclass
class DelayEnergyModel:
    """
    Simple, deterministic proxy model to compute delay/energy from node+task+network.
    This is used as a placeholder until CloudSim returns true execution metrics.

    Units:
      - delay in "time units"
      - energy in "energy units"
    """
    exec_coeff: float = 1.0
    queue_coeff: float = 1.0
    tx_energy_coeff: float = 0.01   # scales with latency and workload for offloading
    compute_energy_coeff: float = 0.001  # scales with workload

    def estimate_delay(self, node: NodeSnapshot, workload: float) -> float:
        cpu = max(node.cpu_avail, 1e-9)
        exec_time = self.exec_coeff * (workload / cpu)
        queue_wait = self.queue_coeff * (node.queue_len / cpu)
        net = node.lat_ms / 1000.0
        return queue_wait + exec_time + net

    def estimate_energy(self, node: NodeSnapshot, workload: float) -> float:
        compute = self.compute_energy_coeff * workload
        tx = self.tx_energy_coeff * (node.lat_ms / 1000.0) * workload
        # Cloud typically implies more transmission; approximate by higher latency already
        return compute + tx
