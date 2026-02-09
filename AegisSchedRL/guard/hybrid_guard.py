from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from env.state_representation import NodeSnapshot, TaskSnapshot


@dataclass
class GuardConfig:
    """
    Hybrid guard configuration aligned with Section 4.5 / Eq. (14)–(17).
    - latency_margin_ms: Δ in Eq. (16)
    - prefer_edge_for_priority_at_least: if task priority >= this, apply edge preference more strongly
    - hard_deadline: enforce T_ij <= d_j strictly (True) or allow soft slack (False)
    - est_exec_coeff: coefficient to estimate execution time from workload and cpu_avail
    - est_queue_coeff: coefficient to estimate queue waiting from queue_len and cpu_avail
    """
    latency_margin_ms: float = 10.0
    prefer_edge_for_priority_at_least: float = 4.0
    hard_deadline: bool = True

    est_exec_coeff: float = 1.0
    est_queue_coeff: float = 1.0

    # Fallback selection weights (lower is better for both terms)
    w_util: float = 1.0
    w_lat: float = 1.0


@dataclass
class GuardDecision:
    """
    Result of guard validation.
    """
    safe_action_index: int
    safe_node_id: int
    safe_node_type: str
    used_fallback: bool
    reason: str


class HybridGuard:
    """
    Execution-time guard that validates an actor-proposed action and applies fallback if needed.

    Checks:
      (1) Capacity feasibility (CPU+memory)  -> Eq. (14)
      (2) Deadline admissibility             -> Eq. (15)
      (3) Latency-aware edge preference      -> Eq. (16)
      (4) Fallback selection from feasible set -> Eq. (17)
    """

    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg

    # -------- Feasibility / estimation helpers --------

    @staticmethod
    def _utilization(ns: NodeSnapshot) -> float:
        # u_i(t) = 1 - C_i(t)/C_i^max (Section 4.6)
        if ns.cpu_max <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (ns.cpu_avail / ns.cpu_max)))

    def estimate_completion_time(self, ns: NodeSnapshot, task: TaskSnapshot) -> float:
        """
        Simple, simulator-agnostic completion time estimate for guard checks.
        T_ij(t) ~= queue_wait + exec_time + net_latency_component
        """
        cpu = max(ns.cpu_avail, 1e-9)
        exec_time = self.cfg.est_exec_coeff * (task.workload / cpu)

        queue_wait = self.cfg.est_queue_coeff * (ns.queue_len / max(cpu, 1e-9))

        # latency included as additive term (ms->time units normalized by 1000)
        net = ns.lat_ms / 1000.0
        return queue_wait + exec_time + net

    def capacity_feasible(self, ns: NodeSnapshot, task: TaskSnapshot, task_mem_req: Optional[float] = None) -> bool:
        """
        Eq. (14): w_j <= C_i(t) and M_j <= M_i(t).
        If task memory requirement is not modeled, skip memory check.
        """
        if task.workload > ns.cpu_avail:
            return False
        if task_mem_req is not None and task_mem_req > ns.mem_avail:
            return False
        return True

    def deadline_feasible(self, ns: NodeSnapshot, task: TaskSnapshot) -> bool:
        """
        Eq. (15): T_ij(t) <= d_j
        """
        Tij = self.estimate_completion_time(ns, task)
        if self.cfg.hard_deadline:
            return Tij <= task.deadline
        # soft: allow small slack (5%)
        return Tij <= (1.05 * task.deadline)

    def latency_edge_preferred(self, ns_edge: NodeSnapshot, ns_cloud: NodeSnapshot) -> bool:
        """
        Eq. (16): L_edge <= L_cloud - Δ
        """
        return ns_edge.lat_ms <= (ns_cloud.lat_ms - self.cfg.latency_margin_ms)

    # -------- Guard main API --------

    def validate_or_fallback(
        self,
        nodes: List[NodeSnapshot],
        task: TaskSnapshot,
        proposed_action_index: int,
        task_mem_req: Optional[float] = None,
    ) -> GuardDecision:
        if not nodes:
            raise ValueError("HybridGuard requires at least one node snapshot.")

        if proposed_action_index < 0 or proposed_action_index >= len(nodes):
            # If the actor proposes invalid index, force fallback.
            return self._fallback(nodes, task, task_mem_req, reason="invalid_action_index")

        proposed_node = nodes[proposed_action_index]

        # 1) Capacity feasibility
        if not self.capacity_feasible(proposed_node, task, task_mem_req):
            return self._fallback(nodes, task, task_mem_req, reason="capacity_infeasible")

        # 2) Deadline feasibility
        if not self.deadline_feasible(proposed_node, task):
            return self._fallback(nodes, task, task_mem_req, reason="deadline_violation")

        # 3) Latency-aware edge preference (applied for higher-priority tasks)
        if task.priority >= self.cfg.prefer_edge_for_priority_at_least:
            best_edge = self._best_edge_candidate(nodes, task, task_mem_req)
            best_cloud = self._best_cloud_candidate(nodes, task, task_mem_req)
            if best_edge is not None and best_cloud is not None:
                if self.latency_edge_preferred(best_edge, best_cloud):
                    # if proposed is cloud and edge is preferred, override
                    if proposed_node.node_type.lower() == "cloud":
                        return self._override_with(best_edge, nodes, reason="edge_preferred_for_latency")

        return GuardDecision(
            safe_action_index=proposed_action_index,
            safe_node_id=proposed_node.node_id,
            safe_node_type=proposed_node.node_type,
            used_fallback=False,
            reason="accepted",
        )

    # -------- Fallback logic (Eq. 17) --------

    def feasible_set(self, nodes: List[NodeSnapshot], task: TaskSnapshot, task_mem_req: Optional[float] = None) -> List[NodeSnapshot]:
        """
        Eq. (17): N_f(t) = { i in N | C_ij(t)=1 AND D_ij(t)=1 }
        """
        out: List[NodeSnapshot] = []
        for ns in nodes:
            if self.capacity_feasible(ns, task, task_mem_req) and self.deadline_feasible(ns, task):
                out.append(ns)
        return out

    def _fallback(self, nodes: List[NodeSnapshot], task: TaskSnapshot, task_mem_req: Optional[float], reason: str) -> GuardDecision:
        feas = self.feasible_set(nodes, task, task_mem_req)
        if not feas:
            # As last resort, pick the highest-capacity node (often cloud) to avoid deadlock.
            chosen = max(nodes, key=lambda n: n.cpu_avail)
            return self._override_with(chosen, nodes, used_fallback=True, reason=f"no_feasible_nodes:{reason}")

        # Choose min score = w_util * utilization + w_lat * normalized_latency
        lat_max = max(n.lat_ms for n in feas) if feas else 1.0
        lat_max = max(lat_max, 1e-9)

        def score(n: NodeSnapshot) -> float:
            util = self._utilization(n)
            lat = n.lat_ms / lat_max
            return self.cfg.w_util * util + self.cfg.w_lat * lat

        chosen = min(feas, key=score)
        return self._override_with(chosen, nodes, used_fallback=True, reason=f"fallback:{reason}")

    def _override_with(self, chosen: NodeSnapshot, nodes: List[NodeSnapshot], used_fallback: bool = True, reason: str = "override") -> GuardDecision:
        # Find its index in the provided node list
        idx = None
        for k, n in enumerate(nodes):
            if n.node_id == chosen.node_id and n.node_type == chosen.node_type:
                idx = k
                break
        if idx is None:
            # If not found, default to 0
            idx = 0
        return GuardDecision(
            safe_action_index=idx,
            safe_node_id=chosen.node_id,
            safe_node_type=chosen.node_type,
            used_fallback=used_fallback,
            reason=reason,
        )

    def _best_edge_candidate(self, nodes: List[NodeSnapshot], task: TaskSnapshot, task_mem_req: Optional[float]) -> Optional[NodeSnapshot]:
        feas = [n for n in nodes if n.node_type.lower() == "edge" and self.capacity_feasible(n, task, task_mem_req) and self.deadline_feasible(n, task)]
        if not feas:
            return None
        return min(feas, key=lambda n: n.lat_ms)

    def _best_cloud_candidate(self, nodes: List[NodeSnapshot], task: TaskSnapshot, task_mem_req: Optional[float]) -> Optional[NodeSnapshot]:
        feas = [n for n in nodes if n.node_type.lower() == "cloud" and self.capacity_feasible(n, task, task_mem_req) and self.deadline_feasible(n, task)]
        if not feas:
            return None
        return min(feas, key=lambda n: n.lat_ms)
