from __future__ import annotations

from pathlib import Path
import yaml

from env.state_representation import NodeSnapshot
from simulation.workload_generator import SyntheticWorkloadGenerator, WorkloadConfig
from guard.hybrid_guard import HybridGuard, GuardConfig
from evaluation.metrics import load_imbalance, DelayEnergyModel
from env.reward import RewardFunction, RewardWeights, RewardCaps, StepOutcome


def main():
    # Toy snapshot (CloudSim will replace these metrics later)
    nodes = [
        NodeSnapshot(node_id=1, node_type="edge",
                     cpu_avail=60, cpu_max=100, mem_avail=3, mem_max=4,
                     energy_avail=80, energy_max=100,
                     queue_len=2, queue_max=10,
                     lat_ms=20, bw_mbps=50, bw_max=100),
        NodeSnapshot(node_id=2, node_type="edge",
                     cpu_avail=30, cpu_max=100, mem_avail=2, mem_max=4,
                     energy_avail=50, energy_max=100,
                     queue_len=6, queue_max=10,
                     lat_ms=35, bw_mbps=40, bw_max=100),
        NodeSnapshot(node_id=3, node_type="cloud",
                     cpu_avail=800, cpu_max=1000, mem_avail=64, mem_max=128,
                     energy_avail=None, energy_max=None,
                     queue_len=1, queue_max=50,
                     lat_ms=120, bw_mbps=200, bw_max=500),
    ]

    # One task
    gen = SyntheticWorkloadGenerator(WorkloadConfig(arrival_rate=1.0, seed=9))
    tasks = gen.step()
    if not tasks:
        print("No tasks arrived in this demo step. Re-run main.")
        return
    task = tasks[0]

    # Proposed action (cloud)
    proposed = 2

    guard = HybridGuard(GuardConfig(latency_margin_ms=10.0, prefer_edge_for_priority_at_least=4.0))
    decision = guard.validate_or_fallback(nodes=nodes, task=task, proposed_action_index=proposed)

    chosen_node = nodes[decision.safe_action_index]

    # Placeholder delay/energy until CloudSim returns true metrics
    dem = DelayEnergyModel()
    delay = dem.estimate_delay(chosen_node, task.workload)
    energy = dem.estimate_energy(chosen_node, task.workload)

    sigma = load_imbalance(nodes)
    sla_ok = guard.deadline_feasible(chosen_node, task)

    rf = RewardFunction(RewardWeights(), RewardCaps())
    r = rf.compute(StepOutcome(delay=delay, energy=energy, load_imbalance=sigma,
                              sla_satisfied=sla_ok, used_fallback=decision.used_fallback))

    print("Task:", task)
    print("Guard decision:", decision)
    print("Delay:", round(delay, 4), "Energy:", round(energy, 4), "Sigma_u:", round(sigma, 4))
    print("SLA satisfied:", sla_ok, "Fallback:", decision.used_fallback)
    print("Reward r_t:", round(r, 6))


if __name__ == "__main__":
    main()
