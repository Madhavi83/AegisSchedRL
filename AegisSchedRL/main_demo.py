from __future__ import annotations

import json
from pathlib import Path

import yaml

from utils.normalizer import FeatureNormalizer, MinMaxSpec
from env.state_representation import NodeSnapshot
from env.state_representation import StateBuilder, TaskSnapshot
from env.action_space import DiscreteActionSpace, NodeInfo

from simulation.workload_generator import SyntheticWorkloadGenerator, WorkloadConfig


def _load_normalizer(cfg_path: str) -> FeatureNormalizer:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    specs = {}
    for k, v in cfg["normalization"].items():
        specs[k] = MinMaxSpec(float(v["min"]), float(v["max"]), bool(v.get("clip", True)))
    return FeatureNormalizer(specs)


def main():
    # Demo-only: create a toy edge-cloud snapshot (normally produced by CloudSim)
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

    normalizer = _load_normalizer("config/system_config.yaml")
    sb = StateBuilder(normalizer)

    # Action space
    action_space = DiscreteActionSpace([
        NodeInfo(1, "edge"), NodeInfo(2, "edge"), NodeInfo(3, "cloud")
    ])

    # Synthetic workload
    gen = SyntheticWorkloadGenerator(WorkloadConfig(arrival_rate=2.0, seed=7))
    tasks = gen.step()
    if not tasks:
        print("No tasks arrived in this demo step. Re-run main.")
        return

    t0 = tasks[0]
    s = sb.build(nodes, t0)

    print("State vector length:", len(s))
    print("First 12 values (preview):", [round(x, 4) for x in s[:12]])

    # Example: pick the best edge by lowest latency as a naive placeholder policy
    best_action = min(range(action_space.n), key=lambda a: nodes[a].lat_ms)
    chosen = action_space.node_for_action(best_action)
    print("Chosen action index:", best_action, "=> node:", chosen)


if __name__ == "__main__":
    main()
