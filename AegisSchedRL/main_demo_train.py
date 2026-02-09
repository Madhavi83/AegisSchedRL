from __future__ import annotations

from pathlib import Path
import yaml
import torch

from utils.normalizer import FeatureNormalizer, MinMaxSpec
from env.state_representation import NodeSnapshot, StateBuilder
from simulation.workload_generator import SyntheticWorkloadGenerator, WorkloadConfig
from guard.hybrid_guard import HybridGuard, GuardConfig
from env.reward import RewardFunction, RewardWeights, RewardCaps
from models.ppo_agent import PPOAgent, PPOConfig
from training.trainer import Trainer, TrainLoopConfig


def _load_normalizer(cfg_path: str) -> FeatureNormalizer:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    specs = {}
    for k, v in cfg["normalization"].items():
        specs[k] = MinMaxSpec(float(v["min"]), float(v["max"]), bool(v.get("clip", True)))
    return FeatureNormalizer(specs)


def main():
    device = "cpu"
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

    # Determine state_dim
    dummy_task = SyntheticWorkloadGenerator(WorkloadConfig(arrival_rate=1.0, seed=1)).step()
    if not dummy_task:
        # create a dummy
        from env.state_representation import TaskSnapshot
        t = TaskSnapshot(task_id=0, workload=100, priority=3, deadline=20, workload_max=500, priority_max=5, deadline_max=50)
    else:
        t = dummy_task[0]
    s = sb.build(nodes, t)
    state_dim = len(s)
    action_dim = len(nodes)

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device, cfg=PPOConfig(epochs=2, minibatch_size=64))
    guard = HybridGuard(GuardConfig())
    reward_fn = RewardFunction(RewardWeights(), RewardCaps())

    trainer = Trainer(sb, agent, guard, reward_fn, device=device)
    cfg = TrainLoopConfig(steps_per_update=256, max_updates=5)

    gen = SyntheticWorkloadGenerator(WorkloadConfig(arrival_rate=2.0, seed=7))

    for upd in range(cfg.max_updates):
        total_r = 0.0
        steps = 0
        while steps < cfg.steps_per_update:
            tasks = gen.step()
            for task in tasks:
                total_r += trainer.step(nodes, task)
                steps += 1
                if steps >= cfg.steps_per_update:
                    break
            if steps == 0:
                # no arrivals, continue
                continue

        stats = trainer.update()
        print(f"Update {upd+1}/{cfg.max_updates} | avg_reward={total_r/max(1,steps):.6f} | {stats}")


if __name__ == "__main__":
    main()
