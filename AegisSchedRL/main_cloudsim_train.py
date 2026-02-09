from __future__ import annotations

from pathlib import Path
import yaml

import torch

from utils.normalizer import FeatureNormalizer, MinMaxSpec
from env.state_representation import StateBuilder
from simulation.cloudsim_interface import CloudSimFileBridge, CloudSimBridgeConfig
from simulation.cloudsim_runner import CloudSimLoop
from guard.hybrid_guard import HybridGuard, GuardConfig
from env.reward import RewardFunction, RewardWeights, RewardCaps
from models.ppo_agent import PPOAgent, PPOConfig
from training.rollout_buffer import RolloutBuffer


def _load_normalizer(cfg_path: str) -> FeatureNormalizer:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    specs = {}
    for k, v in cfg["normalization"].items():
        specs[k] = MinMaxSpec(float(v["min"]), float(v["max"]), bool(v.get("clip", True)))
    return FeatureNormalizer(specs)


def main():
    device = "cpu"

    # IMPORTANT: state_dim depends on number of nodes; we infer it from first CloudSim step.
    normalizer = _load_normalizer("config/system_config.yaml")
    sb = StateBuilder(normalizer)

    bridge = CloudSimFileBridge(CloudSimBridgeConfig(bridge_dir="bridge"))

    # Wait first step to infer state_dim and action_dim
    nodes, task = bridge.wait_for_step()
    s = sb.build(nodes, task)
    state_dim = len(s)
    action_dim = len(nodes)

    # Re-create bridge marker consumption already happened; Java should write again for actual run.
    # (Alternative: you can keep nodes/task and proceed; but this helps keep handshake simple.)

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device,
                     cfg=PPOConfig(epochs=2, minibatch_size=64))
    guard = HybridGuard(GuardConfig())
    reward_fn = RewardFunction(RewardWeights(), RewardCaps())

    loop = CloudSimLoop(
        bridge=bridge,
        state_builder=sb,
        agent_actor=agent.actor,
        agent_critic=agent.critic,
        guard=guard,
        reward_fn=reward_fn,
        device=device,
    )

    buffer = RolloutBuffer(device=device)

    steps_per_update = 512
    max_updates = 50

    for upd in range(max_updates):
        total_r = 0.0
        for _ in range(steps_per_update):
            s, a, r, s_next, v, used_fb, logp = loop.step()
            total_r += r
            buffer.add(s, a, logp, r, 0.0, v, s_next)

        stats = agent.update(buffer.get())
        buffer.clear()
        print(f"Update {upd+1}/{max_updates} | avg_reward={total_r/steps_per_update:.6f} | {stats}")


if __name__ == "__main__":
    main()
