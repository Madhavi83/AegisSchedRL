from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from training.rollout_buffer import RolloutBuffer
from models.ppo_agent import PPOAgent
from guard.hybrid_guard import HybridGuard, GuardConfig
from env.reward import RewardFunction, RewardWeights, RewardCaps, StepOutcome
from evaluation.metrics import load_imbalance, DelayEnergyModel
from env.state_representation import NodeSnapshot, TaskSnapshot, StateBuilder


@dataclass
class TrainLoopConfig:
    steps_per_update: int = 512
    max_updates: int = 50


class Trainer:
    """
    Minimal trainer that demonstrates end-to-end: state -> actor -> guard -> reward -> buffer -> PPO update.
    CloudSim integration will plug into the step() method later.
    """
    def __init__(self, state_builder: StateBuilder, agent: PPOAgent, guard: HybridGuard, reward_fn: RewardFunction, device: str = "cpu"):
        self.sb = state_builder
        self.agent = agent
        self.guard = guard
        self.reward_fn = reward_fn
        self.device = device
        self.buffer = RolloutBuffer(device=device)
        self.dem = DelayEnergyModel()

    def step(self, nodes: List[NodeSnapshot], task: TaskSnapshot) -> float:
        # Build state
        s = self.sb.build(nodes, task)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Actor proposes action
        a, logp = self.agent.actor.act(s_t)
        a_idx = int(a.item())

        # Guard validates
        decision = self.guard.validate_or_fallback(nodes=nodes, task=task, proposed_action_index=a_idx)

        chosen = nodes[decision.safe_action_index]

        # Placeholder outcome metrics (CloudSim will replace these)
        delay = self.dem.estimate_delay(chosen, task.workload)
        energy = self.dem.estimate_energy(chosen, task.workload)
        sigma = load_imbalance(nodes)
        sla_ok = self.guard.deadline_feasible(chosen, task)

        r = self.reward_fn.compute(StepOutcome(delay=delay, energy=energy, load_imbalance=sigma,
                                              sla_satisfied=sla_ok, used_fallback=decision.used_fallback))

        # Next state placeholder: in CloudSim, next nodes would change; for now reuse nodes
        s_next = self.sb.build(nodes, task)

        # Value estimate
        with torch.no_grad():
            v = self.agent.critic(s_t).squeeze(0)

        # Store transition
        self.buffer.add(s, a_idx, float(logp.item()), float(r), 0.0, float(v.item()), s_next)
        return float(r)

    def update(self):
        batch = self.buffer.get()
        stats = self.agent.update(batch)
        self.buffer.clear()
        return stats
