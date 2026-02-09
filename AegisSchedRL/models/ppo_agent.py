from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import math

from models.actor import ActorNet, MLPConfig
from models.critic import CriticNet
from training.rollout_buffer import RolloutBatch


@dataclass
class PPOConfig:
    gamma: float = 0.99
    clip_eps: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 5
    minibatch_size: int = 64


class PPOAgent:
    """
    PPO with clipped surrogate objective (Eq. 21) and critic MSE (Eq. 22).
    Advantage uses 1-step TD (Eq. 20). (GAE can be added later.)
    """
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu", cfg: PPOConfig = PPOConfig()):
        self.device = device
        self.cfg = cfg

        mlp_cfg = MLPConfig(input_dim=state_dim)
        self.actor = ActorNet(mlp_cfg, action_dim).to(device)
        self.critic = CriticNet(mlp_cfg).to(device)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

    @torch.no_grad()
    def get_value(self, s: torch.Tensor) -> torch.Tensor:
        return self.critic(s)

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        # Move to device
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        old_logps = batch.logps.to(self.device)
        rewards = batch.rewards.to(self.device)
        dones = batch.dones.to(self.device)
        values = batch.values.to(self.device)
        next_states = batch.next_states.to(self.device)

        # TD target and advantage (Eq. 20)
        with torch.no_grad():
            next_values = self.critic(next_states)
            targets = rewards + self.cfg.gamma * (1.0 - dones) * next_values
            advantages = targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = states.shape[0]
        idxs = torch.randperm(N, device=self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.cfg.epochs):
            for start in range(0, N, self.cfg.minibatch_size):
                mb_idx = idxs[start:start + self.cfg.minibatch_size]

                s_mb = states[mb_idx]
                a_mb = actions[mb_idx]
                oldlog_mb = old_logps[mb_idx]
                adv_mb = advantages[mb_idx]
                tgt_mb = targets[mb_idx]

                dist = self.actor.dist(s_mb)
                logp = dist.log_prob(a_mb)
                entropy = dist.entropy().mean()

                # ratio (Eq. 19)
                ratio = torch.exp(logp - oldlog_mb)

                # clipped surrogate (Eq. 21)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss (Eq. 22)
                vpred = self.critic(s_mb)
                critic_loss = nn.functional.mse_loss(vpred, tgt_mb)

                loss = actor_loss + self.cfg.value_coef * critic_loss - self.cfg.entropy_coef * entropy

                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                loss.backward()

                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         self.cfg.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

                total_actor_loss += float(actor_loss.detach().cpu())
                total_critic_loss += float(critic_loss.detach().cpu())
                total_entropy += float(entropy.detach().cpu())

        denom = max(1, (self.cfg.epochs * math.ceil(N / self.cfg.minibatch_size)))
        return {
            "actor_loss": total_actor_loss / denom,
            "critic_loss": total_critic_loss / denom,
            "entropy": total_entropy / denom,
        }
