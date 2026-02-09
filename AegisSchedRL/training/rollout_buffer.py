from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    logps: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_states: torch.Tensor


class RolloutBuffer:
    """
    Stores transitions for PPO updates.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.clear()

    def clear(self) -> None:
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logps: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.next_states: List[torch.Tensor] = []

    def add(self, s, a, logp, r, done, v, s_next) -> None:
        self.states.append(torch.as_tensor(s, dtype=torch.float32, device=self.device))
        self.actions.append(torch.as_tensor(a, dtype=torch.int64, device=self.device))
        self.logps.append(torch.as_tensor(logp, dtype=torch.float32, device=self.device))
        self.rewards.append(torch.as_tensor(r, dtype=torch.float32, device=self.device))
        self.dones.append(torch.as_tensor(done, dtype=torch.float32, device=self.device))
        self.values.append(torch.as_tensor(v, dtype=torch.float32, device=self.device))
        self.next_states.append(torch.as_tensor(s_next, dtype=torch.float32, device=self.device))

    def get(self) -> RolloutBatch:
        return RolloutBatch(
            states=torch.stack(self.states),
            actions=torch.stack(self.actions),
            logps=torch.stack(self.logps),
            rewards=torch.stack(self.rewards),
            dones=torch.stack(self.dones),
            values=torch.stack(self.values),
            next_states=torch.stack(self.next_states),
        )
