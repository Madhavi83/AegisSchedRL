from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: Tuple[int, ...] = (128, 128)
    activation: str = "tanh"  # "relu" | "tanh"
    dropout: float = 0.0


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class ActorNet(nn.Module):
    """
    Actor network π_θ(a|s): outputs logits over discrete actions.
    """
    def __init__(self, cfg: MLPConfig, action_dim: int):
        super().__init__()
        layers = []
        d = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(_act(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            d = h
        self.backbone = nn.Sequential(*layers)
        self.logits = nn.Linear(d, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.logits(z)

    def dist(self, x: torch.Tensor) -> Categorical:
        logits = self.forward(x)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and log-prob.
        Returns: action (B,), logp (B,)
        """
        d = self.dist(x)
        a = d.sample()
        logp = d.log_prob(a)
        return a, logp
