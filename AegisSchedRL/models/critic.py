from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from models.actor import MLPConfig, _act


class CriticNet(nn.Module):
    """
    Critic network V_φ(s): outputs scalar value.
    """
    def __init__(self, cfg: MLPConfig):
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
        self.v = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.v(z).squeeze(-1)
