from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from env.state_representation import NodeSnapshot, TaskSnapshot


@dataclass
class RewardWeights:
    """
    Reward weights in Eq. (18).
    """
    lambda_D: float = 1.0
    lambda_E: float = 0.5
    lambda_L: float = 0.5
    lambda_S: float = 2.0
    lambda_F: float = 0.5


@dataclass
class RewardCaps:
    """
    Normalization caps for delay and energy terms.
    """
    D_max: float = 5.0    # upper bound for normalized delay (time units)
    E_max: float = 10.0   # upper bound for normalized energy


@dataclass
class StepOutcome:
    """
    Minimal outcome from simulator for reward computation.
    Values are expected in consistent units across experiments.
    """
    delay: float                     # D_ij(t)
    energy: float                    # E_ij(t)
    load_imbalance: float            # σ_u(t)
    sla_satisfied: bool              # 1_sla(t)
    used_fallback: bool              # 1_fb(t)


class RewardFunction:
    """
    Multi-objective reward function aligned with Eq. (18):
        r_t = - (λD D~ + λE E~ + λL σ_u) + λS 1_sla - λF 1_fb
    """
    def __init__(self, weights: RewardWeights, caps: RewardCaps):
        self.w = weights
        self.c = caps

    def compute(self, outcome: StepOutcome) -> float:
        D_tilde = min(max(outcome.delay / max(self.c.D_max, 1e-9), 0.0), 1.0)
        E_tilde = min(max(outcome.energy / max(self.c.E_max, 1e-9), 0.0), 1.0)
        sla = 1.0 if outcome.sla_satisfied else 0.0
        fb = 1.0 if outcome.used_fallback else 0.0

        r = -(self.w.lambda_D * D_tilde + self.w.lambda_E * E_tilde + self.w.lambda_L * outcome.load_imbalance)
        r += self.w.lambda_S * sla
        r -= self.w.lambda_F * fb
        return float(r)
