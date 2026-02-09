from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, List, Optional

from env.state_representation import TaskSnapshot


@dataclass
class WorkloadConfig:
    """
    Synthetic workload generator config.
    - arrival_rate: average tasks per time step (Poisson)
    - workload range: in "MI" or abstract compute units
    - priority levels: integer levels (e.g., 1..5)
    - deadline: relative (time steps) or absolute units; here relative steps
    """
    arrival_rate: float = 1.0

    workload_min: float = 50.0
    workload_max: float = 500.0

    priority_min: int = 1
    priority_max: int = 5

    deadline_min: float = 5.0
    deadline_max: float = 50.0

    seed: int = 42


class SyntheticWorkloadGenerator:
    """
    Generates TaskSnapshot objects consistent with Eq. (1) and the paper's workload assumptions.
    Uses Poisson arrivals per time step and uniform sampling for attributes.
    """
    def __init__(self, cfg: WorkloadConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._task_id = 0

    def _poisson(self, lam: float) -> int:
        # Simple Knuth Poisson sampler (no numpy dependency)
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return max(0, k - 1)

    def step(self) -> List[TaskSnapshot]:
        """Generate the list of tasks arriving at the current time step."""
        k = self._poisson(self.cfg.arrival_rate)
        tasks: List[TaskSnapshot] = []
        for _ in range(k):
            self._task_id += 1
            w = self.rng.uniform(self.cfg.workload_min, self.cfg.workload_max)
            p = self.rng.randint(self.cfg.priority_min, self.cfg.priority_max)
            d = self.rng.uniform(self.cfg.deadline_min, self.cfg.deadline_max)
            tasks.append(
                TaskSnapshot(
                    task_id=self._task_id,
                    workload=w,
                    priority=float(p),
                    deadline=d,
                    workload_max=self.cfg.workload_max,
                    priority_max=float(self.cfg.priority_max),
                    deadline_max=self.cfg.deadline_max,
                )
            )
        return tasks
