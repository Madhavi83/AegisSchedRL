from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class MinMaxSpec:
    """Min-max normalization spec for one scalar feature."""
    min_val: float
    max_val: float
    clip: bool = True
    eps: float = 1e-12

    def transform(self, x: float) -> float:
        denom = (self.max_val - self.min_val)
        if abs(denom) < self.eps:
            return 0.0
        z = (x - self.min_val) / denom
        if self.clip:
            if z < 0.0:
                return 0.0
            if z > 1.0:
                return 1.0
        return z


class FeatureNormalizer:
    """Dictionary-driven feature normalizer."""
    def __init__(self, specs: Dict[str, MinMaxSpec]):
        self.specs = specs

    def normalize(self, features: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in features.items():
            spec = self.specs.get(k)
            out[k] = float(v) if spec is None else spec.transform(float(v))
        return out

    def normalize_vector(self, keys: Iterable[str], features: Dict[str, float]) -> list[float]:
        out = []
        for k in keys:
            v = float(features.get(k, 0.0))
            spec = self.specs.get(k)
            out.append(v if spec is None else spec.transform(v))
        return out
