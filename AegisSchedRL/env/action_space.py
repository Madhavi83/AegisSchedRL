from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NodeInfo:
    node_id: int
    node_type: str  # "edge" or "cloud"


class DiscreteActionSpace:
    """
    Discrete actions correspond to selecting a node index.
    Eq. (11): A = {1, 2, ..., N}
    Note: actions are 0-indexed in code.
    """
    def __init__(self, nodes: List[NodeInfo]):
        if not nodes:
            raise ValueError("ActionSpace requires at least one node.")
        self.nodes = nodes

    @property
    def n(self) -> int:
        return len(self.nodes)

    def node_for_action(self, action: int) -> NodeInfo:
        if action < 0 or action >= self.n:
            raise IndexError(f"Invalid action {action}; valid range [0, {self.n - 1}]")
        return self.nodes[action]
