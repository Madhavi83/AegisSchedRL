from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch

from simulation.cloudsim_interface import CloudSimFileBridge, CloudSimBridgeConfig
from env.state_representation import NodeSnapshot, TaskSnapshot, StateBuilder
from guard.hybrid_guard import HybridGuard
from env.reward import RewardFunction, StepOutcome
from evaluation.metrics import load_imbalance


@dataclass
class CloudSimStepMetrics:
    """
    Metrics optionally returned by CloudSim after dispatch (preferred path).
    If CloudSim doesn't provide these yet, you can set them to None and fall back to proxy model.
    """
    delay: Optional[float] = None
    energy: Optional[float] = None
    sla_satisfied: Optional[bool] = None


class CloudSimLoop:
    """
    End-to-end loop that connects CloudSim(+)/Java simulation to the Python scheduler.
    This is the place where the placeholder proxy model gets replaced by real CloudSim metrics.

    Protocol (file bridge):
      Java writes: bridge/nodes.json, bridge/task.json, bridge/step.done
      Python reads and writes: bridge/action.json, bridge/action.done
      Java dispatches task and (optionally) writes: bridge/outcome.json, bridge/outcome.done
    """
    def __init__(
        self,
        bridge: CloudSimFileBridge,
        state_builder: StateBuilder,
        agent_actor,
        agent_critic,
        guard: HybridGuard,
        reward_fn: RewardFunction,
        device: str = "cpu",
    ):
        self.bridge = bridge
        self.sb = state_builder
        self.actor = agent_actor
        self.critic = agent_critic
        self.guard = guard
        self.reward_fn = reward_fn
        self.device = device

        # Optional outcome files (Java can write these)
        self.outcome_path = self.bridge.base / "outcome.json"
        self.outcome_done = self.bridge.base / "outcome.done"

    def _read_outcome_if_available(self) -> CloudSimStepMetrics:
        if self.outcome_done.exists() and self.outcome_path.exists():
            try:
                data = json.loads(self.outcome_path.read_text())
                # consume marker
                try:
                    self.outcome_done.unlink()
                except FileNotFoundError:
                    pass
                return CloudSimStepMetrics(
                    delay=float(data["delay"]) if data.get("delay") is not None else None,
                    energy=float(data["energy"]) if data.get("energy") is not None else None,
                    sla_satisfied=bool(data["sla_satisfied"]) if data.get("sla_satisfied") is not None else None,
                )
            except Exception:
                return CloudSimStepMetrics()
        return CloudSimStepMetrics()

    def step(self) -> Tuple[List[float], int, float, List[float], float, bool]:
        """
        Performs one CloudSim-driven scheduling step.
        Returns: (s, action_index, reward, s_next, value, used_fallback)
        """
        nodes, task = self.bridge.wait_for_step()

        # Build state
        s = self.sb.build(nodes, task)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Actor proposes action
        with torch.no_grad():
            dist = self.actor.dist(s_t)
            a = dist.sample()
            logp = dist.log_prob(a)
            v = self.critic(s_t).squeeze(0)

        proposed = int(a.item())

        # Guard validates
        decision = self.guard.validate_or_fallback(nodes=nodes, task=task, proposed_action_index=proposed)

        # Write safe action back to CloudSim
        self.bridge.write_action(
            action_index=decision.safe_action_index,
            node_id=decision.safe_node_id,
            node_type=decision.safe_node_type,
        )

        # Optional ack (depends on Java side)
        self.bridge.wait_for_action_consumed()

        # Optional outcome from CloudSim
        outcome = self._read_outcome_if_available()

        chosen = nodes[decision.safe_action_index]
        sigma = load_imbalance(nodes)

        # If CloudSim provides delay/energy, use them; else set zeros (trainer can substitute proxy if desired)
        delay = 0.0 if outcome.delay is None else float(outcome.delay)
        energy = 0.0 if outcome.energy is None else float(outcome.energy)

        sla_ok = self.guard.deadline_feasible(chosen, task) if outcome.sla_satisfied is None else bool(outcome.sla_satisfied)

        r = self.reward_fn.compute(
            StepOutcome(
                delay=delay,
                energy=energy,
                load_imbalance=sigma,
                sla_satisfied=sla_ok,
                used_fallback=decision.used_fallback,
            )
        )

        # Next state: CloudSim should send next snapshot on next step; use current snapshot as placeholder
        s_next = s

        return s, decision.safe_action_index, float(r), s_next, float(v.item()), decision.used_fallback, float(logp.item())
