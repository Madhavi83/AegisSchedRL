from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from env.state_representation import NodeSnapshot, TaskSnapshot


@dataclass
class CloudSimBridgeConfig:
    """
    File-based bridge configuration.

    Java/CloudSim side writes:
      - nodes.json
      - task.json
      - step.done (empty file to signal 'ready')

    Python reads, computes, and optionally writes:
      - action.json
      - action.done (empty file to signal 'action ready')
    """
    bridge_dir: str = "bridge"
    poll_interval_sec: float = 0.05
    timeout_sec: float = 30.0


class CloudSimFileBridge:
    """
    Minimal, robust file-based interface between CloudSim(+)/Java and Python RL.

    This avoids sockets and keeps experiments reproducible in CI.
    """
    def __init__(self, cfg: CloudSimBridgeConfig):
        self.cfg = cfg
        self.base = Path(cfg.bridge_dir)
        self.base.mkdir(parents=True, exist_ok=True)

        # Expected filenames
        self.nodes_path = self.base / "nodes.json"
        self.task_path = self.base / "task.json"
        self.step_done = self.base / "step.done"

        self.action_path = self.base / "action.json"
        self.action_done = self.base / "action.done"

    def wait_for_step(self) -> Tuple[List[NodeSnapshot], TaskSnapshot]:
        """Block until CloudSim indicates a new step is ready, then load snapshots."""
        start = time.time()
        while True:
            if self.step_done.exists() and self.nodes_path.exists() and self.task_path.exists():
                break
            if time.time() - start > self.cfg.timeout_sec:
                raise TimeoutError("Timed out waiting for CloudSim step files.")
            time.sleep(self.cfg.poll_interval_sec)

        nodes = self._load_nodes(self.nodes_path)
        task = self._load_task(self.task_path)

        # Remove step marker to allow next step
        try:
            self.step_done.unlink()
        except FileNotFoundError:
            pass

        return nodes, task

    def write_action(self, action_index: int, node_id: int, node_type: str) -> None:
        payload = {
            "action_index": int(action_index),
            "node_id": int(node_id),
            "node_type": str(node_type),
        }
        self.action_path.write_text(json.dumps(payload, indent=2))
        self.action_done.write_text("")  # signal action ready

    def wait_for_action_consumed(self) -> None:
        """Optional: wait until Java side deletes action.done to acknowledge consumption."""
        start = time.time()
        while True:
            if not self.action_done.exists():
                return
            if time.time() - start > self.cfg.timeout_sec:
                # don't hard fail; some setups may not delete it
                return
            time.sleep(self.cfg.poll_interval_sec)

    @staticmethod
    def _load_nodes(path: Path) -> List[NodeSnapshot]:
        data = json.loads(path.read_text())
        nodes: List[NodeSnapshot] = []
        for n in data["nodes"]:
            nodes.append(
                NodeSnapshot(
                    node_id=int(n["node_id"]),
                    node_type=str(n["node_type"]),
                    cpu_avail=float(n["cpu_avail"]),
                    cpu_max=float(n["cpu_max"]),
                    mem_avail=float(n["mem_avail"]),
                    mem_max=float(n["mem_max"]),
                    energy_avail=None if n.get("energy_avail") is None else float(n["energy_avail"]),
                    energy_max=None if n.get("energy_max") is None else float(n["energy_max"]),
                    queue_len=float(n["queue_len"]),
                    queue_max=float(n["queue_max"]),
                    lat_ms=float(n["lat_ms"]),
                    bw_mbps=float(n["bw_mbps"]),
                    bw_max=float(n["bw_max"]),
                )
            )
        return nodes

    @staticmethod
    def _load_task(path: Path) -> TaskSnapshot:
        t = json.loads(path.read_text())
        return TaskSnapshot(
            task_id=int(t["task_id"]),
            workload=float(t["workload"]),
            priority=float(t["priority"]),
            deadline=float(t["deadline"]),
            workload_max=float(t["workload_max"]),
            priority_max=float(t["priority_max"]),
            deadline_max=float(t["deadline_max"]),
        )
