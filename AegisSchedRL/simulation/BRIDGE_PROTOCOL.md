## CloudSim ↔ Python Bridge Protocol (Module 6)

### Files written by Java/CloudSim per decision step
- `bridge/nodes.json` (node snapshots; see `simulation/schema_nodes.json`)
- `bridge/task.json`  (arriving task; see `simulation/schema_task.json`)
- `bridge/step.done`  (empty marker file meaning "step ready")

### Files written by Python scheduler
- `bridge/action.json` (selected safe action)
- `bridge/action.done` (empty marker file meaning "action ready")

`action.json` payload:
```json
{"action_index": 0, "node_id": 1, "node_type": "edge"}
```

### Optional: files written by Java after dispatch (recommended)
- `bridge/outcome.json`
- `bridge/outcome.done`

`outcome.json` payload:
```json
{"delay": 0.85, "energy": 0.42, "sla_satisfied": true}
```

Python uses outcome metrics for reward (Eq. 18). If absent, delay/energy default to 0.0
and you can switch to the proxy model if desired.

