# CloudSim Plus Java Template (Module 7)

This Maven project demonstrates how to implement the file-based bridge protocol for AegisSchedRL.

## What it does
- Creates a minimal CloudSim Plus simulation (edge + cloud)
- Generates synthetic tasks
- For each task:
  1) Writes `bridge/nodes.json` and `bridge/task.json`
  2) Touches `bridge/step.done`
  3) Waits for Python to write `bridge/action.json` and `bridge/action.done`
  4) Dispatches the task to the chosen VM (by node_id)
  5) Writes `bridge/outcome.json` and `bridge/outcome.done` (delay/SLA)

## Run
From this folder:
- `mvn -q -DskipTests package`
- `mvn -q exec:java`

In parallel, run the Python training script:
- `python main_cloudsim_train.py`

## Notes
This template is intentionally minimal. For a full experiment:
- Use `DatacenterBrokerSimple` to submit VMs and Cloudlets properly.
- Replace placeholder cpu_avail/queue_len with actual computed values.
- Add energy modeling (if required) or keep energy term weight small.

