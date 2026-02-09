# AegisSchedRL
**A Guarded Actor–Critic Reinforcement Learning Framework for Adaptive Task Scheduling and Load Balancing in Edge–Cloud Environments**

## Overview
AegisSchedRL is an AI-driven task scheduling framework that integrates **Proximal Policy Optimization (PPO)** with a **hybrid guard mechanism** to ensure safe, SLA-aware, and stable decision-making in dynamic **edge–cloud computing environments**.  
The framework combines reinforcement learning with deterministic constraint enforcement to achieve low latency, balanced resource utilization, and robust performance under non-stationary workloads.

This repository provides a **fully reproducible reference implementation** aligned with the methodology, algorithms, and experimental setup described in the corresponding SCI research paper.

---

## Key Contributions
- Guarded Actor–Critic architecture for safe RL-based scheduling  
- Hybrid guard mechanism enforcing capacity, priority, deadline, and latency constraints  
- Multi-objective reward design balancing delay, energy, load, and SLA compliance  
- CloudSim Plus–integrated simulation for realistic edge–cloud evaluation  
- Modular, extensible, and reproducible research codebase  

---

## Repository Structure
```
AegisSchedRL/
├── env/
├── guard/
├── models/
├── training/
├── simulation/
├── cloudsim_java_template/
├── config/
├── main_demo_train.py
├── main_cloudsim_train.py
└── README.md
```

---

## Installation

### Python Environment
```bash
python >= 3.9
pip install numpy torch pyyaml
```

### Java Environment (CloudSim Plus)
- Java 17+
- Maven 3.8+

```bash
cd cloudsim_java_template
mvn -DskipTests package
```

---

## Running the Framework

### Synthetic Training
```bash
python main_demo_train.py
```

### CloudSim-Based Training
Terminal 1:
```bash
cd cloudsim_java_template
mvn exec:java
```

Terminal 2:
```bash
python main_cloudsim_train.py
```

---

## Reproducibility Notes
- Fixed random seeds for workload generation  
- Configurable reward weights and normalization  
- Deterministic CloudSim-based execution  
- Modular design supports ablation studies  

---

## Citation
Please cite the corresponding paper if you use this code.

---

## License
Released for academic and research use.

---

## Contact
For questions or collaborations, please open an issue.
