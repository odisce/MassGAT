# MassGAT: a graph-based collective learning approach for untargeted detection and annotation of LC-MS data


## 1. Project Overview
MassGAT is a **Graph Neural Network (GNN)** framework for LC-MS1 data processing. It performs **peak validation** by a collective learning approach, taking advantage of analytical redundancy.

* **Input:** Raw LC-MS data converted into ROIs (Regions of Interest).
* **Graph Construction:** Nodes = ROIs; Edges = Chemical relationships ($\Delta m/z$ + correlation).
* **Task:** Multi-task classification (Node Real/Noise, Edge Valid/Invalid, Component Real/Noise).

---


---

## 2. Repository Structure

```text
.
├── scripts/          # Main scripts (train, predict)
├── Rscripts/         # R scripts for data processing (Ionshed, XCMS, CliqueMS)
├── src/              # Source code
│   ├── data/         # Data loading and synthetic generation
│   ├── models/       # Neural network architecture
│   ├── training/     # Training logic and loss functions
│   └── utils/        # Utility functions (annotation, visualization)
├── Dockerfile        # Container configuration
├── requirements.txt  # Python dependencies
└── LICENSE           # License
```


## Setup



## Configuration

Key parameters in `config.yaml`:
```yaml
# Model architecture
hidden_channels: 64
embedding_dim: 64
gnn_layers: 3

# Training
learning_rate: 0.0003
epochs: 30
alpha: 1.0  # Binary loss weight
beta: 1.0   # Clustering loss weight

# Memory management
max_nodes_per_graph: 50000
use_attention: true
```

## Dependencies

Install with: `pip install -r requirements.txt`

Main ones: PyTorch, PyTorch Geometric, pandas, scikit-learn, plotly, jupyter