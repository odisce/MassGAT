# MassGAT

**MassGAT** is a graph-based collective learning framework for the untargeted detection and annotation of LC-HRMS (Liquid Chromatography – High-Resolution Mass Spectrometry) data.

Instead of treating peak detection and annotation as two separate steps, MassGAT models each LC-MS file as a set of hierarchical graphs, where:
- **nodes** are individual ion traces (segmented signal regions),
- **edges** are candidate chemical relationships between them (isotopes, adducts, fragments, dimers).

A Graph Attention Network (GAT) then jointly decides which nodes are true peaks and which edges are true chemical links, using the redundancy between related ion species to rescue valid but low-intensity signals that single-peak methods would discard.

![MassGAT Pipeline](docs/massgat_overview_horizontal.png)
The pipeline runs in four stages — Trace Segmentation → Graph Construction → Model Inference → Annotation — matching the sections under [Usage](#usage) below.

---

## Table of Contents

- [MassGAT](#massgat)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation (Local)](#installation-local)
    - [Option A: Singularity / Docker](#option-a-singularity--docker)
    - [Option B : Local (Linux/MacOS/Windows)](#option-b--local-linuxmacoswindows)
      - [Prerequisites](#prerequisites)
      - [Set-up the environment](#set-up-the-environment)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Interactive Tutorial](#interactive-tutorial)
    - [Command Line Interface](#command-line-interface)
  - [Training](#training)
  - [Citation](#citation)

## Requirements

Installing **MassGAT** can be done from a local environment or directly inside containers:
- **Local:** [Pixi >= 0.73](https://pixi.prefix.dev/latest/) and on Windows: [Rtools](https://cran.r-project.org/bin/windows/Rtools/).
- **Containers:** [Docker](https://docs.docker.com/) (Windows, Ubuntu) or [Apptainer](https://apptainer.org/) (Ubuntu).


## Installation

## Installation

First, clone this repository (eg.: using [git](https://git-scm.com/)):

```bash
git clone https://github.com/odisce/MassGAT.git
cd MassGAT
```

### Local (Linux/MacOS/Windows)

#### Set-up the environment

| with Pixi   |  |
| :-------- | :--------- |
| to use cpu | `pixi shell -e cpu` |
| to use gpu | `pixi shell -e gpu` |


#### Install R dependencies :
```bash
pixi run install-ionshed
```

## Configuration


MassGAT allows you to tune the analytical pipeline without modifying the source code via the config.yaml file located in the root directory. Key sections include:

1. **Peak Picking (peak_picking):** Configure standard LC-MS parameters (ppm, rttol) passed directly to the [ionshed](https://github.com/odisce/ionshed) segmentation algorithm.
2. **Graph (graph):** Adjust rt_window and mz_tolerance to define how nodes are linked geographically in the dataset.
3. **Thresholds (thresholds):** Tune correlation thresholds (isotope_corr, adduct_corr) to make chemical edge assignments more or less strict. Lowering these will yield denser graphs.




## Usage

### Interactive Tutorial

We provide an interactive tool using Jupyter Notebook to demonstrate the MassGAT pipeline ([ionshed](https://github.com/odisce/ionshed) segmentation, graph construction, inference, and component vizualization) on a test file.
We provide an interactive tool using Jupyter Notebook to demonstrate the MassGAT pipeline ([ionshed](https://github.com/odisce/ionshed) segmentation, graph construction, inference, and component vizualization) on a test file.

1. Create a directory in the `data` folder with your project name, create a `mzml` subdirectory, and place your .mzML (or .mzXML) files inside it. For testing, you can also use the provided test file in the `data/test` folder.

2. Install Jupyter integrations in `massgat_env`: ```bash conda install -c conda-forge jupyter rpy2 -y```

3. Open the `massgat.ipynb` notebook, select the `massgat_env` kernel, and follow the instructions in each cell to run the pipeline.

4. The results will be saved in the `results` subfolder of your project directory, and you can explore the output tables and visualizations directly in the notebook.



### Command Line Interface

For batch processing or execution on remote servers, you can run the pipeline sequentially from the terminal.

**Data Preparation**
A small example is provided in a project called `test` and can be run directly but if to use other files:
- Replace `test` by your project name in all the following commands
- First, set up your input directory by creating a folder named `data/test/mzml` in the root of your repository.
- Place your LC-MS `.mzML` or `.mzXML` files (centroid mode) inside.

The structure looks like this:
```text
MassGAT/
└── data/
    └── test/
        └── mzml/
            ├── sample_01.mzML
            └── sample_02.mzML
```

**1. Trace Segmentation (R)**
Run the IonShed script to extract peaks and Region of Interest (ROI) signals into `.feather` format.
```bash
Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output
```

**Configuration Change Detection:** If you modify parameters in `config.yaml` (e.g., `ppm`, `rttol`, `mz_res`, etc.), the script automatically detects the change and reprocesses files. To force reprocessing without changing config, use:
```bash
Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output --force
```

**2. Graph Construction (Python)**
Build PyTorch Geometric graphs (`.pt`) from the extracted `.feather` tables.
```bash
python scripts/build_graphs.py --input_dir data/test/ionshed_output --output_dir data/test/graphs --ionization_mode positive
```

**Configuration Change Detection:** If you modify graph parameters in `config.yaml` (e.g., `rt_window`, `mz_tolerance`, correlation thresholds), the script automatically detects and reprocesses. To force reprocessing without changing config:
```bash
python scripts/build_graphs.py --input_dir data/test/ionshed_output --output_dir data/test/graphs --ionization_mode positive --force
```

**3. Model Inference (Python)**
Pass the constructed graphs through the trained MassGAT model. This step evaluates probabilities, groups peaks into chemical components, and optionally annotates them.
The `--with_top_annotations` flag can be used to obtain the top 5 annotations for each component suggested by the annotation heuristic. By default, only peaks with a probability above 0.5 are kept, but you can adjust this threshold using the `--min_node_prob` argument.

```bash
python scripts/predict.py --folder data/test/graphs --data_type real --weights weights/model_positive.pth --model_type iterative --output_dir data/test/results --min_node_prob 0.5 --with_top_annotations
```


Use `weights/model_negative.pth` for negative-mode data, matching the `--ionization_mode` used in Step 2.

Key arguments:
- `--min_node_prob`: probability threshold above which a peak is kept (default: `0.5`).
- `--with_top_annotations`: also outputs the top 5 candidate annotations per component, as ranked by the annotation heuristic.

### Configuration Management & Change Detection

The pipeline automatically detects when parameters in `config.yaml` change, ensuring reproducibility:

- **Trace Segmentation** tracks: `peak_picking` parameters (ppm, rttol, mz_res, rt_res, xic_rttol)
- **Graph Construction** tracks: `graph` and `thresholds` sections (rt_window, mz_tolerance, correlation thresholds)
- Config metadata is stored as hidden `.config_*.json` files in output directories

**Behavior:**
1. On first run or when config changes → files are reprocessed automatically
2. If config remains unchanged → existing files are reused (skip processing)
3. To force reprocessing without config changes, use `--force` flag

This ensures that changing configuration parameters automatically triggers a rebuild, preventing issues where old parameter values would be silently reused with new settings.


### Understanding the Output

Results are written to the `--output_dir` you specified (e.g. `data/test/results`). This includes:
- **A final peak list** with detection probabilities and, if `--with_top_annotations` was used, the top-ranked candidate annotations per component (main ion, isotopes, adducts, losses, dimers) and the estimated neutral mass.
- **Pseudo-spectra visualizations** — per-component views of validated ion species (m/z vs. intensity) alongside their elution profiles, which can be explored interactively in the notebook (see [Interactive Tutorial](#interactive-tutorial-recommended-for-first-use)).

---

## Training a Custom Model

To retrain MassGAT on custom or synthetic data, first generate synthetic training data, then run the training script.

### Step 1: Generate Synthetic Training Data

Generate synthetic components for each family type:
```bash
python scripts/demo_generate_graphs.py --family <FAMILY_TYPE> --num <NUM_SAMPLES> --out synthetic_dataset/<FAMILY_TYPE> --mode positive --seed 42
```

`<FAMILY_TYPE>` options:

| Value | Description |
|---|---|
| `true_with_noise` | True peaks with chemical relationships (isotopes, adducts, dimers) + noise peaks |
| `true` | True peaks only, with chemical relationships |
| `false` | False peaks only (noise) |
| `true_single` | Single isolated true peaks |
| `false_single` | Single isolated false peaks |

**Example** — generate 10,000 `true_with_noise` samples:
```bash
python scripts/demo_generate_graphs.py --family true_with_noise --num 10000 --out synthetic_dataset/true_with_noise --mode positive --seed 42
```

### Step 2: Train MassGAT

Basic training with default hyperparameters:
```bash
python scripts/train.py --use_synthetic --synthetic_path synthetic_dataset --num_epochs 15 --batch_size 64 --learning_rate 0.0003 --seed 42
```

**Key training arguments:**

| Argument | Description | Default |
|---|---|---|
| `--synthetic_path` | Path to synthetic dataset directory | — |
| `--num_epochs` | Number of training epochs | `15` |
| `--batch_size` | Batch size | `64` |
| `--learning_rate` | Learning rate | `0.0003` |
| `--hidden_dim` | Hidden dimension in the GNN | `48` |
| `--embedding_dim` | Embedding dimension | `24` |
| `--encoder_type` | Signal encoder: `transformer` or `cnn` | `transformer` |
| `--block_type` | GNN block type: `gate`, `gat`, or `gcn` | `gate` |
| `--augmentation` | Data augmentation: `none`, `node_drop`, `edge_drop`, `feature_noise`, `feature_mask`, `mixed` | `none` |
| `--seed` | Random seed for reproducibility | `42` |

Trained weights are saved to the `weights/` directory, named after the ionization `--mode` used during data generation (e.g. `model_positive.pth`, `model_negative.pth`) — matching the files you'd point `predict.py` at in [Step 3 — Model Inference](#step-3--model-inference-python).

**Tracking experiments with MLflow:**
```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser to view training metrics and compare runs.

---

## Performance & System Notes

- Inference runs entirely on **CPU** — no GPU required.
- Typical end-to-end runtime per file (segmentation + graph construction + inference + annotation), on 5 CPU cores:
  - ~2.3 minutes for a Time-of-Flight (TOF) file
  - ~8.4 minutes for an Orbitrap (QE HF) file
- The model itself is lightweight (~98,000 trainable parameters), so retraining is feasible on CPU-only machines as well.

---

## Citation
