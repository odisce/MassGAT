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
- [Requirements](#requirements)
- [Installation](#installation)
  - [Option A: Docker](#option-a-docker)
  - [Option B: Local install](#option-b-local-install)
- [Configuration](#configuration)
  - [Chemical Rules & Mass Differences](#chemical-rules--mass-differences-mass_differencesxlsx)
- [Usage](#usage)
  - [Interactive Tutorial (recommended for first use)](#interactive-tutorial-recommended-for-first-use)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
    - [Running with Docker or Singularity](#running-with-docker-or-singularity)
    - [Step 1 — Trace Segmentation (R)](#step-1--trace-segmentation-r)
    - [Step 2 — Graph Construction (Python)](#step-2--graph-construction-python)
    - [Step 3 — Model Inference (Python)](#step-3--model-inference-python)
  - [Understanding the Output](#understanding-the-output)
- [Training a Custom Model](#training-a-custom-model)
  - [Step 1: Generate Synthetic Training Data](#step-1-generate-synthetic-training-data)
  - [Step 2: Train MassGAT](#step-2-train-massgat)
- [Performance & System Notes](#performance--system-notes)
- [Citation](#citation)

---

## Requirements

* **Running on a server / HPC:** Singularity or Apptainer.
* **Running locally:** Miniforge or Miniconda, plus R.
* **Hardware:** No GPU required for inference. MassGAT runs on CPU only (see [Performance & System Notes](#performance--system-notes) for expected runtimes). A GPU is optional and only speeds up training.

---

## Installation

First, clone the repository:
```bash
git clone https://github.com/odisce/MassGAT.git
cd MassGAT
```

### Option A: Docker

Build the Docker image locally from the provided `Dockerfile`:
```bash
docker build -t odisce/massgat:latest environment/
```

Alternatively, pull the pre-built image we publish on Docker Hub, so you don't have to build it yourself:
```bash
docker pull path/to/massgat:latest  # TODO: replace with real Docker Hub image once published
```

Either way, this is the simplest install option since R, Python, and all dependencies are already installed inside the image — skip straight to [Running with Docker or Singularity](#running-with-docker-or-singularity).



### Option B: Local install (Linux/MacOS/Windows)

Open the Miniforge/Miniconda terminal to run the setup. If you'd prefer to use the standard Windows terminal instead, first run the following inside the Miniforge/Miniconda terminal, then restart your terminal:
```bash
conda init cmd.exe
```

**1. Create and activate the environment:**
```bash
conda create --name massgat_env python=3.10 --channel conda-forge --override-channels -y
conda activate massgat_env
```

**2. Install PyTorch & PyTorch Geometric:**

For CPU-only, replace `cu118` with `cpu` in both commands below:
```bash
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

**3. Install MassGAT and its Python dependencies:**
```bash
pip install -e .
```

**4. Install R dependencies (IonShed & XCMS):**
```bash
Rscript src/massgat/r_scripts/install_packages.R
```

✅ Once installed, pretrained model weights are already included in the repository at `weights/model_positive.pth` and `weights/model_negative.pth` (positive and negative ionization mode respectively). No separate download is needed to run inference — you only need to train your own weights if you want to customize the model (see [Training a Custom Model](#training-a-custom-model)).

---

## Configuration

MassGAT lets you tune the analytical pipeline without touching the source code, via the `config.yaml` file in the repository root. Key sections:

1. **`peak_picking`**: Standard LC-MS parameters (`ppm`, `rttol`) passed directly to the IonShed segmentation algorithm.
2. **`graph`**: `rt_window` and `mz_tolerance`, which control how nodes are linked into candidate components.
3. **`thresholds`**: Correlation thresholds (`isotope_corr`, `adduct_corr`) controlling how strict chemical edge assignment is. Lowering these yields denser graphs (more candidate edges kept).

> **Note:** Command-line arguments passed to individual scripts (e.g. `--min_node_prob` in [Model Inference](#step-3--model-inference-python)) take precedence over the corresponding values in `config.yaml` for that run.

### Chemical Rules & Mass Differences (`mass_differences.xlsx`)

Beyond `config.yaml`, MassGAT relies on a customizable Excel file — `src/mass_differences.xlsx` — to define the chemical relationships it looks for. This file is used both for **synthetic data generation** and for **graph construction during inference**.

To modify the default chemical rules, open `src/mass_differences.xlsx` and edit the relevant sheet:

| Sheet | Purpose | Required columns |
|---|---|---|
| `isotopes` | Target isotopes (e.g., 13C, 15N). | `name`, `diff` (exact mass difference), `freq` (*Synthetic data generation only.*) |
| `mods_positive` / `mods_negative` | Adducts and modifications per ionization mode. | `name`, `diff`, `freq`(*Synthetic data generation only.*) |
| `dimers_positive` / `dimers_negative` | Possible dimer formations. | `name`, `diff`, `freq`(*Synthetic data generation only.*) |
| `compat_positive` / `compat_negative` | Compatibility matrix — which modifications can chain together. *Synthetic data generation only.* | `source`, `target`, `weight` (chaining weight) |

**Naming convention:** name adduct/modification rules with a `adduct_` or `loss_` prefix to ensure correct edge directionality.

**Limits:**
- Do **not** add new isotope or dimer links — the defaults are recommended as-is.
- For adducts and fragments, keep the list to **20 rules maximum**.

---

## Usage

### Interactive Tutorial (recommended for first use)

A Jupyter Notebook is provided to walk through the full pipeline (IonShed segmentation, graph construction, inference, and component visualization) on a test file.

1. Create a directory under `data/` named after your project, with an `mzml` subdirectory inside it, and place your `.mzML` (or `.mzXML`) files there. For a quick test, you can use the files already provided in `data/test`.
2. Install the Jupyter integrations into `massgat_env`:
   ```bash
   conda install -c conda-forge jupyter rpy2 -y
   ```
3. Open `massgat.ipynb`, select the `massgat_env` kernel, and follow the instructions in each cell.
4. Results are saved to the `results` subfolder of your project directory; tables and visualizations can also be explored directly inside the notebook.

### Command Line Interface (CLI)

For batch processing or execution on remote servers, run the three pipeline stages sequentially from the terminal.

#### Running with Docker or Singularity

Use **Docker** on your own machine or a server. Use **Singularity/Apptainer** on shared HPC clusters, where Docker is disabled for security reasons.

Commands are identical to the local ones below  — just add a prefix to run them inside the container, mounting your current folder.

**Docker:**
```bash
docker run --gpus all -v $(pwd):/app -w /app massgat:latest <command>
```
*(Drop `--gpus all` if using CPU only.)*

Example:
```bash
docker run --gpus all -v $(pwd):/app -w /app massgat:latest Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output
```

**Singularity:**
```bash
singularity exec --nv /path/to/massgat.sif <command>
```
*(Drop `--nv` if using CPU only.)*

Example:
```bash
singularity exec --nv /path/to/massgat.sif Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output
```

The same prefix pattern applies to all three steps below.

---

#### Step 1 — Trace Segmentation (R)

Set up your input directory first. Create `data/<project_name>/mzml/` and place your raw `.mzML`/`.mzXML` files inside:
```text
MassGAT/
└── data/
    └── project_name/
        └── mzml/
            ├── sample_01.mzML
            └── sample_02.mzML
```

Run the IonShed script to extract peaks and Region of Interest (ROI) signals into `.feather` format (replace `test` with your project folder path):
```bash
Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output
```

#### Step 2 — Graph Construction (Python)

Build PyTorch Geometric graphs (`.pt`) from the extracted `.feather` tables:
```bash
python scripts/build_graphs.py --input_dir data/test/ionshed_output --output_dir data/test/graphs --ionization_mode positive
```
Use `--ionization_mode negative` if your data was acquired in negative mode.

#### Step 3 — Model Inference (Python)

Pass the constructed graphs through the trained MassGAT model. This evaluates node/edge probabilities, groups peaks into chemical components, and (optionally) annotates them.

```bash
python scripts/predict.py --folder data/test/graphs --data_type real --weights weights/model_positive.pth --model_type iterative --output_dir data/test/results --min_node_prob 0.5 --with_top_annotations
```

Use `weights/model_negative.pth` for negative-mode data, matching the `--ionization_mode` used in Step 2.

Key arguments:
- `--min_node_prob`: probability threshold above which a peak is kept (default: `0.5`).
- `--with_top_annotations`: also outputs the top 5 candidate annotations per component, as ranked by the annotation heuristic.

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