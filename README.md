# MassGAT

**MassGAT** is a graph-based collective learning framework for untargeted detection and annotation of LC-HRMS data. By modeling LC-MS files as hierarchical graphs where nodes represent ion traces and edges represent chemical relationships (isotopes, adducts, dimers), MassGAT performs simultaneous peak detection and grouping, leveraging analytical redundancy to rescue valid, low-intensity signals.

![MassGAT Pipeline ](docs/massgat_overview_horizontal.png)


## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
  - [Option A: Docker](#option-a-docker)
  - [Option B: Local](#option-b-local)
- [Usage](#usage)
  - [Interactive Tutorial](#interactive-tutorial)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
    - [Running with Docker or Singularity](#running-with-docker-or-singularity)
    - [Trace Segmentation (R)](#trace-segmentation-r)
    - [Graph Construction (Python)](#graph-construction-python)
    - [Model Inference (Python)](#model-inference-python)
- [Training](#training)
- [Citation](#citation)

## Requirements

* **Servers:** Singularity or Apptainer.
* **Local :** Miniforge or Miniconda + R


## Installation (Local)

First, clone the repository:
```bash
git clone https://github.com/odisce/MassGAT.git
cd MassGAT
```


### Option A: Singularity / Docker

A faire quand on aura l'image sur dockerhub


### Option B : Local (Linux/MacOS/Windows)

Open the Miniforge/Miniconda terminal to run the setup. If you wish to use a standard terminal on Windows, run :

```bash
conda init cmd.exe
```

in the Miniforge/Miniconda terminal, then restart the terminal.


1. Create the environment and activate it:
```bash
conda create --name massgat_env python=3.10 --channel conda-forge --override-channels -y
conda activate massgat_env
```

2. Intsall PyTorch & PyTorch Geometric

For CPU-only, replace cu118 with cpu in the following commands:
```bash
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

3. Install MassGAT and its dependencies:
```bash
pip install -e .
```

4. Install R Dependencies (IonShed & XCMS) :
```bash
Rscript src/massgat/r_scripts/install_packages.R
```


## Configuration
MassGAT allows you to tune the analytical pipeline without modifying the source code via the config.yaml file located in the root directory. Key sections include:

1. Peak Picking (peak_picking): Configure standard LC-MS parameters (ppm, rttol) passed directly to the IonShed segmentation algorithm.
2. Graph (graph): Adjust rt_window and mz_tolerance to define how nodes are linked geographically in the dataset.
3. Thresholds (thresholds): Tune correlation thresholds (isotope_corr, adduct_corr) to make chemical edge assignments more or less strict. Lowering these will yield denser graphs.


### Chemical Rules & Mass Differences (`mass_differences.xlsx`)

In addition to the algorithmic parameters, MassGAT relies on a customizable Excel file (`src/mass_differences.xlsx`) to define the chemical relationships used. This file is used for both generating synthetic data and graph construction for the inference.

To modify the default chemical rules, open `src/mass_differences.xlsx` and edit the following sheets:

* **`isotopes`**: Defines the target isotopes (e.g., 13C, 15N). Requires the isotope `name`, exact mass difference (`diff`), and expected natural abundance/frequency (`freq`) for synthetic data generation only.
* **`mods_positive` & `mods_negative`**: Defines the adducts and modifications for each ionization mode. Requires `name`, `diff`, and `freq`. Please name your rule properly (e.g. `adduct_` for adducts, `loss_` for losses) to ensure a coherence in edge directionality.
* **`dimers_positive` & `dimers_negative`**: Defines possible dimers formations. Requires `name`, `diff`, and `freq`.
* **`compat_positive` & `compat_negative`** (for synthetic data generation only): Defines the compatibility matrix (which modifications can chain together) using `source`, `target`, and a transition `weight`.

**Important:** It is not recommended to add any new isotope or dimer link. For adducts and fragments, please put up to 20 rules.

## Usage

### Interactive Tutorial

We provide an interactive tool using Jupyter Notebook to demonstrate the MassGAT pipeline (Ionshed segmentation, graph construction, inference, and component vizualization) on a test file.

1. Create a directory in the `data` folder with your project name, create a `mzml` subdirectory, and place your .mzML (or .mzXML) files inside it. For testing, you can also use the provided test file in the `data/test` folder.

2. Install Jupyter integrations in `massgat_env`: ```bash conda install -c conda-forge jupyter rpy2 -y```

3. Open the `massgat.ipynb` notebook, select the `massgat_env` kernel, and follow the instructions in each cell to run the pipeline.

4. The results will be saved in the `results` subfolder of your project directory, and you can explore the output tables and visualizations directly in the notebook.


### Command Line Interface

For batch processing or execution on remote servers, you can run the pipeline sequentially from the terminal.


### Running with Docker or Singularity

If you are using Docker or Singularity (recommended for servers and HPC), the commands for the pipeline are exactly the same as the local ones. You just need to add a prefix to run them inside the container and link your current folder.

**For Docker:**
Add `docker run --gpus all -v $(pwd):/app -w /app odisce/massgat:latest` before any command.
*(Note: Remove `--gpus all` if you are only using a CPU).*

Example:
```bash
docker run --gpus all -v $(pwd):/app -w /app odisce/massgat:latest Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output
```

**For Singularity:**
Add `singularity exec --nv /path/to/massgat.sif` before any command.
*(Note: Remove `--nv` if you are only using a CPU).*

Example:
```bash
singularity exec --nv /path/to/massgat.sif Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output
```

All subsequent steps (Graph Construction and Model Inference) follow the same pattern.

---

### Trace Segmentation (R)

First, set up your input directory. Create a folder named `data/mzml` in the root of your repository and place your raw `.mzML` or `.mzXML` files inside it. Your structure should look like this:
```text
MassGAT/
└── data/
    └── project_name/
        └── mzml/
            ├── sample_01.mzML
            └── sample_02.mzML
```

**1. Trace Segmentation (R)**
Run the IonShed script to extract peaks and Region of Interest (ROI) signals into `.feather` format. If you create a new project folder, replace `test` with your project folder path in the command below.
```bash
Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output
```


### Graph Construction (Python)
Build PyTorch Geometric graphs (`.pt`) from the extracted `.feather` tables.
```bash
python scripts/build_graphs.py --input_dir  data/test/ionshed_output --output_dir data/test/graphs --ionization_mode positive
```

### Model Inference (Python)
Pass the constructed graphs through the trained MassGAT model. This step evaluates probabilities, groups peaks into chemical components, and optionally annotates them.
The `--with_top_annotations` flag can be used to obtain the top 5 annotations for each component suggested by the annotation heuristic. By default, only peaks with a probability above 0.5 are kept, but you can adjust this threshold using the `--min_node_prob` argument.

```bash
python scripts/predict.py --folder data/test/graphs --data_type real --weights weights/model_positive.pth --model_type iterative --output_dir data/test/results --min_node_prob 0.5 --with_top_annotations
```



## Training

To train MassGAT on a custom or synthetic dataset, you first generate synthetic data using `demo_generate_graphs.py`, then train using `train.py`.

### Step 1: Generate Synthetic Training Data

Generate synthetic datasets with different types of chemical components. Run the following command for each family type by `--family` and `--num`:

```bash
python scripts/demo_generate_graphs.py --family <FAMILY_TYPE> --num <NUM_SAMPLES> --out synthetic_dataset/<FAMILY_TYPE> --mode positive --seed 42
```

Replace `<FAMILY_TYPE>` with one of the following:

- `true_with_noise` - True peaks with chemical relationships (isotopes, adducts, dimers) + noise peaks
- `true` - True peaks only with chemical relationships
- `false` - False peaks only (noise)
- `true_single` - Single isolated true peaks
- `false_single` - Single isolated false peaks

**Example - Generate 10000 samples of `true_with_noise`:**

```bash
python scripts/demo_generate_graphs.py --family true_with_noise --num 10000 --out synthetic_dataset/true_with_noise --mode positive --seed 42
```


### Step 2: Train MassGAT

**Basic training with default hyperparameters:**

```bash
python scripts/train.py --use_synthetic --synthetic_path synthetic_dataset/training_data --num_epochs 15 --batch_size 64 --learning_rate 0.0003 --seed 42
```


**Key Training Arguments:**

- `--synthetic_path`: Path to synthetic dataset directory
- `--num_epochs`: Number of training epochs (default: 15)
- `--batch_size`: Batch size for training (default: 64)
- `--learning_rate`: Learning rate (default: 0.0003)
- `--hidden_dim`: Hidden dimension in GNN (default: 48)
- `--embedding_dim`: Embedding dimension (default: 24)
- `--encoder_type`: Signal encoder - `transformer` or `cnn` (default: transformer)
- `--block_type`: GNN block type - `gate`, `gat`, or `gcn` (default: gate)
- `--augmentation`: Data augmentation - `none`, `node_drop`, `edge_drop`, `feature_noise`, `feature_mask`, `mixed` (default: none)
- `--seed`: Random seed for reproducibility (default: 42)


Trained model weights are saved to the `weights/` directory with the filename format:


Metrics and model artifacts can also be viewed using MLflow:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser to view training metrics and compare experiments.



## Citation