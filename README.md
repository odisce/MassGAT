# MassGAT

**MassGAT** is a graph-based collective learning framework for untargeted detection and annotation of LC-HRMS data. By modeling LC-MS files as hierarchical graphs where nodes represent ion traces and edges represent chemical relationships (isotopes, adducts, dimers), MassGAT performs simultaneous peak detection and grouping, leveraging analytical redundancy to rescue valid, low-intensity signals.

![MassGAT Pipeline ](docs/massgat_overview_horizontal.png)

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

**2. Graph Construction (Python)**
Build PyTorch Geometric graphs (`.pt`) from the extracted `.feather` tables.
```bash
python scripts/build_graphs.py --input_dir  data/test/ionshed_output --output_dir data/test/graphs --ionization_mode positive
```

**3. Model Inference (Python)**
Pass the constructed graphs through the trained MassGAT model. This step evaluates probabilities, groups peaks into chemical components, and optionally annotates them.
The `--with_top_annotations` flag can be used to obtain the top 5 annotations for each component suggested by the annotation heuristic. By default, only peaks with a probability above 0.5 are kept, but you can adjust this threshold using the `--min_node_prob` argument.

```bash
python scripts/predict.py --folder data/test/graphs --data_type real --weights weights/model_positive.pth --model_type iterative --output_dir data/test/results --min_node_prob 0.5 --with_top_annotations
```



## Training

To train MassGAT on a custom or synthetic dataset, use the `train.py` script. The script uses command-line arguments to configure hyperparameters and relies on MLflow to track experiments and metrics.

```bash
python scripts/train.py --use_synthetic --synthetic_path data/synthetic_dataset
```



## Citation
