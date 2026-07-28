# MassGAT

**MassGAT** is a graph-based collective learning framework for untargeted detection and annotation of LC-HRMS data. By modeling LC-MS files as hierarchical graphs where nodes represent ion traces and edges represent chemical relationships (isotopes, adducts, dimers), MassGAT performs simultaneous peak detection and grouping, leveraging analytical redundancy to rescue valid, low-intensity signals.

![MassGAT Pipeline ](docs/massgat_overview_horizontal.png)

## Table of Contents

- [MassGAT](#massgat)
  - [Requirements](#requirements)
  - [Local Installation](#local-installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Interactive Tutorial](#interactive-tutorial)
    - [Command Line Interface](#command-line-interface)
    - [Using containers](#using-containers)
    - [Training MassGAT](#training-massgat)
  - [Citation](#citation)

## Requirements

**MassGAT** can be run from a local environment or directly inside a container:
- **Local:** [Pixi >= 0.73](https://pixi.prefix.dev/latest/) and on Windows: [Rtools](https://cran.r-project.org/bin/windows/Rtools/).
- **Container:** [Docker](https://docs.docker.com/) (Windows, Ubuntu) or [Apptainer](https://apptainer.org/) (Ubuntu).

## Local Installation

First, clone this repository (eg.: using [git](https://git-scm.com/)):
```bash
git clone https://github.com/odisce/MassGAT.git
cd MassGAT
```

### Set-up the environment

| with Pixi   |  |
| :-------- | :--------- |
| to use cpu | `pixi shell -e cpu --locked` |
| to use gpu | `pixi shell -e gpu --locked` |

### Install R dependencies :
```bash
pixi run install-ionshed
```


## Configuration

MassGAT can be tuned without modifying the source code via the [`config.yaml`](config.yaml). Key sections include:

1. **Peak Picking (peak_picking):** Configure standard LC-MS parameters (ppm, rttol) passed directly to the [ionshed](https://github.com/odisce/ionshed) segmentation algorithm.
2. **Graph (graph):** Adjust rt_window and mz_tolerance to define how nodes are linked geographically in the dataset.
3. **Thresholds (thresholds):** Tune correlation thresholds (isotope_corr, adduct_corr) to make chemical edge assignments more or less strict. Lowering these will yield denser graphs.


## Usage

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

### Interactive Notebook

Interactive Jupyter Notebook are provided to demonstrate the MassGAT pipeline ([ionshed](https://github.com/odisce/ionshed) segmentation, graph construction, inference, and component vizualization) on a test file.
  - Open the `massgat.ipynb` notebook, select the pixi `cpu` or `gpu` Python and R kernels, and follow the instructions in each cell to run the pipeline.
  - The results will be saved in the `results` subfolder of your project directory, and you can explore the output tables and visualizations directly in the notebook.

### Command Line Interface

Wrapped up scripts can be used to run the pipeline from the terminal.

1. **Trace Segmentation (R)**: Run the IonShed script to extract peaks and Region of Interest (ROI) signals into `.feather` format.
    ```bash
    Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output
    ```
2. **Graph Construction (Python)**: Build PyTorch Geometric graphs (`.pt`) from the extracted `.feather` tables.
    ```bash
    python scripts/build_graphs.py --input_dir  data/test/ionshed_output --output_dir data/test/graphs --ionization_mode positive
    ```
3. **Model Inference (Python)**: Pass the constructed graphs through the trained MassGAT model. This step evaluates probabilities, groups peaks into chemical components, and optionally annotates them. The `--with_top_annotations` flag can be used to obtain the top 5 annotations for each component suggested by the annotation heuristic. By default, only peaks with a probability above `--min_node_prob` threshold are kept (default: 0.5).
    ```bash
    python scripts/predict.py --folder data/test/graphs --data_type real --weights weights/model_positive.pth --model_type iterative --output_dir data/test/results --min_node_prob 0.5 --with_top_annotations
    ```

### Using containers

Here are example of commands to run the different part inside a container. Assuming the following paths:
  - input directory for mzML(s): 
  - input directory for config.yaml: 
  - output directory:   

1. **Trace Segmentation (R)**: `docker exec ...`
2. **Graph Construction (Python)**: `docker exec ...`
3. **Model Inference (Python)**: `docker exec ...`

TO COMPLETE

### Training MassGAT
To train MassGAT on a custom or synthetic dataset, use the [scripts/train.py](scripts/train.py) script. The script uses command-line arguments to configure hyperparameters and relies on `MLflow` to track experiments and metrics.

```bash
python scripts/train.py --use_synthetic --synthetic_path data/synthetic_dataset
```



## Output

Describe outputs, tables, figures, etc...

## Citation

To complete with publication reference