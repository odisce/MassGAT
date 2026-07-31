# MassGAT

**MassGAT** is a graph-based collective learning framework for untargeted detection and annotation of LC-HRMS data. By modeling LC-MS files as hierarchical graphs where nodes represent ion traces and edges represent chemical relationships (isotopes, adducts, dimers), MassGAT performs simultaneous peak detection and grouping, leveraging analytical redundancy to rescue valid, low-intensity signals.



![MassGAT Pipeline](docs/massgat_overview_horizontal.png)
The pipeline runs in four stages — Trace Segmentation → Graph Construction → Model Inference → Annotation — matching the sections under [Usage](#usage) below.

---

## Table of Contents
- [Requirements](#requirements)
- [Local Installation](#local-installation)
  - [Set-up the environment](#set-up-the-environment)
  - [Install R dependencies](#install-r-dependencies)
- [Usage](#usage)
  - [Prediction](#prediction)
  - [Training](#training)
- [Performance & System Notes](#performance--system-notes)
- [Citation](#citation)



## Requirements
 
**MassGAT** can be run from a local environment or directly inside a container:
- **Local:** [Pixi >= 0.73](https://pixi.prefix.dev/latest/) and on Windows: [Rtools](https://cran.r-project.org/bin/windows/Rtools/).
- **Container:** [Docker](https://docs.docker.com/) (Windows, Ubuntu) or [Apptainer](https://apptainer.org/) (Ubuntu).



## Local Installation
 
First, clone this repository (e.g. using [git](https://git-scm.com/)):
```bash
git clone https://github.com/odisce/MassGAT.git
cd MassGAT
```


### Set-up the environment
 
| with Pixi | |
| :-------- | :--------- |
| to use cpu | `pixi shell -e cpu --locked` |
| to use gpu | `pixi shell -e gpu --locked` |
 
### Install R dependencies
 
```bash
pixi run install-ionshed
```


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

**Configuration**

MassGAT uses two configuration files:

1. **[`config.yaml`](config.yaml)** — Analytical parameters (peak picking, graph construction, correlation thresholds). Changes are automatically detected by the trace segmentation and graph construction scripts, which will reprocess files accordingly. Use the `--force` flag to force reprocessing without changing config.

2. **[`mass_differences.xlsx`](src/mass_differences.xlsx)** — Chemical mass differences (isotopes, adducts, dimers, and compatibility rules). Entry names should use appropriate prefixes (`adduct_`, `loss_`, `mod_`) to define directionality in annotations. When customized, you must regenerate graphs via Step 2 of [Prediction](#prediction). If you plan to retrain the model, also regenerate synthetic training data via Step 1 of [Training](#training).

### Prediction

1. **Trace Segmentation (R)**: Run the IonShed script to extract peaks and Region of Interest (ROI) signals into `.feather` format.
    ```bash
    Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output
    ```

2. **Graph Construction (Python)**: Build PyTorch Geometric graphs (`.pt`) from the extracted `.feather` tables.
    ```bash
    python scripts/build_graphs.py --input_dir  data/test/ionshed_output --output_dir data/test/graphs --ionization_mode positive
    ```


3. **Model Inference (Python)**: Pass the constructed graphs through the trained MassGAT model. This step evaluates probabilities, groups peaks into chemical components, and optionally annotates them.
    ```bash
    python scripts/predict.py --folder data/test/graphs --data_type real --weights weights/model_positive.pth --output_dir data/test/results --min_node_prob 0.5 --with_top_annotations
    ```

    **Key inference arguments:**

    | Argument | Description | Default |
    |---|---|---|
    | `--folder` | Path to the directory containing graph files (`.pt` format) | — |
    | `--weights` | Path to trained model weights (e.g., `weights/model_positive.pth` for positive-mode, `weights/model_negative.pth` for negative-mode) | — |
    | `--output_dir` | Directory where results will be saved | — |
    | `--min_node_prob` | Probability threshold above which a peak is kept | `0.5` |
    | `--with_top_annotations` | Also output top 5 candidate annotations per component | — |


> **Note**
> **For interactive visualization:** Once the results have been obtained by CLI, you can navigate through the results. To run the `visualization.ipynb` notebook in `docs/`. Select the (cpu or gpu) python Kernel, if modified, update the project name in `os.environ["PROJ_NAME"] = "test"` and run the notebook.



#### Outputs

- **Segmented ionshed features** in `.feather` format in subfolder `ionshed_output` of your project.
- **Graph representations** in `.pt` format in subfolder `graphs` of your project.
- **A final peak list** with detection probabilities and, if `--with_top_annotations` was used at Step 3, the top-ranked candidate annotations per component (main ion, isotopes, adducts, losses, dimers).

### Training

To retrain MassGAT on custom or synthetic data, first generate synthetic training data, then run the training script.

1. **Generate Synthetic Training Data**: generate synthetic components for each family type:
    ```bash
    python scripts/demo_generate_graphs.py --family <FAMILY_TYPE> --num <NUM_SAMPLES> --out synthetic_dataset/<FAMILY_TYPE> --mode positive --seed 42
    ```

    `--family <FAMILY_TYPE>` options:

    | Value | Description |
    |---|---|
    | `true` | True peaks component only |
    | `false` | False peaks component only (noise) |
    | `true_with_noise` | True peaks component with false peaks included |
    | `true_single` | Single isolated true peaks |
    | `false_single` | Single isolated false peaks |

    **Example**: generate 10,000 `true_with_noise` samples:
    ```bash
    python scripts/demo_generate_graphs.py --family true_with_noise --num 10000 --out synthetic_dataset/true_with_noise --mode positive --seed 42
    ```

2. **Train MassGAT**: Basic training with default hyperparameters:
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
    | `--seed` | Random seed for reproducibility | `42` |

    Trained weights are saved to the `weights/` directory, named after the ionization `--mode` used during data generation (e.g. `model_positive.pth`, `model_negative.pth`) — matching the files you'd point `predict.py` at in Step 3 of the [Prediction](#prediction) section above.

    **Tracking experiments with MLflow:**
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
    ```

    Then open `http://localhost:5001` in your browser to view training metrics and compare runs.

## Performance & System Notes

- Inference runs entirely on **CPU** — no GPU required.
- Typical end-to-end runtime per file (segmentation + graph construction + inference + annotation), on 5 CPU cores:
  - ~2.3 minutes for a Time-of-Flight (TOF) file
  - ~8.4 minutes for an Orbitrap (QE HF) file
- The model itself is lightweight (~98,000 trainable parameters), so retraining is feasible on CPU-only machines as well.

---

## Citation
