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

**Automatic Configuration Change Detection:**
The trace segmentation and graph construction steps automatically detect when relevant [`config.yaml`](config.yaml) parameters change and reprocess files accordingly. To force reprocessing without changing config, use the `--force` flag on either script. See [Configuration Management & Change Detection](#configuration-management--change-detection) below for details on which parameters are tracked.

### Prediction


1. **Trace Segmentation (R)**: Run the IonShed script to extract peaks and Region of Interest (ROI) signals into `.feather` format.
    ```bash
    Rscript scripts/ionshed_batch_processing.R data/test/mzml data/test/ionshed_output
    ```


2. **Graph Construction (Python)**: Build PyTorch Geometric graphs (`.pt`) from the extracted `.feather` tables.
    ```bash
    python scripts/build_graphs.py --input_dir  data/test/ionshed_output --output_dir data/test/graphs --ionization_mode positive
    ```


3. **Model Inference (Python)**:
TODO : tableau
Pass the constructed graphs through the trained MassGAT model. This step evaluates probabilities, groups peaks into chemical components, and optionally annotates them.
    - `--min_node_prob`: probability threshold above which a peak is kept (default: `0.5`).
    - `--with_top_annotations`: also outputs the top 5 candidate annotations per component, as ranked by the annotation heuristic.
    - `--weights`: path to the trained model weights (e.g. `weights/model_positive.pth` for positive-mode data, `weights/model_negative.pth` for negative-mode data).
    - `--output_dir`: Results are written to the directory you specified (e.g. `data/test/results`)

    ```bash
    python scripts/predict.py --folder data/test/graphs --data_type real --weights weights/model_positive.pth --model_type iterative --output_dir data/test/results --min_node_prob 0.5 --with_top_annotations
    ```


> **Note**
> **For interactive visualization:** Once the results have been obtained by CLI, you can navigate through the results. To run the `visualization.ipynb` notebook in `docs/`, install `ipywidgets`, `networkx`, and `jupyter`.



#### Outputs

TODO : rajouter ionshed + graphes
- **A final peak list** with detection probabilities and, if `--with_top_annotations` was used at Step 3, the top-ranked candidate annotations per component (main ion, isotopes, adducts, losses, dimers).
- **Pseudo-spectra visualizations** per-component views of validated ion species (m/Z vs. intensity) alongside their elution profiles.

### Training

To retrain MassGAT on custom or synthetic data, first generate synthetic training data, then run the training script.

1. **Generate Synthetic Training Data**: generate synthetic components for each family type:
    ```bash
    python scripts/demo_generate_graphs.py --family <FAMILY_TYPE> --num <NUM_SAMPLES> --out synthetic_dataset/<FAMILY_TYPE> --mode positive --seed 42
    ```

    `--family <FAMILY_TYPE>` options:

    | Value | Description |
    |---|---|
    | `true_with_noise` | True peaks with chemical relationships (isotopes, adducts, dimers) + noise peaks |
    | `true` | True peaks only, with chemical relationships |
    | `false` | False peaks only (noise) |
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
    | `--augmentation` | Data augmentation: `none`, `node_drop`, `edge_drop`, `feature_noise`, `feature_mask`, `mixed` | `none` |
    | `--seed` | Random seed for reproducibility | `42` |

    Trained weights are saved to the `weights/` directory, named after the ionization `--mode` used during data generation (e.g. `model_positive.pth`, `model_negative.pth`) — matching the files you'd point `predict.py` at in [Step 3: Model Inference](#3-model-inference-python).

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