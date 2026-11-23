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

1. **[`config.yaml`](config.yaml)** — Analytical parameters (peak picking, graph construction, correlation thresholds). Changes are automatically detected by the trace segmentation and graph construction scripts, which will reprocess files accordingly. Use the `--force` flag to force reprocessing without changing config. Set `MASSGAT_CONFIG=<file>` to use another parameter file. Large components are split to `graph.max_component_size` nodes, by cutting their weakest edges (`graph.component_split: cut`) or by merging nodes from the strongest edge under the size limit (`merge`, used for the annotation benchmark of the article); with `thresholds.isotope_label_rule: ratio`, a M+2 pair is labelled 34S, 37Cl (or 41K, kept as a 34S edge) from its intensity ratio.

2. **[`mass_differences.xlsx`](src/mass_differences.xlsx)** — Chemical mass differences (isotopes, adducts, dimers). Entry names should use appropriate prefixes (`adduct_`, `loss_`, `mod_`) to define directionality in annotations. The file contains the following sheets:
   - **isotopes**: Isotope mass shifts (mode-independent). Columns: `name`, `mass_difference`, `frequency`
   - **mods_positive / mods_negative**: Adducts/modifications for each ionization mode. Columns: `name`, `mass_difference`, `frequency`, `is_loss` (marks entries as losses)
   - **dimers_positive / dimers_negative**: Dimer mass shifts for each ionization mode. Columns: `name`, `mass_difference`, `frequency`
   - **compat_positive / compat_negative**: Compatibility for a chain of m/Z differences (e.g., how likely a H2O loss follows Na-H m/Z difference) for synthetic generation of components. Columns: `source_adduct`, `target_adduct`, `weight`.
   
   When customized, you must regenerate graphs via before training the model (see [Training](#training)).

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

    For negative-mode data, build the graphs with `--ionization_mode negative` and use the same option at inference:
    ```bash
    python scripts/predict.py --folder data/test/graphs --data_type real --weights weights/model_negative.pth --output_dir data/test/results --min_node_prob 0.5 --with_top_annotations --ionization_mode negative
    ```

    **Key inference arguments:**

    | Argument | Description | Default |
    |---|---|---|
    | `--folder` | Path to the directory containing graph files (`.pt` format) | — |
    | `--weights` | Path to trained model weights (e.g., `weights/model_positive.pth` for positive-mode, `weights/model_negative.pth` for negative-mode) | — |
    | `--output_dir` | Directory where results will be saved | — |
    | `--min_node_prob` | Probability threshold above which a peak is kept | `0.5` |
    | `--with_top_annotations` | Also output top 5 candidate annotations per component | — |
    | `--ionization_mode` | `positive` or `negative`, as used to build the graphs | `positive` |
    | `--all_node_probs` | Also write `<output>_all_probs.tsv` with the probability of every ion (the other outputs are unchanged) | off |


> **Note**
> **For interactive visualization:** Once the results have been obtained by CLI, you can navigate through the results. To run the `visualization.ipynb` notebook in `docs/`. Select the (cpu or gpu) python Kernel, if modified, update the project name in `os.environ["PROJ_NAME"] = "test"` and run the notebook. Retention times are in seconds (also in the notebook search box).

#### Outputs

- **Segmented ionshed features** in `.feather` format in subfolder `ionshed_output` of your project.
- **Graph representations** in `.pt` format in subfolder `graphs` of your project.
- **A final peak list** with detection probabilities and, if `--with_top_annotations` was used at Step 3, the top-ranked candidate annotations per component (main ion, isotopes, adducts, losses, dimers). Each candidate main ion is also tested as [M+Na]+, [M+NH4]+ and [M+K]+ ([M+Cl]- and [M+HCOO]- in the negative mode) and scored by the ions it explains; ions that remove an atom the molecule does not carry, carry a wrong charge state (isotope spacing of 0.5) or gain a neutral molecule are inconsistent, and implausible ions are left out of the peak group.

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
    | `--real_path` | Folder(s) of labelled real components, in the same per-component `.pt` format as the synthetic data | — |
    | `--no_synthetic` | Train on the `--real_path` folders only | — |

    **Training on labelled real data**, alone or together with the synthetic components:
    ```bash
    python scripts/train.py --no_synthetic --real_path data/labelled_components --num_epochs 15 --seed 42
    python scripts/train.py --synthetic_path synthetic_dataset --real_path data/labelled_components --num_epochs 15 --seed 42
    ```

    Trained weights are saved to the `weights/` directory, named after the ionization `--mode` used during data generation (e.g. `model_positive.pth`, `model_negative.pth`) — matching the files you'd point `predict.py` at in Step 3 of the [Prediction](#prediction) section above.

    **Settings of the released models.** The defaults above are those of the script, not those of the released weights. The weights in `weights/` were trained on 70% of the components (20% validation, 10% test) with Adam, a batch size of 64, hidden and embedding dimensions of 32 and 64, a weight decay of 1e-4 and the seed 42:

    | | `model_positive.pth` | `model_negative.pth` |
    |---|---|---|
    | Learning rate | 5e-5, multiplied by 0.9 when the validation loss has not improved for 5 epochs | 5e-5 (not reduced) |
    | Early stopping | patience 10, stopped at epoch 37 | patience 5, stopped at epoch 24 |
    | Weights released | last epoch | last epoch |

    The loss weights are `alpha = beta = gamma = 1` and `delta = 0.5` (Note S1 of the article). For the positive model:
    ```bash
    python scripts/train.py --use_synthetic --synthetic_path synthetic_dataset --real_path <labelled real components> --num_epochs 100 --early_stopping_patience 10 --batch_size 64 --learning_rate 5e-5 --hidden_dim 32 --embedding_dim 64 --seed 42
    ```
    (`--num_epochs` is only a maximum here). The real training data of the released models come from an in-house library that cannot be shared.
    **Tracking experiments with MLflow:**
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
    ```

    Then open `http://localhost:5001` in your browser to view training metrics and compare runs.

## Performance & System Notes

- Inference runs entirely on **CPU** — no GPU required.
- Typical end-to-end runtime per file (segmentation + graph construction + inference + annotation), on one CPU core (Intel Xeon Gold 6348):
  - ~1.8 minutes for a Time-of-Flight (TOF) file (TripleTOF 6600)
  - ~5.8 minutes for an Orbitrap (QE HF) file
- The model itself is lightweight (~98,000 trainable parameters), so retraining is feasible on CPU-only machines as well.

---

## Citation

If you use MassGAT in your research, please cite our preprint:

```bibtex
@article{Pinart2026MassGAT,
  author  = {Pinart, Paul-Henri and Damont, Annelaure and Dechaumet, Sylvain and Thevenot, Etienne},
  title   = {MassGAT: a graph-based collective learning approach for untargeted detection and annotation of LC-MS data},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.07.29.741473},
  url     = {https://doi.org/10.64898/2026.07.29.741473}
}
