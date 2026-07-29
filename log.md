## TO DELETE

### 28 JULY 2026
  - removed Table of Contents](#table-of-contents) in Table of content
  - rewritten the Table of content to match sections
  - added interactive links to file (e.g.: [`config.yaml`](config.yaml))
  - missing ipywidgets dependency
  - [TODO] Convert all french message to english (e.g.: Mode d'ionisation détecté) but there are others
  - [TODO] Clean unnecessary files, some script are neither described or used (e.g.: xcms_alignement.R, ...) remove them or describe them in the readme.
  - trailing "," in classifiers section
  - some heavy dependencies which seems to overlap (torch, pytorch, mlflow)
  - [FAILING] Can't reproduce jupyter dependencies due to rpy2 which constrains all the others and fails
  - Comments with instructions to install inside the .toml removed, it should be stated in the readme
  - [IMPROVEMENT] There are scripts duplication between files in ./scripts and the jupyter notebook.
                  To avoid mismatch or breaking change the common script should be in a single file which define
                  the common functions/wrappers which are then called in both medias.
  - la section training fait partie des usages -> déplacé dedands
  - [IMPROVEMENT]  some dependencies are specific for vizu and jupyter, not needed for the CLI commands and should be separated (not included in the computational scripts)
  - [IMPROVEMENT]  Adding a very small .mzml file to run unitary tests
  - [IMPROVEMENT]  Adding unitary tests
  - recompilation en partant avec un .toml vide
    - problem when adding mlflow, dependencies conflict
    - not sure why mlflow is needed for predict.py ?
  - [BUG] Warning during inference to solve:  UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True

### 27 JULY 2026
  - Description to create a project folder for user is duplicated in Interactive tutorial and Command line interface.
    - written on time at the beginning
    - also suggest user to use the provided `test` project
    - Jupyter module not needed, already installed by pixi
    - put notebook in doc/ and change to a link:  Open the `massgat.ipynb`
      -  changed to [doc/massgat.ipynb](doc/massgat.ipynb)
    - xlsx file starting with ~ are temporary filew when opened on a computer, they should,'t be included
      - removed all ^~* files
      - added gitignore on them
    - Trying jupyter notebook
      - Added dependencies `ipykernel` and `notebook` needed to run jupyter notebook
      - missing rpy2 dependency in .toml
      


### 24 JULY 2026

  - missing R dependencies (arrow) not from ionshed and not mentionned, added to the .toml
  - renamed SpMat_DescendingPeakPicking to ionshed (changed in last ionshed version)
  - **1. Trace Segmentation (R)**: CPU OK GPU OK
  - **2. Graph Construction (Python)**: CPU OK GPU OK
  - **3. Model Inference (Python)**: CPU OK GPU OK
  - [WARNING] I saw a skipped functionality in the scripts which is nice to have. The problem
  is when we changed the parameters in the `config.yaml`, it's not detected by the scripts and
  they will reuse previous data even if the parameters changed. This behavior must be solved or
  documented. Also if the steps 1. or 2. changed but 3. already exists, then it won't run again.
  - Added outputs to `.gitignore`
  - **python scripts/train.py --use_synthetic --synthetic_path data/synthetic_dataset** not working

### 22 JULY 2026

  - Added pixi support in pyproject.toml (advantages: it install python, R and everything needed for cpu and/or gpu and on e)
  - Added prerequisites for both pixi and miniforge
  - Removed part of miniforge setup
    - `conda init cmd.exe`: It's part on how to install miniforge and should then the official doc should be refered as above. Also this command doesn't give access to conda on windows, the command is `conda init cmd.exe`, if not, then `conda` is not available in the powershell (default terminal)
      Open the Miniforge/Miniconda terminal to run the setup. If you wish to use a standard terminal on Windows, run :
  - Added authors in pyprojects.toml
  - Added CeCILL License
  - Added r-base in pyprojects.toml
  - Changed version for pytorch
  - Added cuda version based on Dockerfile
  - Added support for pixi to get a cpu or gpu environment
  - Can test cpu/gpu install with `pixi run -e cpu test-gpu` and `pixi run -e gpu test-gpu`
  - Conform module versions between readme and .toml:
    - ex: no: `conda create --name massgat_env python>=3.10` and `requires-python = ">=3.8,<3.11"`
          yes: `conda create --name massgat_env python>=3.8` and `requires-python = ">=3.8"`
    - ex: no: `"numpy>=1.22.0,<2.0.0"`
          yes: `"numpy>=1.22.0"`
  - Script doesn't exists at this location: `src/massgat/r_scripts/install_packages.R`
    - move script from `./install_packages.R` to `./environment/install_packages.R` (since it's used to install the dependencies I think it's better to put it there)
    - changed script to a simpler form
      - Comments
        - checks using CONDA_PREFIX won't always work
        - library paths should be taken care off by the dependencies solvers (conda and or pak and or pixi))
        - dependencies should be declared and installed in ionshed (using dependencies = TRUE should be sufficient)
        - cloning is not necessary, remotes, renv or pak can handle gitlab and github packages directly
      - Changed
        - changed `install_packages.R`
        - added pixi tasks: `install-ionshed = "Rscript environment/install_package.R"`







  - [ ] Test massgat
  - [ ] Test conda/miniforge
    - Following commands are in conflict with what's inside the .toml:
      `pip install torch==2.1.0 torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)`
      `pip install torch-scatter torch-sparse torch-cluster -f [https://data.pyg.org/whl/torch-2.1.0+cu118.html](https://data.pyg.org/whl/torch-2.1.0+cu118.html)`
    - Added theses dependencies in the .toml, they should be done with `pip install -e .` directly from the .toml

  - [ ] Remove comments in .toml
