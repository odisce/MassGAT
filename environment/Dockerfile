FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/conda/bin:$PATH" \
    R_LIBS_USER=/library/R \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git wget curl lsb-release \
    software-properties-common dirmngr gnupg2 ca-certificates \
    libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev \
    libharfbuzz-dev libfribidi-dev libglpk-dev libnetcdf-dev \
    libudunits2-dev libgdal-dev libproj-dev libgeos++-dev \
    libhdf5-dev zlib1g-dev libbz2-dev liblzma-dev libpcre2-dev \
    pandoc libgmp3-dev \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 \
    && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends r-base r-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel ninja \
    && python -m pip install --no-cache-dir \
    pandas openpyxl numpy scipy scikit-learn matplotlib seaborn \
    mlflow pyarrow tqdm numba pyyaml wandb torch-geometric \
    && python -m pip install --no-cache-dir --no-build-isolation \
    torch-scatter torch-sparse torch-cluster

RUN mkdir -p /library/R && chmod 777 /library/R \
    && echo 'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"), download.file.method = "libcurl", HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))' >> /usr/lib/R/etc/Rprofile.site \
    && R -e ".libPaths('/library/R'); pkgs <- c('remotes', 'arrow', 'BiocManager', 'fs', 'htmlwidgets', 'plotly', 'igraph'); install.packages(pkgs); if (any(!(pkgs %in% installed.packages()[,'Package']))) stop('ERREUR CRAN');" \
    && R -e ".libPaths('/library/R'); bioc_pkgs <- c('mzR', 'MSnbase', 'QFeatures', 'MsExperiment', 'xcms'); BiocManager::install(bioc_pkgs, ask = FALSE, update = TRUE); if (any(!(bioc_pkgs %in% installed.packages()[,'Package']))) stop('ERREUR Bioconductor');" \
    && git clone https://github.com/odisce/ionshed.git /opt/repo \
    && R -e ".libPaths('/library/R'); remotes::install_local('/opt/repo', dependencies = TRUE, upgrade = 'never', force = TRUE)"

CMD ["/bin/bash"]