options(
    repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"),
    download.file.method = "libcurl"
)

# Detect the active Conda environment path dynamically
conda_prefix <- Sys.getenv("CONDA_PREFIX")
if (conda_prefix == "") {
    stop("Error: Please activate your massgat_env Conda environment before running this script.")
}

# Match standard path structures across Windows/Unix
if (.Platform$OS.type == "windows") {
    lib_path <- file.path(conda_prefix, "Lib", "R", "library")
} else {
    lib_path <- file.path(conda_prefix, "lib", "R", "library")
}

if (!dir.exists(lib_path)) dir.create(lib_path, recursive = TRUE)
.libPaths(lib_path)

# 1. CRAN dependencies
cran_pkgs <- c("remotes", "arrow", "BiocManager", "fs", "htmlwidgets", "plotly", "igraph")
install.packages(cran_pkgs, lib = lib_path)

# 2. Bioconductor dependencies
bioc_pkgs <- c("mzR", "MSnbase", "QFeatures", "MsExperiment", "xcms")
BiocManager::install(bioc_pkgs, lib = lib_path, ask = FALSE, update = TRUE)

# 3. Ionshed Installation
repo_dir <- tempfile("ionshed_")
system(sprintf("git clone http://132.166.53.123:8081/tools/r-packages/ionshed.git %s", repo_dir))

if (dir.exists(repo_dir)) {
    remotes::install_local(repo_dir, lib = lib_path, dependencies = TRUE, upgrade = "never", force = TRUE)
    unlink(repo_dir, recursive = TRUE)
} else {
    stop("Error: Cannot reach the internal GitLab repository.")
}

cat("Success: All local laptop dependencies installed inside the environment.\n")