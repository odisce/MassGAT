options(repos = c(CRAN = "https://cloud.r-project.org/"))


## Install from a binary file
if (!requireNamespace("pak")) {
  install.packages("pak", ask = FALSE, type = "binary")
}
if (.Platform$OS.type == "windows") {
  download.file(
    "https://github.com/odisce/ionshed/releases/download/v1.0.1/win_ionshed_1.0.1.zip",
    "ionshed.zip"
  )
  install.packages(
    "ionshed.zip",
    type = "binary",
    dependencies = "TRUE"
  )
  unzip("ionshed.zip")
  pak::local_install_deps("ionshed", dependencies = TRUE)
  unlink("ionshed.zip")
  unlink("ionshed", recursive = TRUE)
} else {
  pak::pkg_install("odisce/ionshed@v1.0.1", dependencies = TRUE)
}

if (require("ionshed")) {message("ionshed installed")}