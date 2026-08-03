options(repos = c(CRAN = "https://cloud.r-project.org/"))


## Install pak (binary install only where supported, e.g. Windows/macOS)
if (!requireNamespace("pak", quietly = TRUE)) {
  pak_type <- if (.Platform$OS.type == "windows" || Sys.info()["sysname"] == "Darwin") {
    "binary"
  } else {
    "source"
  }
  install.packages("pak", ask = FALSE, type = pak_type)
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

deps <- pak::pkg_deps("odisce/ionshed@v1.0.1", dependencies = TRUE)
compile_deps <- deps$package[deps$needscompilation == TRUE]
print(compile_deps)
