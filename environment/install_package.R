options(repos = c(CRAN = "https://cloud.r-project.org/"))


## Install from a binary file
if (.Platform$OS.type == "windows") {
  install.packages("pak", ask = FALSE, type = "binary")
  binary_path <- "local::C:/Users/SD265344/Desktop/ionshed_1.0.0_binary_win.zip"
  pak::pkg_install(binary_path, dependencies = TRUE)
} else {
  install.packages("pak", ask = FALSE, type = "source")
  binary_path <- "local::C:/Users/SD265344/Desktop/ionshed_1.0.0.zip"
  pak::pkg_install(binary_path, dependencies = TRUE)
}

if (require("ionshed")) {message("ionshed installed")}
