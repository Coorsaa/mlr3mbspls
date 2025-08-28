#!/usr/bin/env Rscript

# Build script for mlr3mbspls package

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Default options
clean <- FALSE
build_vignettes <- FALSE
run_tests <- FALSE
install_deps <- FALSE

# Process arguments
if ("--clean" %in% args) clean <- TRUE
if ("--vignettes" %in% args) build_vignettes <- TRUE
if ("--test" %in% args) run_tests <- TRUE
if ("--deps" %in% args) install_deps <- TRUE
if ("--help" %in% args || "-h" %in% args) {
  cat("Usage: Rscript build.R [options]\n")
  cat("Options:\n")
  cat("  --clean      Clean and rebuild the package\n")
  cat("  --vignettes  Build vignettes\n")
  cat("  --test       Run tests after building\n")
  cat("  --deps       Install dependencies\n")
  cat("  --help, -h   Show this help message\n")
  quit(status = 0)
}

# Working directory should be the package root
if (!file.exists("DESCRIPTION")) {
  stop("This script should be run from the package root directory")
}

# Install dependencies if needed
if (install_deps) {
  cat("Installing dependencies...\n")
  if (!requireNamespace("pak", quietly = TRUE)) {
    install.packages("pak")
  }
  pak::pkg_install_deps(dependencies = TRUE)
}

# Generate documentation
cat("Generating documentation with roxygen2...\n")
if (!requireNamespace("roxygen2", quietly = TRUE)) {
  install.packages("roxygen2")
}
roxygen2::roxygenize()

# Compile Rcpp code
cat("Compiling Rcpp code...\n")
if (!requireNamespace("Rcpp", quietly = TRUE)) {
  install.packages("Rcpp")
}
Rcpp::compileAttributes()

# Install command
install_cmd <- "R CMD INSTALL"
if (clean) install_cmd <- paste(install_cmd, "--preclean")
if (build_vignettes) install_cmd <- paste(install_cmd, "--with-keep.source", "--install-tests")

# Install the package
cat("Installing package...\n")
system(install_cmd)

# Run tests if requested
if (run_tests) {
  cat("Running tests...\n")
  if (!requireNamespace("testthat", quietly = TRUE)) {
    install.packages("testthat")
  }
  testthat::test_package("mlr3mbspls")
}

cat("Done!\n")
