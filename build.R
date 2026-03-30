#!/usr/bin/env Rscript

# Build script for mlr3mbspls package

args <- commandArgs(trailingOnly = TRUE)

clean <- "--clean" %in% args
build_vignettes <- "--vignettes" %in% args
run_tests <- "--test" %in% args
install_deps <- "--deps" %in% args

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

if (!file.exists("DESCRIPTION")) {
  stop("This script should be run from the package root directory")
}

run_checked <- function(cmd, args = character()) {
  status <- system2(cmd, args = args)
  if (!identical(status, 0L)) {
    stop(sprintf(
      "Command failed (%s %s) with exit status %s",
      cmd,
      paste(args, collapse = " "),
      status
    ))
  }
  invisible(status)
}

if (install_deps) {
  cat("Installing dependencies...\n")
  if (!requireNamespace("pak", quietly = TRUE)) {
    install.packages("pak")
  }
  pak::pkg_install_deps(dependencies = TRUE)
}

cat("Compiling Rcpp code...\n")
if (!requireNamespace("Rcpp", quietly = TRUE)) {
  install.packages("Rcpp")
}
Rcpp::compileAttributes(".")

cat("Generating documentation with roxygen2...\n")
if (!requireNamespace("roxygen2", quietly = TRUE)) {
  install.packages("roxygen2")
}
roxygen2::roxygenize(".")

install_args <- c("CMD", "INSTALL")
if (clean) {
  install_args <- c(install_args, "--preclean")
}
if (build_vignettes) {
  install_args <- c(install_args, "--with-keep.source", "--install-tests")
}
install_args <- c(install_args, ".")

cat("Installing package...\n")
run_checked("R", install_args)

if (run_tests) {
  cat("Running tests...\n")
  if (!requireNamespace("testthat", quietly = TRUE)) {
    install.packages("testthat")
  }
  testthat::test_package("mlr3mbspls")
}

cat("Done!\n")
