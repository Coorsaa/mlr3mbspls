#!/usr/bin/env Rscript

# Build script for mlr3mbspls package

args = commandArgs(trailingOnly = TRUE)

clean = "--clean" %in% args
build_vignettes = "--vignettes" %in% args
run_tests = "--test" %in% args
install_deps = "--deps" %in% args

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

run_checked = function(cmd, args = character()) {
  status = system2(cmd, args = args)
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

cat("Applying the pinned styler.mlr guide...\n")
run_checked("Rscript", "tools/style.R")

cat("Building source archive...\n")
build_args = c("CMD", "build")
if (!build_vignettes) {
  build_args = c(build_args, "--no-build-vignettes")
}
run_checked("R", c(build_args, "."))
metadata = read.dcf("DESCRIPTION")
archive = sprintf("%s_%s.tar.gz", metadata[1L, "Package"], metadata[1L, "Version"])

install_args = c("CMD", "INSTALL")
if (clean) {
  install_args = c(install_args, "--preclean")
}
if (run_tests) {
  install_args = c(install_args, "--install-tests")
}
install_args = c(install_args, shQuote(archive))

cat("Installing package...\n")
run_checked("R", install_args)

if (run_tests) {
  cat("Running tests...\n")
  if (!requireNamespace("testthat", quietly = TRUE)) {
    install.packages("testthat")
  }
  # Documentation generation loads the source namespace; test the installed
  # archive in a fresh process so it cannot accidentally use that namespace.
  run_checked("Rscript", c("-e", shQuote(
    'testthat::test_package("mlr3mbspls", stop_on_failure = TRUE)'
  )))
}

cat("Done!\n")
