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
  cat("  --clean      Remove compiled objects in src/ and old archives first\n")
  cat("  --vignettes  Build vignettes\n")
  cat("  --test       Run tests after building\n")
  cat("  --deps       Install dependencies, including suggested packages\n")
  cat("  --help, -h   Show this help message\n")
  quit(status = 0)
}

if (!file.exists("DESCRIPTION")) {
  stop("This script should be run from the package root directory")
}

metadata = read.dcf("DESCRIPTION")
package = metadata[1L, "Package"]
archive = sprintf("%s_%s.tar.gz", package, metadata[1L, "Version"])

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
  for (installer in c("pak", "remotes")) {
    if (!requireNamespace(installer, quietly = TRUE)) {
      install.packages(installer)
    }
  }
  # neuroCombat is only available from GitHub, and pak cannot resolve the
  # Bioconductor remote declared in its DESCRIPTION. Resolve everything else
  # with pak, then install the revision pinned in CI with its hard
  # dependencies only.
  pak::pkg_install(
    c("deps::.", "neuroCombat=?ignore"),
    dependencies = TRUE,
    upgrade = FALSE
  )
  remotes::install_github(
    "Jfortin1/neuroCombat_Rpackage@fbec46a61bc92bedb450b0e44addae4ce6afa934",
    dependencies = NA,
    upgrade = "never"
  )
}

if (clean) {
  cat("Removing compiled objects and old source archives...\n")
  unlink(c(
    list.files(
      "src",
      pattern = "[.](o|so|sl|dylib|a|dll)$|^symbols[.]rds$",
      full.names = TRUE,
      recursive = TRUE
    ),
    Sys.glob(sprintf("%s_*.tar.gz", package))
  ))
}

cat("Compiling Rcpp code...\n")
if (!requireNamespace("Rcpp", quietly = TRUE)) {
  install.packages("Rcpp")
}
Rcpp::compileAttributes(".")

# Format before generating documentation: styler also rewrites roxygen
# examples, and the Rd files must be generated from the formatted sources.
cat("Applying the pinned styler.mlr guide...\n")
run_checked("Rscript", "tools/style.R")

cat("Generating documentation with roxygen2...\n")
if (!requireNamespace("roxygen2", quietly = TRUE)) {
  install.packages("roxygen2")
}
roxygen2::roxygenize(".")

cat("Building source archive...\n")
build_args = c("CMD", "build")
if (!build_vignettes) {
  build_args = c(build_args, "--no-build-vignettes")
}
run_checked("R", c(build_args, "."))

# `R CMD build` reads only `.Rbuildignore`, not the git ignore rules, so local
# files in the working tree can reach the archive. Fail on top-level entries
# outside the package layout and on hidden files that R does not use.
entries = sub("/$", "", sub("^[^/]+/", "", utils::untar(archive, list = TRUE)))
entries = entries[nzchar(entries)]
allowed = c(
  "DESCRIPTION", "NAMESPACE", "NEWS.md", "README.md", "R", "build", "data",
  "inst", "man", "src", "tests", "vignettes", ".Rinstignore", ".aspell"
)
unexpected = c(
  setdiff(unique(sub("/.*$", "", entries)), allowed),
  grep("/[.]", setdiff(entries, "vignettes/.install_extras"), value = TRUE)
)
if (length(unexpected) > 0L) {
  stop(sprintf(
    "%s contains unexpected entries: %s. Remove them or add them to .Rbuildignore.",
    archive,
    paste(unexpected, collapse = ", ")
  ))
}

install_args = c("CMD", "INSTALL")
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
