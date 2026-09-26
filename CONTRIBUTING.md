# Contributing to mlr3mbspls

Write R code and documentation in English and follow the mlr style guide. In
particular, use `=` for assignment, double-quoted strings, lower snake case for
functions and variables, `UpperCamelCase` for R6 classes, one statement per
line, and two-space indentation.

The repository pins the newest verified-compatible engine, `styler` 1.10.3,
and `styler.mlr` 0.1.0 at revision
`e950499afb6b28610400489f3ae6faacc6897358`. Install that toolchain with:

```r
install.packages("remotes")
remotes::install_version("styler", version = "1.10.3", upgrade = "never")
remotes::install_github(
  "mlr-org/styler.mlr@e950499afb6b28610400489f3ae6faacc6897358",
  upgrade = "never"
)
```

`styler` 1.11.0 is not currently compatible with that `styler.mlr` revision:
the mlr guide calls an internal transformer removed in 1.11.0. Keep the pinned
pair together until `styler.mlr` publishes a compatible revision.

Format the repository before committing:

```sh
Rscript tools/style.R
```

The formatter covers package R sources, tests, standalone validation and
example scripts, R Markdown and Quarto chunks, `inst/CITATION`, and R code
fences in Markdown files. It considers tracked files and untracked files that
no ignore rule matches, so ignored local files are never checked or rewritten.
`R/RcppExports.R` is generator output and is not formatted: commit it exactly
as `Rcpp::compileAttributes()` writes it. Check without modifying files with:

```sh
Rscript tools/style.R --check
```

The check also rejects residual `<-` assignments, including assignments that
the formatter cannot rewrite safely inside an unbraced control-flow body or a
function call. Add an explicit `{ ... }` block in those cases and use `=`
inside it.

The same check runs in pre-commit and continuous integration. After formatting,
run the package tests and `R CMD check` before opening a pull request.

## Development dependencies

`neuroCombat`, which ComBat site correction needs, is only available from
GitHub, and pak cannot resolve the Bioconductor remote declared in its
`DESCRIPTION`. Install the other dependencies with pak and the revision pinned
in continuous integration with remotes:

```r
install.packages(c("pak", "remotes"))
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
```

`Rscript build.R --deps` runs the same steps. Without options, `build.R`
regenerates the Rcpp wrappers, formats the repository, regenerates the
documentation, builds the source archive, checks that it contains no local
files, and installs it. Add `--test` to run the tests against the installed
archive and `--clean` to remove compiled objects and old archives first.
