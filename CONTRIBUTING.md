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

The formatter covers package R sources, generated Rcpp wrappers, tests,
standalone validation and example scripts, R Markdown and Quarto chunks,
`inst/CITATION`, and R code fences in Markdown files. Check without modifying
files with:

```sh
Rscript tools/style.R --check
```

The check also rejects residual `<-` assignments, including assignments that
the formatter cannot rewrite safely inside an unbraced control-flow body or a
function call. Add an explicit `{ ... }` block in those cases and use `=`
inside it.

The same check runs in pre-commit and continuous integration. After formatting,
run the package tests and `R CMD check` before opening a pull request.
