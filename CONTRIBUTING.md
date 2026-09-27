# Contributing to mlr3mbspls

## Getting help and reporting problems

Use the [issue tracker](https://github.com/Coorsaa/mlr3mbspls/issues) for
questions, bug reports and feature requests.

- **Questions:** describe what you want to do and include the code you tried.
- **Bugs:** include a minimal reproducible example, the full error message,
  `packageVersion("mlr3mbspls")` and the output of `sessionInfo()`.
- **Feature requests:** describe the analysis or use case that the feature
  would support.

Report security-relevant problems privately to the maintainer at
<mail@stefancoors.de> instead of opening a public issue.

## Contributing changes

1. For larger changes, open an issue first to discuss the approach.
2. Fork the repository, create a branch from `main`, and keep each pull
   request focused on one change.
3. Add or update tests for every change in behaviour, update the roxygen
   documentation, and add an entry to `NEWS.md`.
4. Format the code and run the tests and `R CMD check` as described below.
5. Open a pull request that explains the change and its motivation.

## Support and governance

`mlr3mbspls` is maintained by Stefan Coors (maintainer) and Clara Sophie
Vetter. Issues and pull requests are answered on a best-effort basis.
Proposed changes are discussed in issues and pull requests; the maintainers
merge them into `main` after review and passing continuous integration.
Releases are tagged on GitHub and described in `NEWS.md`. The package follows
the conventions of the `mlr3` ecosystem.

## Code style

Write R code and documentation in English and follow the mlr style guide. In
particular, use `=` for assignment, double-quoted strings, lower snake case for
functions and variables, `UpperCamelCase` for R6 classes, one statement per
line, and two-space indentation.

The repository pins `styler` 1.10.3 and `styler.mlr` 0.1.0 at revision
`e950499afb6b28610400489f3ae6faacc6897358`. Install that toolchain with:

```r
install.packages(c("remotes", "roxygen2", "knitr"))
remotes::install_version("styler", version = "1.10.3", upgrade = "never")
remotes::install_github(
  "mlr-org/styler.mlr@e950499afb6b28610400489f3ae6faacc6897358",
  upgrade = "never"
)
```

`roxygen2` and `knitr` let `styler` format roxygen examples and R Markdown
chunks. `styler` 1.11.0 is not currently compatible with that `styler.mlr` revision:
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
archive, `--vignettes` to build the vignettes, and `--clean` to remove compiled
objects and old archives first.

## Tests and validation

Rebuild the native code after changing `src/`, and run the package tests and
`R CMD check` on the built archive. Changes that affect fitted results need a
regression test. MB-sPLS fits do not depend on seeds, so tests can compare
fitted weights and objectives directly.

When changing permutation code or complete-analysis inference, also run the
simulation scripts installed in `validation/`:

```sh
# Number of simulations, then number of permutations (both optional)
Rscript "$(Rscript -e 'cat(system.file("validation", "null_behavior.R", package = "mlr3mbspls"))')" 400 199

# Size set through environment variables (both optional)
MBSPLS_VALIDATION_N_SIM=200 MBSPLS_VALIDATION_N_PERM=99 \
  Rscript -e 'source(system.file("validation", "omnibus_permutation.R", package = "mlr3mbspls"))'
```

`null_behavior.R` reads its positional arguments only when the installed file
is run directly with `Rscript`, as above; sourcing it uses the defaults of 400
simulations (at least 100) and 199 permutations (at least 19).
`omnibus_permutation.R` reads `MBSPLS_VALIDATION_N_SIM` (default 200, at least
20) and `MBSPLS_VALIDATION_N_PERM` (default 99, at least 59). Both scripts stop
when a pre-specified bound fails. They check the implemented calculations in
small reference scenarios; they do not validate a study-specific design.
