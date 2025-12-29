#' @title Internal utilities for mlr3mbspls
#'
#' @description
#' These are internal utility functions for mlr3mbspls.
#'
#' @keywords internal
#' @name mlr3mbspls_utils
NULL

# Define the null-coalescing operator
`%||%` = function(x, y) if (is.null(x) || length(x) == 0L) y else x

# Internal RNG helper: run a function with a temporary seed and restore the
# previous RNG state.
#
# This avoids polluting the global RNG state when PipeOps are used inside
# resampling / tuning loops.
with_seed_local = function(seed, fn) {
  if (is.null(seed) || length(seed) != 1L || !is.finite(seed)) {
    return(fn())
  }

  seed = as.integer(seed)
  if (!is.finite(seed) || seed <= 0L) {
    return(fn())
  }

  had_seed = exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed = if (had_seed) get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL

  on.exit({
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(seed)
  fn()
}
