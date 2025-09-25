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
