.locate_mbspls_model = function(gl, mbspls_id = NULL) {
  if (!inherits(gl, "GraphLearner")) {
    stop("'gl' must be a GraphLearner.", call. = FALSE)
  }
  if (is.null(gl$graph)) {
    stop("GraphLearner has no graph.", call. = FALSE)
  }

  # Identify the PipeOpMBsPLS node id
  if (is.null(mbspls_id)) {
    ids = names(gl$graph$pipeops)
    is_mbspls = vapply(gl$graph$pipeops, inherits, logical(1L), "PipeOpMBsPLS")
    cand = ids[is_mbspls]
    if (length(cand) < 1L) {
      stop("No PipeOpMBsPLS node found in the graph.", call. = FALSE)
    }
    if (length(cand) > 1L) {
      warning(
        sprintf(
          "Multiple PipeOpMBsPLS nodes found (%s). Using '%s'.",
          paste(cand, collapse = ", "),
          cand[[1L]]
        ),
        call. = FALSE
      )
    }
    mbspls_id = cand[[1L]]
  }

  fit_po = gl$graph$pipeops[[mbspls_id]]
  if (is.null(fit_po)) {
    stop(sprintf("PipeOp '%s' not found in graph.", mbspls_id), call. = FALSE)
  }

  mod = gl$model
  fit = mod[[mbspls_id]] %||% fit_po
  state = fit$state %||% fit

  if (!is.list(state)) {
    stop("Could not extract a valid MB-sPLS state from the GraphLearner.", call. = FALSE)
  }
  state
}
