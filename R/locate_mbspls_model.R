.find_pipeop_id_by_class = function(graph, class_name, id = NULL, where = "graph") {
  if (is.null(graph) || is.null(graph$pipeops)) {
    stop(sprintf("%s has no pipeops.", where), call. = FALSE)
  }

  ids = names(graph$pipeops)
  if (!length(ids)) {
    stop(sprintf("%s has no pipeops.", where), call. = FALSE)
  }

  if (!is.null(id)) {
    po = graph$pipeops[[id]]
    if (is.null(po)) {
      stop(sprintf("PipeOp '%s' not found in %s.", id, where), call. = FALSE)
    }
    if (!inherits(po, class_name)) {
      stop(sprintf("PipeOp '%s' in %s is not of class '%s'.", id, where, class_name), call. = FALSE)
    }
    return(id)
  }

  is_match = vapply(graph$pipeops, inherits, logical(1L), class_name)
  cand = ids[is_match]
  if (length(cand) < 1L) {
    stop(sprintf("No %s node found in %s.", class_name, where), call. = FALSE)
  }
  if (length(cand) > 1L) {
    warning(
      sprintf("Multiple %s nodes found in %s (%s). Using '%s'.", class_name, where, paste(cand, collapse = ", "), cand[[1L]]),
      call. = FALSE
    )
  }
  cand[[1L]]
}

.mbspls_pipeop_id = function(graph, mbspls_id = NULL, where = "graph") {
  .find_pipeop_id_by_class(graph = graph, class_name = "PipeOpMBsPLS", id = mbspls_id, where = where)
}

.locate_mbspls_model = function(gl, mbspls_id = NULL) {
  if (!inherits(gl, "GraphLearner")) {
    stop("'gl' must be a GraphLearner.", call. = FALSE)
  }
  if (is.null(gl$graph)) {
    stop("GraphLearner has no graph.", call. = FALSE)
  }

  mbspls_id = .mbspls_pipeop_id(gl$graph, mbspls_id = mbspls_id, where = "GraphLearner$graph")

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
