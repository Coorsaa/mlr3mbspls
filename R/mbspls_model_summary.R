#' Summarize a fitted multi-block model
#'
#' @title Tidy summaries for MB-sPLS, MB-sPLS-XY, and MB-sPCA fits
#'
#' @description
#' `mbspls_model_summary()` extracts compact, tidy summaries from fitted
#' `PipeOpMBsPLS`, `PipeOpMBsPLSXY`, `PipeOpMBsPCA`, or `GraphLearner` objects.
#'
#' The helper is intended for reporting, interpretation, and handoff to domain
#' experts. It turns the internal model state into tables that are easy to review
#' in notebooks, manuscripts, clinical reports, or downstream dashboards.
#'
#' The returned list always includes:
#'
#' * `overview`: high-level model metadata,
#' * `components`: one row per latent component,
#' * `blocks`: one row per component-block pair,
#' * `weights`: one row per feature (optional).
#'
#' For MB-sPLS graphs with a bootstrap-selection node, the list also includes a
#' `stability` table if `include_stability = TRUE`.
#'
#' @param x A fitted [mlr3pipelines::GraphLearner], [PipeOpMBsPLS()],
#'   [PipeOpMBsPLSXY()], or [PipeOpMBsPCA()].
#' @param id Optional pipeop id to use when `x` is a `GraphLearner` containing
#'   multiple multi-block nodes.
#' @param include_weights Logical; include the feature-level weights table.
#' @param include_stability Logical; include bootstrap stability summaries when
#'   they are available.
#'
#' @return A named list of `data.table` objects.
#'
#' @examples
#' task = mlr3::tsk("mbspls_synthetic_blocks")
#' gl = mbspls_graph_learner(
#'   task = task,
#'   learner = mlr3::lrn("clust.kmeans", centers = 2L),
#'   ncomp = 1L,
#'   bootstrap = FALSE,
#'   bootstrap_selection = FALSE,
#'   B = 1L,
#'   val_test = "none"
#' )
#' # gl$train(task)
#' # mbspls_model_summary(gl)
#'
#' @export
mbspls_model_summary = function(x, id = NULL, include_weights = TRUE, include_stability = TRUE) {
  checkmate::assert_flag(include_weights, .var.name = "include_weights")
  checkmate::assert_flag(include_stability, .var.name = "include_stability")

  resolved = mb_resolve_summary_target(x = x, id = id)
  state = resolved$state
  if (!is.list(state) || !length(state)) {
    stop("`mbspls_model_summary()` requires a fitted object with a non-empty state.", call. = FALSE)
  }

  out = switch(
    resolved$model_kind,
    mbspls = mb_summary_from_mbspls(
      state = state,
      pipeop_id = resolved$pipeop_id,
      source = resolved$source,
      selection_state = resolved$selection_state,
      include_weights = include_weights,
      include_stability = include_stability
    ),
    mbspca = mb_summary_from_mbspca(
      state = state,
      pipeop_id = resolved$pipeop_id,
      source = resolved$source,
      include_weights = include_weights
    ),
    mbsplsxy = mb_summary_from_mbsplsxy(
      state = state,
      pipeop_id = resolved$pipeop_id,
      source = resolved$source,
      include_weights = include_weights
    ),
    stop(sprintf("Unsupported model kind '%s'.", resolved$model_kind), call. = FALSE)
  )

  out
}


mb_resolve_summary_target = function(x, id = NULL) {
  if (inherits(x, "GraphLearner")) {
    if (is.null(x$model)) {
      stop("GraphLearner appears to be untrained (model is NULL).", call. = FALSE)
    }
    graph = x$graph
    if (is.null(graph) || is.null(graph$pipeops)) {
      stop("GraphLearner has no graph or pipeops.", call. = FALSE)
    }

    kinds = list(
      mbspls = "PipeOpMBsPLS",
      mbspca = "PipeOpMBsPCA",
      mbsplsxy = "PipeOpMBsPLSXY"
    )

    if (!is.null(id)) {
      po_template = graph$pipeops[[id]]
      if (is.null(po_template)) {
        stop(sprintf("PipeOp '%s' not found in GraphLearner$graph.", id), call. = FALSE)
      }
      model_kind = names(kinds)[vapply(kinds, function(cls) inherits(po_template, cls), logical(1))]
      if (!length(model_kind)) {
        stop(sprintf("PipeOp '%s' is not an MB-sPLS/MB-sPLS-XY/MB-sPCA node.", id), call. = FALSE)
      }
      model_kind = model_kind[[1L]]
    } else {
      model_kind = NULL
      pipeop_id = NULL
      for (nm in names(kinds)) {
        hit = tryCatch(
          .find_pipeop_id_by_class(graph, class_name = kinds[[nm]], where = "GraphLearner$graph"),
          error = function(e) NULL
        )
        if (!is.null(hit)) {
          model_kind = nm
          pipeop_id = hit
          break
        }
      }
      if (is.null(model_kind)) {
        stop("No MB-sPLS, MB-sPLS-XY, or MB-sPCA node found in the GraphLearner.", call. = FALSE)
      }
      id = pipeop_id
    }

    po_template = graph$pipeops[[id]]
    po_fit = x$model[[id]] %||% po_template
    state = po_fit$state %||% po_fit

    selection_state = NULL
    if (identical(model_kind, "mbspls")) {
      sel_id = tryCatch(
        .find_pipeop_id_by_class(graph, class_name = "PipeOpMBsPLSBootstrapSelect", where = "GraphLearner$graph"),
        error = function(e) NULL
      )
      if (!is.null(sel_id)) {
        sel_fit = x$model[[sel_id]] %||% graph$pipeops[[sel_id]]
        selection_state = sel_fit$state %||% sel_fit
      }
    }

    return(list(
      source = "graphlearner",
      model_kind = model_kind,
      pipeop_id = id,
      state = state,
      selection_state = selection_state
    ))
  }

  if (inherits(x, "PipeOpMBsPLS")) {
    return(list(source = "pipeop", model_kind = "mbspls", pipeop_id = x$id, state = x$state, selection_state = NULL))
  }
  if (inherits(x, "PipeOpMBsPCA")) {
    return(list(source = "pipeop", model_kind = "mbspca", pipeop_id = x$id, state = x$state, selection_state = NULL))
  }
  if (inherits(x, "PipeOpMBsPLSXY")) {
    return(list(source = "pipeop", model_kind = "mbsplsxy", pipeop_id = x$id, state = x$state, selection_state = NULL))
  }

  stop("`mbspls_model_summary()` supports GraphLearner, PipeOpMBsPLS, PipeOpMBsPLSXY, and PipeOpMBsPCA objects.", call. = FALSE)
}


mb_component_names = function(weights, prefix) {
  nm = names(weights)
  if (is.null(nm) || any(!nzchar(nm))) {
    nm = sprintf(prefix, seq_along(weights))
  }
  nm
}


mb_feature_weights_table = function(weights, loadings = NULL, blocks, component_names) {
  rows = list()
  for (k in seq_along(weights)) {
    comp = component_names[[k]]
    Wk = weights[[k]]
    Pk = if (is.null(loadings)) NULL else loadings[[k]]
    for (bn in names(blocks)) {
      feats = blocks[[bn]]
      wb = Wk[[bn]] %||% stats::setNames(rep(0, length(feats)), feats)
      if (is.null(names(wb))) {
        names(wb) = feats
      }
      wb = wb[feats]
      wb[is.na(wb)] = 0
      pb = NULL
      if (!is.null(Pk)) {
        pb = Pk[[bn]]
        if (!is.null(pb)) {
          if (is.null(names(pb))) {
            names(pb) = feats
          }
          pb = pb[feats]
        }
      }
      rows[[length(rows) + 1L]] = data.table::data.table(
        component = comp,
        block = bn,
        feature = feats,
        weight = as.numeric(wb),
        loading = if (is.null(pb)) NA_real_ else as.numeric(pb),
        selected = abs(as.numeric(wb)) > 0
      )
    }
  }
  data.table::rbindlist(rows, use.names = TRUE, fill = TRUE)
}


mb_summary_from_mbspls = function(state, pipeop_id, source, selection_state = NULL, include_weights = TRUE, include_stability = TRUE) {
  blocks = mb_normalize_blocks(state$blocks, .var.name = "state$blocks")
  weights = state$weights
  loadings = state$loadings
  component_names = mb_component_names(weights, "LC_%02d")
  ev_comp = as.numeric(state$ev_comp %||% rep(NA_real_, length(component_names)))
  obj_vec = as.numeric(state$obj_vec %||% rep(NA_real_, length(component_names)))
  p_values = as.numeric(state$p_values %||% rep(NA_real_, length(component_names)))
  ev_block = state$ev_block %||% matrix(NA_real_, nrow = length(component_names), ncol = length(blocks), dimnames = list(component_names, names(blocks)))

  overview = data.table::data.table(
    source = source,
    model = "mbspls",
    pipeop_id = pipeop_id %||% NA_character_,
    n_components = length(component_names),
    n_blocks = length(blocks),
    performance_metric = state$performance_metric %||% NA_character_,
    correlation_method = state$correlation_method %||% NA_character_,
    run_id = state$run_id %||% NA_character_
  )

  components = data.table::data.table(
    component = component_names,
    objective = obj_vec,
    p_value = p_values,
    ev_comp = ev_comp
  )

  block_rows = list()
  for (k in seq_along(component_names)) {
    for (bn in names(blocks)) {
      wb = weights[[k]][[bn]] %||% numeric(0)
      if (is.null(names(wb))) {
        names(wb) = blocks[[bn]][seq_along(wb)]
      }
      wb = wb[blocks[[bn]]]
      wb[is.na(wb)] = 0
      block_rows[[length(block_rows) + 1L]] = data.table::data.table(
        component = component_names[[k]],
        block = bn,
        n_features = length(blocks[[bn]]),
        n_selected = sum(abs(as.numeric(wb)) > 0),
        ev_block = ev_block[k, bn, drop = TRUE]
      )
    }
  }
  blocks_dt = data.table::rbindlist(block_rows, use.names = TRUE, fill = TRUE)

  out = list(
    overview = overview,
    components = components,
    blocks = blocks_dt
  )

  if (isTRUE(include_weights)) {
    out$weights = mb_feature_weights_table(weights = weights, loadings = loadings, blocks = blocks, component_names = component_names)
  }

  if (isTRUE(include_stability) && !is.null(selection_state)) {
    ci_dt = if (!is.null(selection_state$weights_ci) && nrow(as.data.frame(selection_state$weights_ci))) {
      data.table::as.data.table(selection_state$weights_ci)
    } else {
      data.table::data.table(component = character(), block = character(), feature = character(), boot_mean = numeric(), boot_sd = numeric(), ci_lower = numeric(), ci_upper = numeric())
    }
    freq_dt = if (!is.null(selection_state$weights_selectfreq) && nrow(as.data.frame(selection_state$weights_selectfreq))) {
      data.table::as.data.table(selection_state$weights_selectfreq)
    } else {
      data.table::data.table(component = character(), block = character(), feature = character(), freq = numeric())
    }
    stable_dt = data.table::data.table(component = character(), block = character(), feature = character(), stable_weight = numeric())

    stable_w = selection_state$weights_stable
    if (is.list(stable_w) && length(stable_w)) {
      stable_comp_names = mb_component_names(stable_w, "LC_%02d")
      stable_rows = list()
      for (k in seq_along(stable_w)) {
        for (bn in names(stable_w[[k]])) {
          wb = stable_w[[k]][[bn]]
          if (is.null(wb)) next
          feats = names(wb) %||% blocks[[bn]][seq_along(wb)]
          stable_rows[[length(stable_rows) + 1L]] = data.table::data.table(
            component = stable_comp_names[[k]],
            block = bn,
            feature = feats,
            stable_weight = as.numeric(wb)
          )
        }
      }
      if (length(stable_rows)) {
        stable_dt = data.table::rbindlist(stable_rows, use.names = TRUE, fill = TRUE)
      }
    }

    key = c("component", "block", "feature")
    stability = if (nrow(ci_dt) || nrow(freq_dt) || nrow(stable_dt)) {
      merged = merge(ci_dt, freq_dt, by = key, all = TRUE)
      merged = merge(merged, stable_dt, by = key, all = TRUE)
      merged[, stable_selected := !is.na(stable_weight) & abs(stable_weight) > 0]
      merged[, selection_method := selection_state$selection_method %||% NA_character_]
      merged[, frequency_threshold := selection_state$frequency_threshold %||% NA_real_]
      merged[, alignment_method := selection_state$alignment_method %||% NA_character_]
      merged[]
    } else {
      data.table::data.table(
        component = character(), block = character(), feature = character(),
        boot_mean = numeric(), boot_sd = numeric(), ci_lower = numeric(), ci_upper = numeric(),
        freq = numeric(), stable_weight = numeric(), stable_selected = logical(),
        selection_method = character(), frequency_threshold = numeric(), alignment_method = character()
      )
    }
    out$stability = stability
  }

  out
}


mb_summary_from_mbspca = function(state, pipeop_id, source, include_weights = TRUE) {
  blocks = mb_normalize_blocks(state$blocks, .var.name = "state$blocks")
  weights = state$weights
  loadings = state$loadings
  component_names = mb_component_names(weights, "PC%d")
  ev_comp = as.numeric(state$ev_comp %||% rep(NA_real_, length(component_names)))
  ev_block = state$ev_block %||% matrix(NA_real_, nrow = length(component_names), ncol = length(blocks), dimnames = list(component_names, names(blocks)))

  overview = data.table::data.table(
    source = source,
    model = "mbspca",
    pipeop_id = pipeop_id %||% NA_character_,
    n_components = length(component_names),
    n_blocks = length(blocks),
    performance_metric = NA_character_,
    correlation_method = NA_character_,
    run_id = state$run_id %||% NA_character_
  )

  components = data.table::data.table(
    component = component_names,
    ev_comp = ev_comp
  )

  block_rows = list()
  for (k in seq_along(component_names)) {
    for (bn in names(blocks)) {
      wb = weights[[k]][[bn]] %||% numeric(0)
      if (is.null(names(wb))) {
        names(wb) = blocks[[bn]][seq_along(wb)]
      }
      wb = wb[blocks[[bn]]]
      wb[is.na(wb)] = 0
      block_rows[[length(block_rows) + 1L]] = data.table::data.table(
        component = component_names[[k]],
        block = bn,
        n_features = length(blocks[[bn]]),
        n_selected = sum(abs(as.numeric(wb)) > 0),
        ev_block = ev_block[k, bn, drop = TRUE]
      )
    }
  }
  blocks_dt = data.table::rbindlist(block_rows, use.names = TRUE, fill = TRUE)

  out = list(
    overview = overview,
    components = components,
    blocks = blocks_dt
  )
  if (isTRUE(include_weights)) {
    out$weights = mb_feature_weights_table(weights = weights, loadings = loadings, blocks = blocks, component_names = component_names)
  }
  out
}


mb_summary_from_mbsplsxy = function(state, pipeop_id, source, include_weights = TRUE) {
  blocks = mb_normalize_blocks(state$blocks_x, .var.name = "state$blocks_x")
  weights_x = state$weights_x
  loadings_x = state$loadings_x
  weights_y = state$weights_y
  loadings_y = state$loadings_y
  component_names = mb_component_names(weights_x, "LC_%02d")
  target_cols = state$target_columns
  if (is.null(target_cols) && is.list(weights_y) && length(weights_y)) {
    target_cols = names(weights_y[[1L]])
  }

  overview = data.table::data.table(
    source = source,
    model = "mbsplsxy",
    pipeop_id = pipeop_id %||% NA_character_,
    n_components = length(component_names),
    n_blocks = length(blocks),
    performance_metric = state$performance_metric %||% NA_character_,
    correlation_method = state$correlation_method %||% NA_character_,
    run_id = state$run_id %||% NA_character_,
    emit_y_scores = isTRUE(state$emit_y_scores)
  )

  block_rows = list()
  for (k in seq_along(component_names)) {
    for (bn in names(blocks)) {
      wb = weights_x[[k]][[bn]] %||% numeric(0)
      if (is.null(names(wb))) {
        names(wb) = blocks[[bn]][seq_along(wb)]
      }
      wb = wb[blocks[[bn]]]
      wb[is.na(wb)] = 0
      block_rows[[length(block_rows) + 1L]] = data.table::data.table(
        component = component_names[[k]],
        block = bn,
        n_features = length(blocks[[bn]]),
        n_selected = sum(abs(as.numeric(wb)) > 0)
      )
    }
  }
  blocks_dt = data.table::rbindlist(block_rows, use.names = TRUE, fill = TRUE)

  components = data.table::data.table(component = component_names)
  if (is.list(weights_y) && length(weights_y)) {
    components[, n_target_columns := vapply(weights_y, length, integer(1))]
  }

  out = list(
    overview = overview,
    components = components,
    blocks = blocks_dt
  )

  if (isTRUE(include_weights)) {
    x_weights = mb_feature_weights_table(weights = weights_x, loadings = loadings_x, blocks = blocks, component_names = component_names)
    y_weights = if (is.list(weights_y) && length(weights_y)) {
      rows = list()
      for (k in seq_along(weights_y)) {
        wy = weights_y[[k]]
        py = loadings_y[[k]] %||% rep(NA_real_, length(wy))
        feats = names(wy) %||% target_cols %||% paste0(".Y_", seq_along(wy))
        if (is.null(names(wy))) {
          names(wy) = feats
        }
        if (is.null(names(py))) {
          names(py) = feats
        }
        py = py[feats]
        rows[[length(rows) + 1L]] = data.table::data.table(
          component = component_names[[k]],
          block = ".target",
          feature = feats,
          weight = as.numeric(wy[feats]),
          loading = as.numeric(py),
          selected = abs(as.numeric(wy[feats])) > 0
        )
      }
      data.table::rbindlist(rows, use.names = TRUE, fill = TRUE)
    } else {
      data.table::data.table(component = character(), block = character(), feature = character(), weight = numeric(), loading = numeric(), selected = logical())
    }
    out$weights = data.table::rbindlist(list(x_weights, y_weights), use.names = TRUE, fill = TRUE)
  }

  out
}
