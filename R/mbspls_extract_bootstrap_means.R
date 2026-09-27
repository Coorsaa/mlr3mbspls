#' Extract mean bootstrap weights with CI/frequency filtering
#'
#' Reads the bootstrap summaries stored by [PipeOpMBsPLSBootstrapSelect]:
#' \code{weights_ci} (means and percentile intervals of the replicate weights,
#' already matched to the training components, sign-aligned block by block and
#' filtered by the \code{min_score_cor} acceptance gate) and
#' \code{weights_selectfreq}. Nothing is re-aligned here.
#'
#' For legacy inputs that only provide per-replicate draws in
#' \code{weights_boot_draws} (columns \code{component}, \code{block},
#' \code{feature}, \code{replicate}, \code{weight}), the draws must already be
#' sign-aligned; they are summarised with type-7 percentile intervals.
#'
#' @param model A trained [mlr3pipelines::GraphLearner] or
#'   [mlr3pipelines::Graph] containing a [PipeOpMBsPLSBootstrapSelect] node,
#'   the trained PipeOp itself, its state (e.g.
#'   \code{glearner$model$mbspls_bootstrap_select}), or a list holding that
#'   state under \code{mbspls_bootstrap_select}.
#' @param component Integer LC index (e.g., 1).
#' @param filter_method One of c("ci","frequency").
#'   - "ci": keep features whose interval is strictly above or below zero and
#'     whose |mean| exceeds the selector's \code{magnitude_threshold}
#'     (default \code{1e-3}).
#'   - "frequency": keep features with selection freq >= filter_level.
#' @param filter_level Numeric threshold:
#'   - if filter_method == "ci": confidence level in `(0,1)`. Stored summaries
#'     are available only at the level used for training, \code{1 - alpha},
#'     which is the default; another level is an error (retrain with a
#'     different \code{alpha}). States that do not record \code{alpha} have
#'     intervals of unknown level: \code{filter_level} is then ignored with a
#'     warning.
#'   - if filter_method == "frequency": minimum freq in `[0,1]`, default the
#'     selector's \code{frequency_threshold} (0.5 if it is not stored).
#' @return A data.table with columns \code{component}, \code{block},
#'   \code{feature}, \code{mean}, \code{ci_low}, \code{ci_high} and \code{freq}
#'   (\code{NA} if frequencies are unavailable) for the features that pass the
#'   filter, sorted by block and decreasing |mean|.
#'
#' @export
mbspls_extract_bootstrap_means = function(
  model,
  component = 1,
  filter_method = c("ci", "frequency"),
  filter_level = NULL
) {
  filter_method = match.arg(filter_method)
  component = checkmate::assert_count(component, positive = TRUE, coerce = TRUE)
  comp_lab = sprintf("LC_%02d", component)
  st = .mbspls_bootstrap_select_state(model)

  alpha = st$alpha
  ci_tbl = st$weights_ci
  if (!is.null(ci_tbl) && nrow(ci_tbl)) {
    sum_dt = data.table::as.data.table(ci_tbl)
    if (filter_method == "ci") {
      level = if (is.null(alpha)) 0.95 else 1 - as.numeric(alpha)
      if (!is.null(filter_level)) {
        checkmate::assert_number(filter_level, lower = 0, upper = 1, .var.name = "filter_level")
        if (is.null(alpha)) {
          warning(
            "The selection state does not record the level of its stored intervals (no 'alpha'); 'filter_level' is ignored and the stored intervals are used as they are.",
            call. = FALSE
          )
        } else if (abs(filter_level - level) > sqrt(.Machine$double.eps)) {
          stop(sprintf(
            "The stored intervals have level %s (alpha = %s). Retrain the bootstrap selection with alpha = %s to filter at level %s.",
            format(level), format(alpha), format(1 - filter_level), format(filter_level)
          ), call. = FALSE)
        }
      }
    }
    sum_dt = sum_dt[as.character(sum_dt$component) == comp_lab, ]
    sum_dt = data.table::data.table(
      block = as.character(sum_dt$block),
      feature = as.character(sum_dt$feature),
      mean = as.numeric(sum_dt$boot_mean),
      ci_low = as.numeric(sum_dt$ci_lower),
      ci_high = as.numeric(sum_dt$ci_upper)
    )
  } else if (!is.null(st$weights_boot_draws) && nrow(st$weights_boot_draws)) {
    level = if (is.null(alpha)) 0.95 else 1 - as.numeric(alpha)
    if (filter_method == "ci" && !is.null(filter_level)) {
      level = checkmate::assert_number(filter_level, lower = 0, upper = 1, .var.name = "filter_level")
    }
    draws = data.table::as.data.table(st$weights_boot_draws)
    draws = draws[as.character(draws$component) == comp_lab, ]
    lower_p = (1 - level) / 2
    sum_dt = draws[, list(
      mean = mean(weight, na.rm = TRUE),
      ci_low = stats::quantile(weight, lower_p, na.rm = TRUE, names = FALSE),
      ci_high = stats::quantile(weight, 1 - lower_p, na.rm = TRUE, names = FALSE)
    ), by = list(block = as.character(block), feature = as.character(feature))]
  } else {
    stop(
      "No bootstrap summaries found: the selection state has neither 'weights_ci' nor 'weights_boot_draws'. Train PipeOpMBsPLSBootstrapSelect with bootstrap = TRUE.",
      call. = FALSE
    )
  }
  if (!nrow(sum_dt)) stop("No bootstrap summaries for ", comp_lab, ".", call. = FALSE)

  # --- frequency table (if available)
  freq_tbl = st$weights_selectfreq
  has_freq = is.data.frame(freq_tbl) && nrow(freq_tbl) &&
    all(c("component", "block", "feature", "freq") %in% names(freq_tbl))
  if (has_freq) {
    freq_tbl = data.table::as.data.table(freq_tbl)
    freq_tbl = freq_tbl[as.character(freq_tbl$component) == comp_lab, ]
    idx = match(
      paste(sum_dt$block, sum_dt$feature, sep = "\r"),
      paste(as.character(freq_tbl$block), as.character(freq_tbl$feature), sep = "\r")
    )
    sum_dt$freq = as.numeric(freq_tbl$freq)[idx]
  } else {
    sum_dt$freq = NA_real_
  }

  # --- apply filter
  if (filter_method == "ci") {
    magnitude = as.numeric(st$magnitude_threshold %||% 1e-3)
    keep = (sum_dt$ci_low > 0 | sum_dt$ci_high < 0) & abs(sum_dt$mean) > magnitude
  } else {
    if (!has_freq) {
      stop("Requested filter_method='frequency' but the selection state has no 'weights_selectfreq' ",
        "with columns component/block/feature/freq.", call. = FALSE)
    }
    filter_level = filter_level %||% as.numeric(st$frequency_threshold %||% 0.5)
    checkmate::assert_number(filter_level, lower = 0, upper = 1, .var.name = "filter_level")
    keep = sum_dt$freq >= filter_level
  }
  keep[is.na(keep)] = FALSE
  out = sum_dt[keep, ]

  # order: by block then |mean| desc
  out = out[order(out$block, -abs(out$mean)), ]
  out = cbind(data.table::data.table(component = rep(comp_lab, nrow(out))), out)
  out[]
}

# Locate the bootstrap-select state in the supported inputs.
.mbspls_bootstrap_select_state = function(model) {
  is_state = function(x) {
    is.list(x) && (!is.null(x$weights_ci) || !is.null(x$weights_boot_draws))
  }
  if (inherits(model, "GraphLearner")) {
    st = .mbspls_locate_nodes_general(model)$sel_state
  } else if (inherits(model, "Graph")) {
    st = .mbspls_locate_nodes_graph_general(model)$sel_state
  } else if (inherits(model, "PipeOp")) {
    st = model$state
  } else if (is_state(model)) {
    st = model
  } else if (is.list(model) && !is.null(model$mbspls_bootstrap_select)) {
    st = model$mbspls_bootstrap_select
    if (inherits(st, "PipeOp")) st = st$state
  } else {
    st = NULL
  }
  if (!is_state(st)) {
    stop(
      "Could not find a trained PipeOpMBsPLSBootstrapSelect state with bootstrap summaries in 'model'.",
      call. = FALSE
    )
  }
  st
}
