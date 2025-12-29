#' Autoplot for GraphLearner with MB-sPLS (branch-aware, source-consistent)
#'
#' Requires: ggplot2, dplyr, tibble
#' Optional: patchwork, ggraph, igraph, RColorBrewer, scales, hexbin
#'
#' @importFrom ggplot2 autoplot
#'
#' @param object [GraphLearner]
#'   Trained [GraphLearner].
#' @param type [character]
#'   Plot type identifier. See Details.
#' @param ...
#'   Additional arguments passed down to the corresponding plot helper.
#' @export
#' @method autoplot GraphLearner
autoplot.GraphLearner = function(object,
  type = c("mbspls_weights", "mbspls_heatmap", "mbspls_network",
    "mbspls_variance", "mbspls_scree", "mbspls_scores",
    "mbspls_bootstrap_component", "mbspls_bootstrap_comp"),
  ...) {

  type = match.arg(type)
  dots = list(...)

  # --- safe merge of ... with explicit arguments (prevents duplicate formals)
  .args_merge = function(explicit, dots, fun) {
    if (!length(dots)) {
      return(explicit)
    }
    fml = names(formals(fun))
    if (is.null(fml)) fml <- character(0)
    keep = intersect(setdiff(names(dots), names(explicit)), fml)
    c(explicit, dots[keep])
  }

  # --- locate MB-sPLS fit and (optional) selection nodes + their log_env
  nodes = .mbspls_locate_nodes_general(
    object,
    mbspls_id = dots$mbspls_id %||% NULL,
    select_id = dots$select_id %||% NULL
  )
  use_env = nodes$fit_env %||% nodes$sel_env

  # ----------------- weights ---------------------------------------------------
  if (type == "mbspls_weights") {
    fun = .mbspls_plot_weights_patchwork
    explicit = list(
      fit_state          = nodes$fit_state,
      sel_state          = nodes$sel_state,
      source             = (dots$source %||% "bootstrap"),
      top_n              = dots$top_n %||% NULL,
      patch_ncol         = as.integer(dots$patch_ncol %||% 3L),
      font               = dots$font %||% "sans",
      alpha_by_stability = isTRUE(dots$alpha_by_stability %||% FALSE),
      alpha_nonstable    = as.numeric(dots$alpha_nonstable %||% 0.4),
      freq_min           = dots$freq_min %||% NULL,
      ci_filter          = dots$ci_filter %||% "excludes_zero"
    )
    return(do.call(fun, .args_merge(explicit, dots, fun)))
  }

  # -------- all other types use the same node resolution -----------------------
  if (type == "mbspls_heatmap") {
    fun = .mbspls_plot_heatmap_general
    explicit = list(
      model        = nodes$fit_state,
      sel_state    = nodes$sel_state,
      log_env      = use_env,
      source       = (dots$source %||% "bootstrap"),
      freq_min     = dots$freq_min %||% NULL,
      ci_filter    = dots$ci_filter %||% "excludes_zero",
      compare      = dots$compare %||% "lv", # "lv" | "lc"
      method       = dots$method %||% "spearman",
      absolute     = isTRUE(dots$absolute %||% TRUE),
      title_suffix = ""
    )
    return(do.call(fun, .args_merge(explicit, dots, fun)))
  }

  if (type == "mbspls_network") {
    fun = .mbspls_plot_network_general
    explicit = list(
      model        = nodes$fit_state,
      sel_state    = nodes$sel_state,
      log_env      = use_env,
      source       = (dots$source %||% "bootstrap"),
      freq_min     = dots$freq_min %||% NULL,
      ci_filter    = dots$ci_filter %||% "excludes_zero",
      compare      = dots$compare %||% "lv", # "lv" | "lc"
      method       = dots$method %||% "spearman",
      absolute     = isTRUE(dots$absolute %||% TRUE),
      cutoff       = as.numeric(dots$cutoff %||% 0.3),
      title_suffix = ""
    )
    return(do.call(fun, .args_merge(explicit, dots, fun)))
  }

  if (type == "mbspls_variance") {
    fun = .mbspls_plot_variance_general
    explicit = list(
      model        = nodes$fit_state,
      sel_state    = nodes$sel_state,
      log_env      = use_env,
      source       = (dots$source %||% "bootstrap"),
      freq_min     = dots$freq_min %||% NULL,
      ci_filter    = dots$ci_filter %||% "excludes_zero",
      layout       = dots$layout %||% "grouped",
      show_values  = isTRUE(dots$show_values %||% TRUE),
      show_total   = isTRUE(dots$show_total %||% TRUE),
      accuracy     = as.numeric(dots$accuracy %||% 1),
      title_suffix = ""
    )
    return(do.call(fun, .args_merge(explicit, dots, fun)))
  }

  if (type == "mbspls_scree") {
    fun = .mbspls_plot_scree_general
    explicit = list(
      model        = nodes$fit_state,
      sel_state    = nodes$sel_state,
      log_env      = use_env,
      source       = (dots$source %||% "bootstrap"),
      freq_min     = dots$freq_min %||% NULL,
      ci_filter    = dots$ci_filter %||% "excludes_zero",
      cumulative   = isTRUE(dots$cumulative %||% FALSE),
      title_suffix = ""
    )
    return(do.call(fun, .args_merge(explicit, dots, fun)))
  }

  if (type == "mbspls_scores") {
    fun = .mbspls_plot_scores_general
    explicit = list(
      model        = nodes$fit_state,
      sel_state    = nodes$sel_state,
      log_env      = use_env,
      source       = (dots$source %||% "bootstrap"),
      freq_min     = dots$freq_min %||% NULL,
      ci_filter    = dots$ci_filter %||% "excludes_zero",
      component    = as.integer(dots$component %||% 1L),
      standardize  = isTRUE(dots$standardize %||% TRUE),
      density      = dots$density %||% "none",
      annotate     = isTRUE(dots$annotate %||% TRUE),
      title_suffix = ""
    )
    return(do.call(fun, .args_merge(explicit, dots, fun)))
  }

  if (type %in% c("mbspls_bootstrap_component", "mbspls_bootstrap_comp")) {
    fun = .mbspls_plot_bootstrap_component
    explicit = list(
      model      = object,
      payload    = NULL,
      mbspls_id  = dots$mbspls_id %||% NULL
    )
    return(do.call(fun, .args_merge(explicit, dots, fun)))
  }

  stop("Unknown type: ", type)
}

#' @importFrom ggplot2 autoplot
#' @export
#' @method autoplot Graph
autoplot.Graph = function(object, type = c("mbspls_weights"), ...) {
  type = match.arg(type)
  dots = list(...)

  .args_merge = function(explicit, dots, fun) {
    if (!length(dots)) {
      return(explicit)
    }
    fml = names(formals(fun))
    if (is.null(fml)) fml <- character(0)
    keep = intersect(setdiff(names(dots), names(explicit)), fml)
    c(explicit, dots[keep])
  }

  nodes = .mbspls_locate_nodes_graph_general(
    object,
    mbspls_id = dots$mbspls_id %||% NULL,
    select_id = dots$select_id %||% NULL
  )

  if (type == "mbspls_weights") {
    fun = .mbspls_plot_weights_patchwork
    explicit = list(
      fit_state          = nodes$fit_state,
      sel_state          = nodes$sel_state,
      source             = (dots$source %||% "bootstrap"),
      top_n              = dots$top_n %||% NULL,
      patch_ncol         = as.integer(dots$patch_ncol %||% 3L),
      font               = dots$font %||% "sans",
      alpha_by_stability = isTRUE(dots$alpha_by_stability %||% FALSE),
      alpha_nonstable    = as.numeric(dots$alpha_nonstable %||% 0.4),
      freq_min           = dots$freq_min %||% NULL,
      ci_filter          = dots$ci_filter %||% "excludes_zero"
    )
    return(do.call(fun, .args_merge(explicit, dots, fun)))
  }

  stop("Unknown type: ", type)
}

# ------------------------------------------------------------------------------
# Node location (fit + selection + their log_env)
# ------------------------------------------------------------------------------
.mbspls_locate_nodes_general = function(gl, mbspls_id = NULL, select_id = NULL) {
  if (!inherits(gl, "GraphLearner")) stop("Expected a GraphLearner.", call. = FALSE)
  mod = gl$model
  if (is.null(mod)) stop("GraphLearner appears to be untrained (model is NULL).", call. = FALSE)

  # Fit node
  if (is.null(mbspls_id)) {
    cand = names(gl$graph$pipeops)[vapply(gl$graph$pipeops, inherits, logical(1), "PipeOpMBsPLS")]
    if (!length(cand)) stop("No PipeOpMBsPLS node found in the graph.")
    mbspls_id = cand[1]
  }
  fit_po = gl$graph$pipeops[[mbspls_id]]
  fit = mod[[mbspls_id]] %||% fit_po
  if (is.null(fit)) stop("Cannot locate MB-sPLS node '", mbspls_id, "'.")
  fit_state = if (!is.null(fit$state)) fit$state else fit
  fit_env = tryCatch({
    ve = fit_po$param_set$values$log_env
    if (inherits(ve, "environment")) ve else NULL
  }, error = function(e) NULL)

  # Selection node
  sel_state = NULL
  sel_env = NULL
  if (!is.null(select_id)) {
    sel_po = gl$graph$pipeops[[select_id]]
    sel = mod[[select_id]] %||% sel_po
    if (is.null(sel)) stop("Cannot locate selection node '", select_id, "'.")
    sel_state = if (!is.null(sel$state)) sel$state else sel
    sel_env = tryCatch({
      ve = sel_po$param_set$values$log_env
      if (inherits(ve, "environment")) ve else NULL
    }, error = function(e) NULL)
  } else {
    sel = mod$mbspls_bootstrap_select %||% gl$graph$pipeops$mbspls_bootstrap_select
    sel_po = gl$graph$pipeops$mbspls_bootstrap_select
    if (is.null(sel)) {
      hits = names(gl$graph$pipeops)[vapply(gl$graph$pipeops, inherits, logical(1), "PipeOpMBsPLSBootstrapSelect")]
      if (length(hits)) {
        sel = mod[[hits[1]]] %||% gl$graph$pipeops[[hits[1]]]
        sel_po = gl$graph$pipeops[[hits[1]]]
      }
    }
    if (!is.null(sel)) {
      sel_state = if (!is.null(sel$state)) sel$state else sel
      sel_env = tryCatch({
        ve = sel_po$param_set$values$log_env
        if (inherits(ve, "environment")) ve else NULL
      }, error = function(e) NULL)
    }
  }

  list(fit_state = fit_state, sel_state = sel_state, fit_env = fit_env, sel_env = sel_env)
}

.mbspls_locate_nodes_graph_general = function(gr, mbspls_id = NULL, select_id = NULL) {
  if (!inherits(gr, "Graph")) stop("Expected a Graph.", call. = FALSE)

  # Fit node
  if (is.null(mbspls_id)) {
    cand = names(gr$pipeops)[vapply(gr$pipeops, inherits, logical(1), "PipeOpMBsPLS")]
    if (!length(cand)) stop("No PipeOpMBsPLS node found in the graph.")
    mbspls_id = cand[1]
  }
  fit_po = gr$pipeops[[mbspls_id]]
  fit_state = gr$state[[mbspls_id]] %||% fit_po$state
  if (is.null(fit_state)) stop("Cannot locate trained state for MB-sPLS node '", mbspls_id, "'.")

  # Selection node (optional / auto)
  sel_state = NULL
  if (!is.null(select_id)) {
    sel_po = gr$pipeops[[select_id]]
    sel_state = gr$state[[select_id]] %||% sel_po$state
  } else {
    # try conventional id first
    if (!is.null(gr$pipeops$mbspls_bootstrap_select)) {
      sel_po = gr$pipeops$mbspls_bootstrap_select
      sel_state = gr$state$mbspls_bootstrap_select %||% sel_po$state
    } else {
      hits = names(gr$pipeops)[vapply(gr$pipeops, inherits, logical(1), "PipeOpMBsPLSBootstrapSelect")]
      if (length(hits)) {
        sel_po = gr$pipeops[[hits[1]]]
        sel_state = gr$state[[hits[1]]] %||% sel_po$state
      }
    }
  }

  list(fit_state = fit_state, sel_state = sel_state)
}

# ------------------------------------------------------------------------------
# Palette helpers
# ------------------------------------------------------------------------------
.mbspls_pal = function() {
  if (!requireNamespace("RColorBrewer", quietly = TRUE)) {
    return(c(`TRUE` = "#1b9e77", `FALSE` = "#d95f02"))
  }
  pal = RColorBrewer::brewer.pal(3, "Dark2")
  c(`TRUE` = pal[1], `FALSE` = pal[3])
}

.nice_label = function(x) gsub("_", " ", x, fixed = TRUE)

# ------------------------------------------------------------------------------
# Weights plot helpers
# ------------------------------------------------------------------------------
.mbspls_feat_names = function(w, fallback = NULL) {
  nm = names(w)
  if (is.null(nm)) nm = attr(w, "names")
  if (is.null(nm)) {
    dn = dimnames(w)
    if (!is.null(dn) && length(dn) >= 1) nm = dn[[1]]
  }
  if (is.null(nm)) nm = rownames(w)
  if (is.null(nm)) nm = fallback
  nm
}


# Returns a list-of-lists W_use[[k]][[block]] = named numeric vector (full length)
.mbspls_weights_from_source = function(fit_state, sel_state = NULL,
  source = c("weights", "bootstrap"),
  freq_min = NULL,
  ci_filter = c("excludes_zero", "none", "overlaps_zero")) {
  source = match.arg(source)
  ci_filter = match.arg(ci_filter)
  blocks = fit_state$blocks
  bn = names(blocks)

  .fullvec = function(vals_map, block) {
    feats = blocks[[block]]
    out = as.numeric(vals_map[feats])
    out[is.na(out)] = 0
    stats::setNames(out, feats)
  }

  if (identical(source, "weights")) {
    return(fit_state$weights)
  }

  if (is.null(sel_state)) {
    stop("Bootstrap selection state not found; pass a valid 'select_id' and train with bootstrap selection.")
  }

  if (!is.null(freq_min)) {
    ci = as.data.frame(sel_state$weights_ci)
    fr = as.data.frame(sel_state$weights_selectfreq)
    if (is.null(ci) || !nrow(ci)) stop("weights_ci missing for freq_min path.")
    if (is.null(fr) || !nrow(fr)) stop("weights_selectfreq missing for freq_min path.")

    ci$component = as.character(ci$component)
    fr$component = as.character(fr$component)

    Klabs = unique(ci$component)
    W_use = vector("list", length(Klabs))
    names(W_use) = Klabs

    for (k in Klabs) {
      Wk = list()
      for (b in bn) {
        mu_b = ci[ci$component == k & ci$block == b, c("feature", "boot_mean"), drop = FALSE]
        fq_b = fr[fr$component == k & fr$block == b, c("feature", "freq"), drop = FALSE]
        if (!nrow(mu_b)) {
          Wk[[b]] = stats::setNames(numeric(length(blocks[[b]])), blocks[[b]])
          next
        }
        mu_map = stats::setNames(mu_b$boot_mean, mu_b$feature)
        if (nrow(fq_b)) {
          fq_map = stats::setNames(fq_b$freq, fq_b$feature)
          keep = fq_map[names(mu_map)] >= as.numeric(freq_min)
          keep[is.na(keep)] = FALSE
          mu_map[!keep] = 0
        } else {
          mu_map[] = 0
        }
        Wk[[b]] = .fullvec(mu_map, b)
      }
      W_use[[k]] = Wk
    }
    if (!is.null(sel_state$weights_stable) && length(sel_state$weights_stable)) {
      W_use = lapply(names(sel_state$weights_stable), function(k) {
        lapply(names(blocks), function(b) .fullvec(sel_state$weights_stable[[k]][[b]], b))
      })
      names(W_use) = names(sel_state$weights_stable)
      names(W_use[[1]]) = names(blocks)
      return(W_use)
    }
  }

  if (!is.null(sel_state$weights_stable) && length(sel_state$weights_stable)) {
    return(sel_state$weights_stable)
  }

  ci = as.data.frame(sel_state$weights_ci)
  if (is.null(ci) || !nrow(ci)) {
    stop("No bootstrap summaries available: both weights_stable and weights_ci missing.")
  }

  Klabs = unique(ci$component)
  W_use = vector("list", length(Klabs))
  names(W_use) = Klabs

  for (k in Klabs) {
    Wk = list()
    for (b in bn) {
      sb = ci[ci$component == k & ci$block == b,
        c("feature", "boot_mean", "ci_lower", "ci_upper"), drop = FALSE]
      if (!nrow(sb)) {
        Wk[[b]] = stats::setNames(numeric(length(blocks[[b]])), blocks[[b]])
        next
      }
      keep = switch(match.arg(ci_filter),
        excludes_zero = ((sb$ci_lower >= 0) | (sb$ci_upper <= 0)) & (abs(sb$boot_mean) > 1e-3),
        overlaps_zero = (sb$ci_lower <= 0 & sb$ci_upper >= 0),
        none          = rep(TRUE, nrow(sb))
      )
      mu_map = stats::setNames(ifelse(keep, sb$boot_mean, 0), sb$feature)
      Wk[[b]] = .fullvec(mu_map, b)
    }
    W_use[[k]] = Wk
  }
  W_use
}

# ------------------------------------------------------------------------------
# Scores + EV + objective recomputation (deflation)
# ------------------------------------------------------------------------------
# Returns list(T_mat, ev_block, ev_comp, obj_vec)
.mbspls_recompute_from_weights = function(fit_state, W_use, log_env = NULL) {

  # --- find X by blocks (robust paths)
  X_list = fit_state$X_train_blocks %||%
    fit_state$X_blocks_train %||%
    (if (inherits(log_env, "environment")) log_env$mbspls_state$X_train_blocks else NULL) %||%
    (if (inherits(log_env, "environment")) log_env$mbspls_state$X_blocks_train else NULL) %||%
    (if (inherits(log_env, "environment")) log_env$X_train_blocks else NULL)

  # also resolve blocks order from env if needed
  blocks_map = fit_state$blocks %||%
    (if (inherits(log_env, "environment")) log_env$mbspls_state$blocks else NULL)

  if (is.null(X_list) || is.null(blocks_map)) {
    stop("Cannot recompute from weights: X_train_blocks missing. Train with store_train_blocks=TRUE and pass the matching mbspls_id/select_id.")
  }

  blocks = names(blocks_map)
  K = length(W_use)
  if (!K) stop("No components in W_use.")

  # copy matrices for deflation
  X_cur = lapply(X_list, function(m) {
    storage.mode(m) = "double"
    m
  })
  T_tabs = vector("list", K)
  ev_blk = matrix(0, nrow = K, ncol = length(blocks))
  colnames(ev_blk) = blocks
  rownames(ev_blk) = sprintf("LC_%02d", seq_len(K))
  obj_vec = numeric(K)

  ss_tot_orig = vapply(blocks, function(b) sum(X_list[[b]]^2), numeric(1))
  for (k in seq_len(K)) {
    Tk = matrix(0, nrow = nrow(X_cur[[1]]), ncol = length(blocks))
    colnames(Tk) = paste0("LV", k, "_", blocks)

    for (bi in seq_along(blocks)) {
      b = blocks[bi]
      w_b = W_use[[k]][[b]]
      if (is.null(w_b) || !any(is.finite(w_b)) || all(w_b == 0)) {
        Tk[, bi] = 0
        next
      }
      cols = colnames(X_cur[[b]])
      wv = if (!is.null(names(w_b))) as.numeric(w_b[cols]) else as.numeric(w_b)
      wv[is.na(wv)] = 0
      t_b = drop(X_cur[[b]] %*% wv)
      Tk[, bi] = t_b

      denom = sum(t_b * t_b)
      pb = if (denom > 0) drop(crossprod(X_cur[[b]], t_b) / denom) else rep(0, ncol(X_cur[[b]]))
      if (ss_tot_orig[bi] > 0 && denom > 0) {
        X_hat = tcrossprod(t_b, pb)
        ev_blk[k, b] = if (denom > 0 && ss_tot_orig[bi] > 0) sum(X_hat^2) / ss_tot_orig[bi] else 0

      } else {
        ev_blk[k, b] = 0
      }
    }

    if (ncol(Tk) >= 2) {
      Ck = suppressWarnings(stats::cor(Tk, use = "pairwise.complete.obs"))
      Ck[!is.finite(Ck)] = 0
      obj_vec[k] = mean(abs(Ck[upper.tri(Ck)]))
    } else {
      obj_vec[k] = 0
    }

    # deflation
    for (bi in seq_along(blocks)) {
      b = blocks[bi]
      t_b = Tk[, bi]
      denom = sum(t_b * t_b)
      if (denom <= 0) next
      pb = drop(crossprod(X_cur[[b]], t_b) / denom)
      X_cur[[b]] = X_cur[[b]] - tcrossprod(t_b, pb)
    }

    T_tabs[[k]] = Tk
  }

  T_mat = do.call(cbind, T_tabs)
  w_blocks = ss_tot_orig / sum(ss_tot_orig)
  ev_comp = as.numeric(ev_blk %*% w_blocks)


  list(T_mat = as.matrix(T_mat), ev_block = ev_blk, ev_comp = ev_comp, obj_vec = obj_vec)
}

# ------------------------------------------------------------------------------
# ------------------- HEATMAP (compare = "lv" | "lc") ---------------------------
# ------------------------------------------------------------------------------
.mbspls_plot_heatmap_general = function(
  model, sel_state = NULL, log_env = NULL,
  source = c("bootstrap", "weights"),
  freq_min = NULL,
  ci_filter = c("excludes_zero", "none", "overlaps_zero"),
  compare = c("lv", "lc"),
  method = c("spearman", "pearson"),
  absolute = TRUE,
  title_suffix = "",
  font = "sans"
) {
  requireNamespace("ggplot2")
  requireNamespace("scales")

  source = match.arg(source)
  ci_filter = match.arg(ci_filter)
  compare = match.arg(compare)
  method = match.arg(method)

  W_use = .mbspls_weights_from_source(model, sel_state, source, freq_min, ci_filter)
  rec = .mbspls_recompute_from_weights(model, W_use, log_env = log_env)
  Tmat = rec$T_mat

  blocks = names(model$blocks)
  K = length(W_use)
  cols_for_k = function(k) grep(paste0("^LV", k, "_"), colnames(Tmat), value = TRUE)

  if (compare == "lv") {
    dfl = lapply(seq_len(K), function(k) {
      cn = cols_for_k(k)
      if (length(cn) < 2) {
        return(NULL)
      }
      S = as.data.frame(Tmat[, cn, drop = FALSE])
      colnames(S) = sub("^LV\\d+_", "", cn)
      C = suppressWarnings(stats::cor(S, method = method, use = "pairwise.complete.obs"))
      df = as.data.frame(as.table(C), stringsAsFactors = FALSE)
      names(df) = c("block_x", "block_y", "r")
      df$component = sprintf("LC %d", k)
      df
    })
    dfl = Filter(Negate(is.null), dfl)
    if (!length(dfl)) stop("Not enough LV block-columns to compute correlations.")
    df = do.call(rbind, dfl)
    df$block_x = factor(df$block_x, levels = blocks)
    df$block_y = factor(df$block_y, levels = blocks)

    p = ggplot2::ggplot(df, ggplot2::aes(block_x, block_y, fill = if (absolute) abs(r) else r)) +
      ggplot2::geom_tile(color = "white", linewidth = 0.15) +
      ggplot2::geom_text(
        ggplot2::aes(label = ifelse(is.na(r), "", formatC(if (absolute) abs(r) else r, format = "f", digits = 2))),
        size = 3, na.rm = TRUE
      ) +
      {
        if (absolute) {
          ggplot2::scale_fill_viridis_c(limits = c(0, 1), na.value = "white")
        } else {
          ggplot2::scale_fill_gradient2(
            low = scales::muted("blue", l = 30, c = 80),
            high = scales::muted("red", l = 50, c = 100),
            limits = c(-1, 1), midpoint = 0,
            na.value = "white"
          )
        }
      } +
      ggplot2::facet_wrap(~component) +
      ggplot2::coord_equal() +
      ggplot2::labs(
        x = NULL, y = NULL, fill = if (absolute) "|r|" else "r",
        title = paste0("Cross-block LV correlation heatmap", title_suffix),
        subtitle = sprintf("Method: %s; one panel per component", method)
      ) +
      ggplot2::theme_minimal(base_size = 11, base_family = font) +
      ggplot2::theme(panel.spacing = grid::unit(0.7, "lines"),
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
    return(p)
  }

  # compare == "lc": LC-by-LC
  LC = matrix(NA_real_, nrow = nrow(Tmat), ncol = K)
  colnames(LC) = paste0("LC_", sprintf("%02d", seq_len(K)))
  for (k in seq_len(K)) {
    cn = cols_for_k(k)
    if (!length(cn)) next
    Z = scale(Tmat[, cn, drop = FALSE])
    Z = as.matrix(Z)
    LC[, k] = rowMeans(Z, na.rm = TRUE)
  }

  C = suppressWarnings(stats::cor(LC, method = method, use = "pairwise.complete.obs"))
  df = as.data.frame(as.table(C), stringsAsFactors = FALSE)
  names(df) = c("lc_x", "lc_y", "r")
  df$lc_x = factor(df$lc_x, levels = colnames(LC))
  df$lc_y = factor(df$lc_y, levels = colnames(LC))

  ggplot2::ggplot(df, ggplot2::aes(lc_x, lc_y, fill = if (absolute) abs(r) else r)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.15) +
    ggplot2::geom_text(
      ggplot2::aes(label = formatC(if (absolute) abs(r) else r, format = "f", digits = 2)),
      size = 3, na.rm = TRUE
    ) +
    {
      if (absolute) {
        ggplot2::scale_fill_viridis_c(limits = c(0, 1), na.value = "white")
      } else {
        ggplot2::scale_fill_gradient2(
          low = scales::muted("blue", l = 30, c = 80),
          high = scales::muted("red", l = 50, c = 100),
          limits = c(-1, 1), midpoint = 0,
          na.value = "white"
        )
      }
    } +
    ggplot2::coord_equal() +
    ggplot2::labs(
      x = NULL, y = NULL, fill = if (absolute) "|r|" else "r",
      title = paste0("LC-by-LC correlation heatmap", title_suffix),
      subtitle = sprintf("Method: %s; whole components", method)
    ) +
    ggplot2::theme_minimal(base_size = 11, base_family = font)
}

# ------------------------------------------------------------------------------
# ------------------- NETWORK (compare = "lv" | "lc") --------------------------
# ------------------------------------------------------------------------------
.mbspls_plot_network_general = function(
  model, sel_state = NULL, log_env = NULL,
  source = c("bootstrap", "weights"),
  freq_min = NULL,
  ci_filter = c("excludes_zero", "none", "overlaps_zero"),
  compare = c("lv", "lc"),
  method = c("spearman", "pearson"),
  absolute = TRUE,
  cutoff = 0.3,
  title_suffix = "",
  font = "sans"
) {
  requireNamespace("igraph")
  requireNamespace("ggraph")
  requireNamespace("ggplot2")
  requireNamespace("scales")

  source = match.arg(source)
  ci_filter = match.arg(ci_filter)
  compare = match.arg(compare)
  method = match.arg(method)
  cutoff = as.numeric(cutoff)

  W_use = .mbspls_weights_from_source(model, sel_state, source, freq_min, ci_filter)
  rec = .mbspls_recompute_from_weights(model, W_use, log_env = log_env)
  Tmat = rec$T_mat

  if (compare == "lc") {
    K = length(W_use)
    cols_for_k = function(k) grep(paste0("^LV", k, "_"), colnames(Tmat), value = TRUE)
    LC = matrix(NA_real_, nrow = nrow(Tmat), ncol = K)
    colnames(LC) = paste0("LC_", sprintf("%02d", seq_len(K)))
    for (k in seq_len(K)) {
      cn = cols_for_k(k)
      if (!length(cn)) next
      Z = scale(Tmat[, cn, drop = FALSE])
      Z = as.matrix(Z)
      LC[, k] = rowMeans(Z, na.rm = TRUE)
    }
    S = LC
    node_names = colnames(S)
  } else {
    S = Tmat
    node_names = colnames(S)
  }

  C = suppressWarnings(stats::cor(S, method = method, use = "pairwise.complete.obs"))
  diag(C) = 0
  idx = which(abs(C) >= cutoff, arr.ind = TRUE)
  idx = idx[idx[, 1] < idx[, 2], , drop = FALSE]
  if (!nrow(idx)) {
    max_cor = max(abs(C[upper.tri(C)]), na.rm = TRUE)
    stop(sprintf("No pairs exceed cutoff (%.2f). Max |r| = %.3f", cutoff, max_cor), call. = FALSE)
  }

  edges = data.frame(
    from = node_names[idx[, 1]],
    to = node_names[idx[, 2]],
    r = C[idx],
    stringsAsFactors = FALSE
  )
  g = igraph::graph_from_data_frame(edges, directed = FALSE)

  # cross-version edge guide detection
  edge_guide_fun = get0("guide_edge_colourbar", asNamespace("ggraph"))
  if (is.null(edge_guide_fun)) edge_guide_fun <- get0("guide_edge_colorbar", asNamespace("ggraph"))
  guide_obj = if (is.null(edge_guide_fun)) ggplot2::guide_colourbar() else edge_guide_fun()

  edge_col_scale = if (absolute) {
    # ggraph exposes a generic `scale_edge_colour_viridis()` (with
    # `discrete = FALSE`) rather than a dedicated *_viridis_c() helper.
    ggraph::scale_edge_colour_viridis(
      name = "|r|", limits = c(0, 1), guide = guide_obj,
      discrete = FALSE, option = "D"
    )
  } else {
    ggraph::scale_edge_colour_gradient2(
      name = "r",
      limits = c(-1, 1), midpoint = 0,
      low = scales::muted("blue", l = 30, c = 80),
      high = scales::muted("red", l = 50, c = 100),
      guide = guide_obj
    )
  }

  ggraph::ggraph(g, layout = "fr") +
    ggraph::geom_edge_link(ggplot2::aes(width = abs(.data$r), colour = if (absolute) abs(.data$r) else .data$r)) +
    ggraph::scale_edge_width(range = c(0.3, 3), guide = "none") +
    edge_col_scale +
    ggraph::geom_node_point(size = 4, colour = "grey30") +
    ggraph::geom_node_text(ggplot2::aes(label = name), repel = TRUE, size = 3) +
    ggplot2::theme_void(base_family = font) +
    ggplot2::labs(
      title = if (compare == "lc") {
        sprintf("Component network (|r| >= %.2f, %s)%s", cutoff, method, title_suffix)
      } else {
        sprintf("LV network (|r| >= %.2f, %s)%s", cutoff, method, title_suffix)
      }
    )
}

# ------------------------------------------------------------------------------
# -------------------- VARIANCE (recomputed) -----------------------------------
# ------------------------------------------------------------------------------
.mbspls_plot_variance_general = function(
  model, sel_state = NULL, log_env = NULL,
  source = c("bootstrap", "weights"),
  freq_min = NULL,
  ci_filter = c("excludes_zero", "none", "overlaps_zero"),
  show_total = TRUE,
  viridis_option = "D",
  layout = c("grouped", "facet"),
  show_values = TRUE,
  accuracy = 1,
  title_suffix = "",
  font = "sans"
) {
  requireNamespace("ggplot2")
  requireNamespace("scales")
  source = match.arg(source)
  ci_filter = match.arg(ci_filter)
  layout = match.arg(layout)

  W_use = .mbspls_weights_from_source(model, sel_state, source, freq_min, ci_filter)
  rec = .mbspls_recompute_from_weights(model, W_use, log_env = log_env)
  ev_blk = rec$ev_block
  ev_comp = rec$ev_comp

  rn = rownames(ev_blk)
  if (is.null(rn)) rn <- paste0("LC_", sprintf("%02d", seq_len(nrow(ev_blk))))
  cn = colnames(ev_blk)
  if (is.null(cn)) cn <- names(model$blocks)

  df_long = data.frame(
    component = factor(rep(rn, each = length(cn)), levels = rn),
    block = factor(rep(cn, times = length(rn)), levels = cn),
    explained = as.vector(t(ev_blk)),
    check.names = FALSE
  )

  ev_table = data.frame(component = rn, check.names = FALSE)
  for (j in seq_along(cn)) ev_table[[cn[j]]] = ev_blk[, j]
  ev_table$total = as.numeric(ev_comp)

  ev_table_percent = ev_table
  if (ncol(ev_table_percent) > 1) ev_table_percent[-1] <- lapply(ev_table_percent[-1], scales::percent, accuracy = accuracy)

  p_base = if (layout == "grouped") {
    ggplot2::ggplot(df_long, ggplot2::aes(x = component, y = explained, fill = block)) +
      ggplot2::geom_col(position = ggplot2::position_dodge2(width = 0.9, preserve = "single"))
  } else {
    ggplot2::ggplot(df_long, ggplot2::aes(x = block, y = explained, fill = block)) +
      ggplot2::geom_col() +
      ggplot2::facet_wrap(~component, nrow = 1)
  }

  if (isTRUE(show_values)) {
    lab_fun = function(x) scales::percent(x, accuracy = accuracy)
    if (layout == "grouped") {
      p_base = p_base + ggplot2::geom_text(
        ggplot2::aes(label = lab_fun(explained)),
        position = ggplot2::position_dodge2(width = 0.9, preserve = "single"),
        vjust = -0.3, size = 3
      )
    } else {
      p_base = p_base + ggplot2::geom_text(
        ggplot2::aes(label = lab_fun(explained)),
        vjust = -0.3, size = 3
      )
    }
  }

  p = p_base +
    ggplot2::scale_y_continuous(labels = scales::label_percent(accuracy = accuracy),
      expand = ggplot2::expansion(mult = c(0, 0.12))) +
    ggplot2::scale_fill_viridis_d(option = viridis_option) +
    ggplot2::labs(
      y = "Variance explained", x = NULL,
      title = paste0("MB-sPLS: block-wise variance per component (recomputed from ", source, ")", title_suffix)
    ) +
    ggplot2::guides(fill = ggplot2::guide_legend(title = "block")) +
    ggplot2::theme_minimal(base_size = 11, base_family = font) +
    ggplot2::theme(legend.position = "right")

  if (isTRUE(show_total)) {
    df_tot = data.frame(component = factor(rn, levels = rn), total = as.numeric(ev_comp))
    if (layout == "grouped") {
      p = p +
        ggplot2::geom_line(data = df_tot, ggplot2::aes(component, total, group = 1),
          inherit.aes = FALSE) +
        ggplot2::geom_point(data = df_tot, ggplot2::aes(component, total),
          inherit.aes = FALSE, size = 1.9)
    } else {
      p = p + ggplot2::geom_point(
        data = df_tot, ggplot2::aes(x = Inf, y = total),
        inherit.aes = FALSE, size = 1.9, alpha = 0.9
      )
    }
  }

  attr(p, "ev_table") = ev_table
  attr(p, "ev_table_percent") = ev_table_percent
  p
}

# ------------------------------------------------------------------------------
# -------------------- SCREE (recomputed objective) ----------------------------
# ------------------------------------------------------------------------------
.mbspls_plot_scree_general = function(
  model, sel_state = NULL, log_env = NULL,
  source = c("bootstrap", "weights"),
  freq_min = NULL,
  ci_filter = c("excludes_zero", "none", "overlaps_zero"),
  cumulative = FALSE,
  title_suffix = "",
  font = "sans"
) {
  requireNamespace("ggplot2")
  source = match.arg(source)
  ci_filter = match.arg(ci_filter)

  W_use = .mbspls_weights_from_source(model, sel_state, source, freq_min, ci_filter)
  rec = .mbspls_recompute_from_weights(model, W_use, log_env = log_env)

  obj = rec$obj_vec
  rn = paste0("LC_", sprintf("%02d", seq_along(obj)))
  df = data.frame(
    comp = factor(rn, levels = rn),
    obj  = as.numeric(obj),
    cum  = cumsum(as.numeric(obj))
  )
  ylab = if (cumulative) "Cumulative objective (MAC/Frobenius proxy)" else "Latent correlation (MAC proxy)"

  ggplot2::ggplot(df, ggplot2::aes(.data$comp, if (cumulative) .data$cum else .data$obj, group = 1)) +
    ggplot2::geom_line() + ggplot2::geom_point(size = 2) +
    ggplot2::labs(x = NULL, y = ylab,
      title = paste0("MB-sPLS scree (recomputed from ", source, ")", title_suffix)) +
    ggplot2::theme_minimal(base_size = 11, base_family = font)
}

# ------------------------------------------------------------------------------
# -------------------- SCORES (recomputed) -------------------------------------
# ------------------------------------------------------------------------------
.mbspls_plot_scores_general = function(
  model, sel_state = NULL, log_env = NULL,
  source = c("bootstrap", "weights"),
  freq_min = NULL,
  ci_filter = c("excludes_zero", "none", "overlaps_zero"),
  component = 1,
  standardize = TRUE,
  density = c("none", "contour", "hex"),
  annotate = TRUE,
  title_suffix = "",
  font = "sans"
) {
  requireNamespace("ggplot2")
  density = match.arg(density)
  source = match.arg(source)
  ci_filter = match.arg(ci_filter)

  W_use = .mbspls_weights_from_source(model, sel_state, source, freq_min, ci_filter)
  rec = .mbspls_recompute_from_weights(model, W_use, log_env = log_env)
  Tmat = rec$T_mat

  blocks = names(model$blocks)
  if (component < 1 || component > length(W_use)) stop("`component` out of range.")
  lv_cols = paste0("LV", component, "_", blocks)
  lv_cols = intersect(lv_cols, colnames(Tmat))
  if (length(lv_cols) < 2) stop("Need at least two block LVs for the selected component.")

  S = as.data.frame(Tmat[, lv_cols, drop = FALSE])
  names(S) = blocks
  if (isTRUE(standardize)) {
    S[] = lapply(S, function(v) {
      s = stats::sd(v)
      if (is.finite(s) && s > 0) (v - mean(v)) / s else v
    })
  }

  blks = names(model$blocks)
  pairs = utils::combn(blks, 2, simplify = FALSE)
  df = do.call(rbind, lapply(pairs, function(pr) {
    data.frame(x = S[[pr[1]]], y = S[[pr[2]]],
      block_x = pr[1], block_y = pr[2],
      stringsAsFactors = FALSE)
  }))

  ccc_fun = function(x, y) {
    mx = mean(x)
    my = mean(y)
    vx = stats::var(x)
    vy = stats::var(y)
    sxy = stats::cov(x, y)
    if (vx <= 0 || vy <= 0) {
      return(NA_real_)
    }
    (2 * sxy) / (vx + vy + (mx - my)^2)
  }
  panel_stats = do.call(rbind, lapply(pairs, function(pr) {
    xx = S[[pr[1]]]
    yy = S[[pr[2]]]
    ok = is.finite(xx) & is.finite(yy)
    xx = xx[ok]
    yy = yy[ok]
    n = length(xx)
    r = suppressWarnings(stats::cor(xx, yy, method = "pearson"))
    ccc = ccc_fun(xx, yy)
    fit = if (n >= 2) stats::lm(yy ~ xx) else NULL
    slope = if (!is.null(fit)) unname(stats::coef(fit)[2]) else NA_real_
    data.frame(block_x = pr[1], block_y = pr[2],
      n = n, r = r, ccc = ccc, slope = slope,
      stringsAsFactors = FALSE)
  }))

  p = ggplot2::ggplot(df, ggplot2::aes(.data$x, .data$y)) +
    ggplot2::geom_point(alpha = 0.55, size = 1.5)

  if (density == "contour") {
    p = p + ggplot2::stat_density_2d(ggplot2::aes(level = ggplot2::after_stat(level)), linewidth = 0.3)
  } else if (density == "hex") {
    if (!requireNamespace("hexbin", quietly = TRUE)) {
      stop("Package 'hexbin' is required for density = 'hex'.")
    }
    p = p + ggplot2::stat_bin_hex(bins = 20, alpha = 0.8)
  }

  p = p +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2, linewidth = 0.4) +
    ggplot2::geom_smooth(method = "lm", se = FALSE, linewidth = 0.5, formula = y ~ x)

  if (isTRUE(annotate)) {
    lab = within(panel_stats, {
      label = sprintf("r = %.2f\nccc = %.2f\nslope = %.2f\nn = %d", r, ccc, slope, n)
    })
    p = p + ggplot2::geom_text(
      data = lab, ggplot2::aes(x = -Inf, y = Inf, label = label),
      hjust = -0.05, vjust = 1.05, size = 3, inherit.aes = FALSE
    )
  }

  if (isTRUE(standardize)) {
    p +
      ggplot2::facet_grid(.data$block_y ~ .data$block_x, scales = "fixed") +
      ggplot2::coord_equal() +
      ggplot2::labs(
        x = sprintf("Scores: LV%d (block)", component),
        y = sprintf("Scores: LV%d (block)", component),
        title = sprintf("MB-sPLS cross-block score agreement - LC%d%s", component, title_suffix),
        subtitle = "Z-scored per block, dashed: y = x, solid: LS fit, r = Pearson, ccc = Lin's concordance"
      ) +
      ggplot2::theme_minimal(base_size = 11, base_family = font)
  } else {
    p +
      ggplot2::facet_grid(.data$block_y ~ .data$block_x, scales = "free") +
      ggplot2::labs(
        x = sprintf("Scores: LV%d (block)", component),
        y = sprintf("Scores: LV%d (block)", component),
        title = sprintf("MB-sPLS cross-block score agreement - LC%d%s", component, title_suffix),
        subtitle = "Dashed: y = x, solid: LS fit, r = Pearson, ccc = Lin's concordance"
      ) +
      ggplot2::theme_minimal(base_size = 11, base_family = font)
  }
}

# ------------------------------------------------------------------------------
# -------- WEIGHTS helpers (ci_filter-aware) -----------------------------------
# ------------------------------------------------------------------------------
.mbspls_df_from_fit = function(fit_state) {
  blocks = fit_state$blocks
  bn = names(blocks)

  comps = names(fit_state$weights)
  if (is.null(comps)) comps <- sprintf("LC_%02d", seq_len(fit_state$ncomp %||% 1L))

  dplyr::bind_rows(lapply(comps, function(cn) {
    blist = fit_state$weights[[cn]]
    dplyr::bind_rows(lapply(bn, function(b) {
      w = blist[[b]]
      feats = blocks[[b]]

      v = as.numeric(w)
      nm = .mbspls_feat_names(w, fallback = feats)
      if (is.null(nm)) nm <- paste0("V", seq_along(v))

      # If lengths mismatch but blocks match, prefer blocks
      if (!is.null(feats) && length(feats) == length(v)) nm <- feats
      if (length(nm) != length(v)) nm <- rep_len(nm, length.out = length(v))

      tibble::tibble(
        component = gsub("^LC_0?", "LC ", cn),
        block     = b,
        feature   = nm,
        mean      = v,
        ci_lower  = NA_real_,
        ci_upper  = NA_real_
      )
    }))
  }))
}

.mbspls_df_from_bootstrap_stable = function(sel_state, ci_filter = c("excludes_zero", "none", "overlaps_zero")) {
  ci_filter = match.arg(ci_filter)
  .canon = function(x) gsub("^LC_0?", "LC ", x)

  if (!is.null(sel_state$weights_stable) && length(sel_state$weights_stable)) {
    ws = sel_state$weights_stable
    comps = names(ws)

    df = dplyr::bind_rows(lapply(comps, function(cn) {
      blist = ws[[cn]]
      dplyr::bind_rows(lapply(names(blist), function(b) {
        w = blist[[b]]
        v = as.numeric(w)
        nm = .mbspls_feat_names(w)

        if (is.null(nm)) nm <- paste0("V", seq_along(v))
        if (length(nm) != length(v)) nm <- rep_len(nm, length.out = length(v))

        tibble::tibble(
          component_code = cn,
          component      = .canon(cn),
          block          = b,
          feature        = nm,
          mean           = v
        )
      }))
    })) |>
      dplyr::filter(is.finite(.data$mean) & .data$mean != 0)

    if (!is.null(sel_state$weights_ci) && nrow(sel_state$weights_ci)) {
      ci = as.data.frame(sel_state$weights_ci)
      df = df |>
        dplyr::left_join(ci[, c("component", "block", "feature", "ci_lower", "ci_upper")],
          by = c("component_code" = "component", "block", "feature"))
    }
    df$component_code = NULL
    return(df)
  }

  ci = as.data.frame(sel_state$weights_ci)
  if (is.null(ci) || !nrow(ci)) {
    stop("No bootstrap selection output found (weights_stable/weights_ci).")
  }

  df = ci |>
    dplyr::transmute(
      component = .canon(.data$component),
      block     = .data$block,
      feature   = .data$feature,
      mean      = .data$boot_mean,
      ci_lower  = .data$ci_lower,
      ci_upper  = .data$ci_upper
    )

  if (ci_filter == "excludes_zero") {
    df = df |>
      dplyr::filter((.data$ci_lower >= 0 | .data$ci_upper <= 0) & abs(.data$mean) > 1e-3)
  } else if (ci_filter == "overlaps_zero") {
    df = df |>
      dplyr::filter(.data$ci_lower <= 0 & .data$ci_upper >= 0)
  }
  df
}

.mbspls_stable_keys = function(sel_state, ci_filter = c("excludes_zero", "none", "overlaps_zero")) {
  ci_filter = match.arg(ci_filter)
  .canon = function(x) gsub("^LC_0?", "LC ", x)

  # Prefer explicit stable list if present
  if (!is.null(sel_state$weights_stable) && length(sel_state$weights_stable)) {
    ws = sel_state$weights_stable
    out = list()
    ii = 1L

    for (cn in names(ws)) {
      blist = ws[[cn]]
      for (b in names(blist)) {
        w = blist[[b]]
        if (is.null(w)) next
        v = as.numeric(w)
        nm = .mbspls_feat_names(w)
        if (is.null(nm)) nm <- paste0("V", seq_along(v))
        if (length(nm) != length(v)) nm <- rep_len(nm, length.out = length(v))

        keep = is.finite(v) & v != 0
        if (!any(keep)) next

        out[[ii]] = data.frame(
          component = .canon(cn),
          block = b,
          feature = nm[keep],
          stringsAsFactors = FALSE
        )
        ii = ii + 1L
      }
    }
    if (length(out)) {
      return(unique(do.call(rbind, out)))
    }
  }

  # Fallback: derive stability from weights_ci
  ci = as.data.frame(sel_state$weights_ci)
  if (is.null(ci) || !nrow(ci)) {
    stop("No bootstrap selection output found (weights_stable/weights_ci).")
  }

  keep = switch(ci_filter,
    excludes_zero = ((ci$ci_lower >= 0) | (ci$ci_upper <= 0)) & (abs(ci$boot_mean) > 1e-3),
    overlaps_zero = (ci$ci_lower <= 0 & ci$ci_upper >= 0),
    none          = rep(TRUE, nrow(ci))
  )

  keys = ci[keep, c("component", "block", "feature")]
  keys$component = .canon(keys$component)
  unique(keys)
}

.mbspls_plot_weights_single_component = function(df_sub, block_levels, title = "", font = "sans", alpha_by_stability = FALSE) {
  requireNamespace("ggplot2")
  requireNamespace("grid")

  if (!nrow(df_sub)) {
    return(ggplot2::ggplot() + ggplot2::theme_void() + ggplot2::ggtitle("No non-zero weights"))
  }

  block_pretty = gsub("_", " ", block_levels, fixed = TRUE)
  df_sub$block_lab = factor(gsub("_", " ", df_sub$block, fixed = TRUE), levels = block_pretty)

  wrap_fun = if (requireNamespace("stringr", quietly = TRUE)) {
    function(x) stringr::str_wrap(x, width = 28)
  } else {
    function(x) vapply(x, function(s) paste(strwrap(s, width = 28), collapse = "\n"), character(1))
  }

  df_sub = df_sub |>
    dplyr::mutate(abs_m = abs(.data$mean), signpos = .data$mean >= 0) |>
    dplyr::group_by(.data$block_lab) |>
    dplyr::arrange(.data$abs_m, .by_group = TRUE) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      feat_lab = wrap_fun(.data$feature),
      axis_id  = paste(.data$block_lab, .data$feature, sep = "___")
    )

  df_sub$axis_id = factor(df_sub$axis_id, levels = df_sub$axis_id)
  lab_map = stats::setNames(df_sub$feat_lab, df_sub$axis_id)

  has_ci = all(c("ci_lower", "ci_upper") %in% names(df_sub)) &&
    any(is.finite(df_sub$ci_lower) | is.finite(df_sub$ci_upper))

  # ---- UPDATED: use alpha_plot if provided
  if (isTRUE(alpha_by_stability) && "alpha_plot" %in% names(df_sub)) {
    g = ggplot2::ggplot(df_sub, ggplot2::aes(x = axis_id, y = mean, fill = signpos)) +
      ggplot2::geom_col(ggplot2::aes(alpha = alpha_plot), width = 0.85, show.legend = FALSE) +
      ggplot2::scale_alpha_identity()
  } else {
    g = ggplot2::ggplot(df_sub, ggplot2::aes(x = axis_id, y = mean, fill = signpos)) +
      ggplot2::geom_col(width = 0.85, show.legend = FALSE)
  }

  if (has_ci) {
    g = g + ggplot2::geom_errorbar(
      ggplot2::aes(ymin = .data$ci_lower, ymax = .data$ci_upper),
      width = 0.20, linewidth = 0.25, colour = "grey15"
    )
  }

  g +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey70") +
    ggplot2::facet_grid(rows = ggplot2::vars(block_lab),
      scales = "free_y", space = "free_y", switch = "y") +
    ggplot2::scale_fill_manual(values = .mbspls_pal()) +
    ggplot2::scale_x_discrete(labels = function(x) lab_map[as.character(x)]) +
    ggplot2::coord_flip(clip = "off") +
    ggplot2::labs(x = NULL, y = "Weight", title = title) +
    ggplot2::theme_minimal(base_family = font) +
    ggplot2::theme(
      panel.border       = ggplot2::element_rect(colour = "grey80", fill = NA, linewidth = 0.5),
      panel.spacing      = grid::unit(0.6, "lines"),
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.minor   = ggplot2::element_blank(),
      strip.placement    = "outside",
      strip.background   = ggplot2::element_rect(fill = NA, colour = NA),
      strip.text.y.left  = ggplot2::element_text(angle = 0, face = "bold")
    )
}

.mbspls_plot_weights_patchwork = function(
  fit_state,
  sel_state = NULL,
  source = c("weights", "bootstrap"),
  top_n = NULL,
  patch_ncol = 1L,
  font = "sans",
  alpha_by_stability = FALSE,
  alpha_nonstable = 0.4,
  freq_min = NULL,
  ci_filter = c("excludes_zero", "none", "overlaps_zero")
) {
  requireNamespace("patchwork")
  requireNamespace("ggplot2")
  requireNamespace("dplyr")

  source = match.arg(source)
  ci_filter = match.arg(ci_filter)

  blocks = fit_state$blocks
  block_levels = names(blocks)

  if (identical(source, "weights")) {
    if (!is.null(freq_min)) message("freq_min is ignored for source='weights'.")
    df = .mbspls_df_from_fit(fit_state)
    df = df[is.finite(df$mean) & df$mean != 0, , drop = FALSE]

    # ---- NEW: bootstrap stability overlay on TRAINING weights
    if (isTRUE(alpha_by_stability)) {
      if (is.null(sel_state)) {
        stop("alpha_by_stability=TRUE with source='weights' requires `sel_state` (bootstrap selection output).")
      }
      keys = .mbspls_stable_keys(sel_state, ci_filter = ci_filter)
      keys$stable = TRUE

      df = merge(df, keys, by = c("component", "block", "feature"), all.x = TRUE)
      df$stable[is.na(df$stable)] = FALSE
      df$alpha_plot = ifelse(df$stable, 1, as.numeric(alpha_nonstable))
    }

  } else {
    if (is.null(sel_state)) stop("Selection state missing for source='bootstrap'.")

    if (is.null(freq_min)) {
      df = .mbspls_df_from_bootstrap_stable(sel_state, ci_filter = ci_filter)
    } else {
      ci = as.data.frame(sel_state$weights_ci)
      fr = as.data.frame(sel_state$weights_selectfreq)
      if (is.null(ci) || !nrow(ci) || is.null(fr) || !nrow(fr)) {
        stop("Need both weights_ci and weights_selectfreq for freq_min filtering.")
      }
      ci$component = gsub("^LC_0?", "LC ", ci$component)
      fr$component = gsub("^LC_0?", "LC ", fr$component)

      df = ci |>
        dplyr::transmute(
          component = .data$component,
          block     = .data$block,
          feature   = .data$feature,
          mean      = .data$boot_mean
        )
      df = merge(df, fr[, c("component", "block", "feature", "freq")], all.x = TRUE)
      df = df[is.finite(df$freq) & df$freq >= as.numeric(freq_min), , drop = FALSE]
    }
    df = df[is.finite(df$mean) & df$mean != 0, , drop = FALSE]
  }

  if (!nrow(df)) stop("No weights to plot.")

  if (!is.null(top_n) && is.numeric(top_n) && top_n > 0) {
    df = df |>
      dplyr::mutate(abs_m = abs(.data$mean)) |>
      dplyr::group_by(.data$component, .data$block) |>
      dplyr::slice_max(.data$abs_m, n = top_n, with_ties = FALSE) |>
      dplyr::ungroup()
  }

  comps = unique(df$component)
  plots = lapply(comps, function(cc) {
    df_sub = df[df$component == cc, , drop = FALSE]
    ttl = if (identical(source, "weights")) sprintf("MB-sPLS training weights - %s", cc) else cc
    .mbspls_plot_weights_single_component(
      df_sub, block_levels,
      title = ttl, font = font,
      alpha_by_stability = alpha_by_stability
    )
  })

  patch_ncol = min(patch_ncol, max(1L, length(plots)))
  p = Reduce(`+`, plots) + patchwork::plot_layout(ncol = patch_ncol, guides = "collect")

  main_ttl = if (identical(source, "weights")) {
    if (isTRUE(alpha_by_stability)) {
      sprintf("MB-sPLS training weights per block (opaque = bootstrap-stable; alpha = %.0f%% for non-stable)", alpha_nonstable * 100)
    } else {
      "MB-sPLS training weights per block"
    }
  } else if (is.null(freq_min)) {
    "MB-sPLS bootstrap STABLE weights per block"
  } else {
    sprintf("MB-sPLS bootstrap MEANS (freq \u2265 %.2f)", as.numeric(freq_min))
  }

  p + patchwork::plot_annotation(title = main_ttl)
}

# ------------------------------------------------------------------------------
# ------------------- Bootstrap component (unchanged) --------------------------
# ------------------------------------------------------------------------------
.mbspls_plot_bootstrap_component = function(
  model,
  payload = NULL,
  mbspls_id = NULL,
  show_violin = TRUE,
  show_box = TRUE,
  show_ci = TRUE,
  show_observed = TRUE,
  show_pvalue = TRUE,
  violin_alpha = 0.25,
  box_width = 0.15,
  point_size = 2.6,
  errorbar_width = 0.22,
  font = "sans"
) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for bootstrap plots.")
  }

  pay = .mbspls_get_eval_payload(model, payload, mbspls_id)

  bt = pay$val_bootstrap
  if (is.null(bt)) {
    stop("No component-level bootstrap in payload (val_bootstrap is NULL). ",
      "Enable validation with val_test='bootstrap' for predict().")
  }

  bt = as.data.frame(bt, stringsAsFactors = FALSE)
  req = c("component", "observed_correlation", "boot_mean", "boot_se",
    "boot_p_value", "boot_ci_lower", "boot_ci_upper",
    "confidence_level", "n_boot")
  miss = setdiff(req, names(bt))
  if (length(miss)) stop("val_bootstrap missing columns: ", paste(miss, collapse = ", "))

  comp_lab = paste0("LC_", sprintf("%02d", bt$component))
  df = data.frame(
    component = factor(comp_lab, levels = comp_lab),
    mean = bt$boot_mean,
    lwr = bt$boot_ci_lower,
    upr = bt$boot_ci_upper,
    obs = bt$observed_correlation,
    pval = bt$boot_p_value,
    conf = bt$confidence_level[1],
    nboot = bt$n_boot[1]
  )

  perf = pay$perf_metric %||% "MAC"

  p = ggplot2::ggplot(df, ggplot2::aes(.data$component, .data$mean))

  has_samples = !is.null(pay$val_boot_vectors)
  if (isTRUE(show_violin) && has_samples) {
    boot_long = do.call(rbind, lapply(seq_along(pay$val_boot_vectors), function(i) {
      data.frame(component = comp_lab[i], boot = as.numeric(pay$val_boot_vectors[[i]]))
    }))
    boot_long$component = factor(boot_long$component, levels = comp_lab)
    p = p +
      ggplot2::geom_violin(data = boot_long,
        ggplot2::aes(y = .data$boot, x = .data$component),
        fill = "grey40", alpha = violin_alpha, colour = NA, width = 0.8,
        inherit.aes = FALSE)
  } else if (isTRUE(show_violin)) {
    show_violin = FALSE
  }

  if (isTRUE(show_box) && has_samples) {
    p = p + ggplot2::geom_boxplot(width = box_width, outlier.shape = NA, fill = NA)
  }

  if (isTRUE(show_ci)) {
    p = p +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = .data$lwr, ymax = .data$upr),
        width = errorbar_width) +
      ggplot2::geom_point(size = point_size)
  } else {
    p = p + ggplot2::geom_point(size = point_size)
  }

  if (isTRUE(show_observed)) {
    p = p + ggplot2::geom_point(ggplot2::aes(y = .data$obs),
      shape = 4, colour = "red",
      size = point_size + 2, stroke = 1.2)
  }

  if (isTRUE(show_pvalue)) {
    df$y_lab = df$upr
    df$lab = sprintf("p = %s",
      ifelse(is.finite(df$pval),
        formatC(df$pval, format = "f", digits = 3), "NA"))
    p = p + ggplot2::geom_text(
      data = df,
      ggplot2::aes(x = .data$component, y = .data$y_lab, label = .data$lab),
      inherit.aes = FALSE,
      vjust = -0.8, size = 3.2
    )
  }

  p +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.02, 0.18))) +
    ggplot2::labs(
      title = "Bootstrap validation (component-wise)",
      subtitle = sprintf("Statistic: %s - %.0f%% CI - n_boot = %d",
        tolower(perf), df$conf[1] * 100, df$nboot[1]),
      x = NULL,
      y = "Latent correlation (MAC/Frobenius)",
      caption = "Red cross = observed correlation on original test data; filled dot = bootstrap mean"
    ) +
    ggplot2::theme_minimal(base_size = 11, base_family = font)
}

# ------------------------------------------------------------------------------
# ------------------- Payload reader (unchanged) -------------------------------
# ------------------------------------------------------------------------------
.mbspls_get_eval_payload = function(model = NULL, payload = NULL, mbspls_id = NULL) {
  if (!is.null(payload)) {
    return(payload)
  }

  if (inherits(model, "GraphLearner")) {
    ids = names(model$graph$pipeops)
    if (is.null(mbspls_id)) {
      cand = ids[vapply(model$graph$pipeops, inherits, logical(1), "PipeOpMBsPLS")]
      if (!length(cand)) {
        stop("No PipeOpMBsPLS node found in the graph.")
      }
      mbspls_id = cand[1]
    }
    po = model$graph$pipeops[[mbspls_id]]
    env = po$param_set$values$log_env
    if (inherits(env, "environment") && !is.null(env$last)) {
      return(env$last)
    }
  }

  if (!is.null(model$log_env) && !is.null(model$log_env$last)) {
    return(model$log_env$last)
  }
  if (!is.null(model$last)) {
    return(model$last)
  }

  stop("No evaluation payload found. Call predict() with a log_env set (or pass payload= explicitly).")
}
