#' Autoplot for GraphLearner with MB-sPLS (generalized, branch-aware)
#'
#' See usage notes in the body text. Requires: ggplot2, dplyr, tibble.
#' Optional: patchwork, ggraph, igraph, RColorBrewer, scales, hexbin.
#'
#' @importFrom ggplot2 autoplot
#' @export
#' @method autoplot GraphLearner
autoplot.GraphLearner = function(object,
  type = c("mbspls_weights", "mbspls_heatmap", "mbspls_network",
    "mbspls_variance", "mbspls_scree", "mbspls_scores",
    "mbspls_bootstrap_component", "mbspls_bootstrap_comp"),
  ...) {

  type = match.arg(type)
  dots = list(...)

  # -- weights path with freq_min support for *bootstrap means* ----------------
  if (type == "mbspls_weights") {
    source = (dots$source %||% "bootstrap") # "weights" | "bootstrap"
    select_id = dots$select_id %||% "mbspls_bootstrap_select"
    top_n = dots$top_n %||% NULL
    patch_ncol = as.integer(dots$patch_ncol %||% 3L)
    font = dots$font %||% "sans"
    alpha_by_stability = dots$alpha_by_stability %||% FALSE
    freq_min = dots$freq_min %||% NULL

    # locate nodes; only require selection node when source == "bootstrap"
    nodes = .mbspls_locate_nodes_general(object,
      select_id = if (identical(source, "bootstrap")) select_id else NULL)

    return(
      .mbspls_plot_weights_patchwork(
        fit_state          = nodes$fit_state,
        sel_state          = nodes$sel_state, # may be NULL for raw weights
        source             = source,
        top_n              = top_n,
        patch_ncol         = patch_ncol,
        font               = font,
        alpha_by_stability = alpha_by_stability,
        freq_min           = freq_min
      )
    )
  }

  # -- helper: keep only arguments a function actually accepts ------------------
  .keep_formals = function(d, fun) {
    if (!length(d)) {
      return(d)
    }
    fml = names(formals(fun))
    if (is.null(fml)) {
      return(d)
    }
    d[names(d) %in% fml]
  }

  # -- locate MB-sPLS node once -------------------------------------------------
  mod_nodes = .mbspls_locate_nodes_general(object, select_id = NULL)
  model_for_others = mod_nodes$fit_state

  # NEW: optional validation task (val_task)
  val_task = dots$val_task %||% NULL
  mbspls_id = dots$mbspls_id %||% NULL
  title_suffix = if (!is.null(dots$val_task)) " (validation task)" else ""

  .mbspls_eval_newdata_dispatch = function(gl, task, mbspls_id = NULL) {
    fn = get0("mbspls_eval_new_data", mode = "function") %||%
      get0("mbspls_eval_on_new", mode = "function")
    if (is.null(fn)) stop("Need mbspls_eval_new_data() or mbspls_eval_on_new().")
    fn(gl, task, mbspls_id)
  }

  needs_eval = !is.null(val_task) && type %in% c("mbspls_variance", "mbspls_heatmap", "mbspls_network")
  eval_payload = NULL
  if (needs_eval) {
    eval_payload = .mbspls_eval_newdata_dispatch(object, val_task, mbspls_id)
  }
  dots$val_task = NULL

  switch(
    type,
    mbspls_heatmap = {
      fun = .mbspls_plot_heatmap_from_model
      do.call(fun, c(list(
        model = model_for_others,
        T_override = if (!is.null(eval_payload)) eval_payload$T_mat else NULL,
        title_suffix = title_suffix
      ), .keep_formals(dots, fun)))
    },
    mbspls_network = {
      fun = .mbspls_plot_network_from_model
      do.call(fun, c(list(
        model = model_for_others,
        T_override = if (!is.null(eval_payload)) eval_payload$T_mat else NULL,
        title_suffix = title_suffix
      ), .keep_formals(dots, fun)))
    },
    mbspls_variance = {
      fun = .mbspls_plot_variance_from_model
      do.call(fun, c(list(
        model = model_for_others,
        ev_block_override = if (!is.null(eval_payload)) eval_payload$ev_block else NULL,
        ev_comp_override = if (!is.null(eval_payload)) eval_payload$ev_comp else NULL,
        title_suffix = title_suffix
      ), .keep_formals(dots, fun)))
    },
    mbspls_scree = {
      fun = .mbspls_plot_scree_from_model
      do.call(fun, c(list(model = model_for_others,
        obj_override = NULL,
        title_suffix = title_suffix),
      .keep_formals(dots, fun)))
    },
    mbspls_scores = {
      fun = .mbspls_plot_scores_from_model
      do.call(fun, c(list(model = model_for_others,
        T_override = NULL,
        title_suffix = title_suffix),
      .keep_formals(dots, fun)))
    },
    mbspls_bootstrap_component = {
      fun = .mbspls_plot_bootstrap_component
      do.call(fun, c(list(model = object,
        payload = NULL,
        mbspls_id = mbspls_id),
      .keep_formals(dots, fun)))
    },
    mbspls_bootstrap_comp = {
      fun = .mbspls_plot_bootstrap_component
      do.call(fun, c(list(model = object,
        payload = NULL,
        mbspls_id = mbspls_id),
      .keep_formals(dots, fun)))
    }
  )
}

# ---------- helpers (generalized node locator, data builders, plotting) -------
.mbspls_pal = function() {
  if (!requireNamespace("RColorBrewer", quietly = TRUE)) {
    return(c(`TRUE` = "#1b9e77", `FALSE` = "#d95f02"))
  }
  pal = RColorBrewer::brewer.pal(3, "Dark2")
  c(`TRUE` = pal[1], `FALSE` = pal[3])
}

# Back-compat shim for legacy code that still calls .locate_mbspls_model()
.locate_mbspls_model = function(gl) {
  .mbspls_locate_nodes_general(gl)$fit_state
}

# Find MB-sPLS (fit) and optional bootstrap-select node by id or by class scan
.mbspls_locate_nodes_general = function(gl, mbspls_id = NULL, select_id = NULL) {
  if (!inherits(gl, "GraphLearner")) {
    stop("Expected a GraphLearner.", call. = FALSE)
  }
  mod = gl$model
  if (is.null(mod)) {
    stop("GraphLearner appears to be untrained (model is NULL).", call. = FALSE)
  }

  # --- MB-sPLS fit node (first PipeOpMBsPLS if id not given)
  if (is.null(mbspls_id)) {
    cand = names(gl$graph$pipeops)[vapply(gl$graph$pipeops, inherits, logical(1), "PipeOpMBsPLS")]
    if (!length(cand)) stop("No PipeOpMBsPLS node found in the graph.")
    mbspls_id = cand[1]
  }
  fit = mod[[mbspls_id]] %||% gl$graph$pipeops[[mbspls_id]]
  if (is.null(fit)) stop("Cannot locate MB-sPLS node '", mbspls_id, "' in model/graph.")
  fit_state = if (!is.null(fit$state)) fit$state else fit

  # --- Selection node (optional)
  sel_state = NULL
  if (!is.null(select_id)) {
    sel = mod[[select_id]] %||% gl$graph$pipeops[[select_id]]
    if (is.null(sel)) stop("Cannot locate selection node '", select_id, "'.")
    sel_state = if (!is.null(sel$state)) sel$state else sel
  } else {
    # try canonical id; else scan for any PipeOpMBsPLSBootstrapSelect
    sel = mod$mbspls_bootstrap_select %||% gl$graph$pipeops$mbspls_bootstrap_select
    if (is.null(sel)) {
      hits = names(gl$graph$pipeops)[vapply(gl$graph$pipeops, inherits, logical(1), "PipeOpMBsPLSBootstrapSelect")]
      if (length(hits)) {
        sel = mod[[hits[1]]] %||% gl$graph$pipeops[[hits[1]]]
        message("Using selection node '", hits[1], "' (pass select_id= to pick another branch).")
      }
    }
    if (!is.null(sel)) sel_state <- if (!is.null(sel$state)) sel$state else sel
  }

  list(fit_state = fit_state, sel_state = sel_state)
}

# -------- data builders for weights -------------------------------------------

.mbspls_df_from_fit = function(fit_state) {
  blocks = fit_state$blocks
  comps = names(fit_state$weights)
  if (is.null(comps)) comps <- sprintf("LC_%02d", seq_len(fit_state$ncomp %||% 1L))

  dplyr::bind_rows(lapply(comps, function(cn) {
    blist = fit_state$weights[[cn]]
    dplyr::bind_rows(lapply(names(blist), function(b) {
      w = blist[[b]]
      nm = names(w)
      if (is.null(nm)) nm <- rep.int("", length(w))
      tibble::tibble(
        component = gsub("^LC_0?", "LC ", cn),
        block     = b,
        feature   = nm,
        mean      = as.numeric(w),
        ci_lower  = NA_real_,
        ci_upper  = NA_real_
      )
    }))
  }))
}

.mbspls_df_from_bootstrap_ci = function(sel_state) {
  .canon = function(x) gsub("^LC_0?", "LC ", x)

  ci = as.data.frame(sel_state$weights_ci)
  if (is.null(ci) || !nrow(ci)) {
    stop("No 'weights_ci' found in selection state (need bootstrap_select with summaries).")
  }

  # Expect columns: component, block, feature, boot_mean, ci_lower, ci_upper
  need = c("component", "block", "feature", "boot_mean", "ci_lower", "ci_upper")
  miss = setdiff(need, names(ci))
  if (length(miss)) {
    stop("weights_ci missing columns: ", paste(miss, collapse = ", "))
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
  df
}

.mbspls_df_from_bootstrap_stable = function(sel_state) {
  .canon = function(x) gsub("^LC_0?", "LC ", x)

  if (!is.null(sel_state$weights_stable) && length(sel_state$weights_stable)) {
    ws = sel_state$weights_stable
    comps = names(ws)
    df = dplyr::bind_rows(lapply(comps, function(cn) {
      blist = ws[[cn]]
      dplyr::bind_rows(lapply(names(blist), function(b) {
        v = as.numeric(blist[[b]])
        nm = names(blist[[b]])
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

    # Optional: join CI bounds if present
    if (!is.null(sel_state$weights_ci) && nrow(sel_state$weights_ci)) {
      ci = as.data.frame(sel_state$weights_ci)
      df = df |>
        dplyr::left_join(ci[, c("component", "block", "feature", "ci_lower", "ci_upper")],
          by = c("component_code" = "component", "block", "feature"))
    }
    df$component_code = NULL
    return(df)
  }

  # Fallback to CI table (apply excludes-zero & |mean|>1e-3)
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
    ) |>
    dplyr::filter((.data$ci_lower >= 0 | .data$ci_upper <= 0) & abs(.data$mean) > 1e-3)

  df
}

# -------- weights plot (single + patchwork) -----------------------------------

.mbspls_plot_weights_single_component = function(df_sub, block_levels, title = "", font = "sans", alpha_by_stability = FALSE) {
  if (!nrow(df_sub)) {
    return(ggplot2::ggplot() + ggplot2::theme_void() + ggplot2::ggtitle("No non-zero weights"))
  }

  block_pretty = gsub("_", " ", block_levels, fixed = TRUE)
  df_sub$block_lab = factor(gsub("_", " ", df_sub$block, fixed = TRUE), levels = block_pretty)

  df_sub = df_sub |>
    dplyr::mutate(abs_m = abs(.data$mean), signpos = .data$mean >= 0) |>
    dplyr::group_by(.data$block_lab) |>
    dplyr::arrange(.data$abs_m, .by_group = TRUE) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      feat_lab = stringr::str_wrap(.data$feature, width = 28),
      axis_id  = paste(.data$block_lab, .data$feature, sep = "___")
    )

  df_sub$axis_id = factor(df_sub$axis_id, levels = df_sub$axis_id)
  lab_map = stats::setNames(df_sub$feat_lab, df_sub$axis_id)

  has_ci = all(c("ci_lower", "ci_upper") %in% names(df_sub)) &&
    any(is.finite(df_sub$ci_lower) | is.finite(df_sub$ci_upper))

  # Optional alpha mapping when stability frequencies were merged upstream.
  if (isTRUE(alpha_by_stability) && "alpha_freq" %in% names(df_sub)) {
    g = ggplot2::ggplot(df_sub, ggplot2::aes(x = axis_id, y = mean, fill = signpos)) +
      ggplot2::geom_col(ggplot2::aes(alpha = alpha_freq), width = 0.85, show.legend = FALSE) +
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
  freq_min = NULL # <- only meaningful for source == "bootstrap"
) {
  requireNamespace("patchwork")
  source = match.arg(source)

  blocks = fit_state$blocks
  block_levels = names(blocks)

  if (source == "weights") {
    # RAW weights path: unchanged (no freq filter here)
    if (!is.null(freq_min)) {
      message("freq_min is ignored for source='weights'; it only applies to bootstrap means.")
    }
    df = .mbspls_df_from_fit(fit_state)
    df = df[is.finite(df$mean) & df$mean != 0, , drop = FALSE]

    # Optionally map alpha from stability if selection state present
    if (isTRUE(alpha_by_stability) && !is.null(sel_state) && !is.null(sel_state$weights_selectfreq)) {
      freq_tbl = try(as.data.frame(sel_state$weights_selectfreq), silent = TRUE)
      if (!inherits(freq_tbl, "try-error") && nrow(freq_tbl)) {
        need = c("component", "block", "feature", "freq")
        if (all(need %in% names(freq_tbl))) {
          freq_tbl$component = gsub("^LC_0?", "LC ", freq_tbl$component)
          df$component = gsub("^LC_0?", "LC ", df$component)
          df = merge(df, freq_tbl[, need], by = c("component", "block", "feature"), all.x = TRUE)
          if (!any(is.finite(df$freq))) {
            freq_bf = aggregate(freq ~ block + feature, data = freq_tbl, FUN = function(z) {
              z = z[is.finite(z)]
              if (!length(z)) NA_real_ else max(z)
            })
            df = merge(df[, setdiff(names(df), "freq"), drop = FALSE],
              freq_bf, by = c("block", "feature"), all.x = TRUE)
          }
          df$alpha_freq = df$freq
          df$alpha_freq[!is.finite(df$alpha_freq)] = 1
          df$alpha_freq = pmin(pmax(df$alpha_freq, 0), 1)
        }
      }
    }

  } else { # source == "bootstrap"
    if (is.null(sel_state)) {
      stop("Bootstrap selection state not found; pass a valid select_id and train with bootstrap selection.")
    }

    if (!is.null(freq_min)) {
      # ---- NEW: frequency filter on UNFILTERED bootstrap means (weights_ci) ----
      df = .mbspls_df_from_bootstrap_ci(sel_state)
      # Join selection frequencies
      freq_tbl = try(as.data.frame(sel_state$weights_selectfreq), silent = TRUE)
      if (inherits(freq_tbl, "try-error") || !nrow(freq_tbl)) {
        stop("freq_min provided, but weights_selectfreq is missing from selection state.")
      }
      need = c("component", "block", "feature", "freq")
      if (!all(need %in% names(freq_tbl))) {
        stop("weights_selectfreq must have columns: component, block, feature, freq.")
      }
      freq_tbl$component = gsub("^LC_0?", "LC ", freq_tbl$component)
      df$component = gsub("^LC_0?", "LC ", df$component)

      # primary join: component+block+feature
      df = merge(df, freq_tbl[, need], by = c("component", "block", "feature"), all.x = TRUE)
      # fallback to block+feature if all NA (edge cases)
      if (!any(is.finite(df$freq))) {
        freq_bf = aggregate(freq ~ block + feature, data = freq_tbl, FUN = function(z) {
          z = z[is.finite(z)]
          if (!length(z)) NA_real_ else max(z)
        })
        df = merge(df[, setdiff(names(df), "freq"), drop = FALSE],
          freq_bf, by = c("block", "feature"), all.x = TRUE)
      }

      # keep only features with freq >= freq_min
      df = df[is.finite(df$freq) & df$freq >= as.numeric(freq_min), , drop = FALSE]
      if (!nrow(df)) stop(sprintf("No bootstrap-mean weights pass freq_min = %.3f.", as.numeric(freq_min)))

      # Optional alpha mapping
      if (isTRUE(alpha_by_stability)) {
        df$alpha_freq = df$freq
        df$alpha_freq[!is.finite(df$alpha_freq)] = 1
        df$alpha_freq = pmin(pmax(df$alpha_freq, 0), 1)
      }

      # finally, drop strict zeros (avoid empty rows)
      df = df[is.finite(df$mean) & df$mean != 0, , drop = FALSE]

    } else {
      # default: STABLE weights (previous behavior)
      df = .mbspls_df_from_bootstrap_stable(sel_state)
      df = df[is.finite(df$mean) & df$mean != 0, , drop = FALSE]

      # Optional alpha mapping from frequency (purely visual)
      if (isTRUE(alpha_by_stability) && !is.null(sel_state$weights_selectfreq)) {
        freq_tbl = try(as.data.frame(sel_state$weights_selectfreq), silent = TRUE)
        if (!inherits(freq_tbl, "try-error") && nrow(freq_tbl)) {
          need = c("component", "block", "feature", "freq")
          if (all(need %in% names(freq_tbl))) {
            freq_tbl$component = gsub("^LC_0?", "LC ", freq_tbl$component)
            df$component = gsub("^LC_0?", "LC ", df$component)
            df = merge(df, freq_tbl[, need], by = c("component", "block", "feature"), all.x = TRUE)
            if (!any(is.finite(df$freq))) {
              freq_bf = aggregate(freq ~ block + feature, data = freq_tbl, FUN = function(z) {
                z = z[is.finite(z)]
                if (!length(z)) NA_real_ else max(z)
              })
              df = merge(df[, setdiff(names(df), "freq"), drop = FALSE],
                freq_bf, by = c("block", "feature"), all.x = TRUE)
            }
            df$alpha_freq = df$freq
            df$alpha_freq[!is.finite(df$alpha_freq)] = 1
            df$alpha_freq = pmin(pmax(df$alpha_freq, 0), 1)
          }
        }
      }
    }
  }

  if (!nrow(df)) stop("No weights to plot.")

  # top N per block×component (after any filtering)
  if (!is.null(top_n) && is.numeric(top_n) && top_n > 0) {
    df = df |>
      dplyr::mutate(abs_m = abs(.data$mean)) |>
      dplyr::group_by(.data$component, .data$block) |>
      dplyr::slice_max(.data$abs_m, n = top_n, with_ties = FALSE) |>
      dplyr::ungroup()
  }

  # component order "LC 1", "LC 2", ...
  lc_order = sprintf("LC %d", seq_len(99))
  comp_levels = intersect(lc_order, unique(df$component))
  if (!length(comp_levels)) comp_levels <- unique(df$component)

  plots = lapply(comp_levels, function(cc) {
    df_sub = df[df$component == cc, , drop = FALSE]
    title = if (identical(source, "weights")) {
      sprintf("MB-sPLS raw weights - %s", cc)
    } else if (!is.null(freq_min)) {
      sprintf("%s (bootstrap means; freq >= %.2f)", cc, as.numeric(freq_min))
    } else {
      sprintf("%s", cc)
    }
    .mbspls_plot_weights_single_component(
      df_sub, block_levels, title = title, font = font,
      alpha_by_stability = alpha_by_stability
    )
  })

  patch_ncol = min(patch_ncol, length(plots))
  p = Reduce(`+`, plots) + patchwork::plot_layout(ncol = patch_ncol, guides = "collect")

  ttl = if (identical(source, "weights")) {
    "MB-sPLS raw weights per block"
  } else if (!is.null(freq_min)) {
    "MB-sPLS bootstrap MEANS per block (freq-filtered)"
  } else {
    "MB-sPLS bootstrap STABLE weights per block"
  }
  p + patchwork::plot_annotation(title = ttl)
}

# -------- HEATMAP (new, so your "mbspls_heatmap" type works) -------------------

.mbspls_plot_heatmap_from_model = function(
  model,
  method = c("spearman", "pearson"),
  absolute = TRUE,
  cluster = FALSE,
  label = TRUE,
  digits = 2,
  T_override = NULL,
  title_suffix = "",
  font = "sans"
) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for this plot.")
  }
  method = match.arg(method)

  scores = T_override %||% model$T_mat
  if (is.null(scores)) {
    stop("No latent score matrix (T_mat) available; train or pass T_override.", call. = FALSE)
  }

  # Ensure LV naming "LV##_<block>" if missing
  if (is.null(colnames(scores))) {
    colnames(scores) = unlist(lapply(seq_len(model$ncomp), function(k) {
      paste0("LV", sprintf("%02d", k), "_", names(model$blocks))
    }))
  }

  blks = names(model$blocks)
  ncomp = as.integer(model$ncomp %||% 1L)

  # Helper: pull columns for LV k (accept both LV1_ and LV01_ styles)
  cols_for_k = function(k) {
    patt = paste0("^LV0*", k, "_")
    cn = grep(patt, colnames(scores), value = TRUE)
    # If still nothing, try "LVk" without underscore (rare)
    if (!length(cn)) cn <- grep(paste0("^LV0*", k), colnames(scores), value = TRUE)
    cn
  }

  # Build long heatmap data for each component
  dfl = lapply(seq_len(ncomp), function(k) {
    cn = cols_for_k(k)
    # Keep only columns that correspond to known blocks (suffix after first "_")
    blk_of = function(x) sub("^.*?_", "", x)
    keep = blk_of(cn) %in% blks
    cn = cn[keep]
    if (length(cn) < 2) {
      return(NULL)
    }

    S = as.data.frame(scores[, cn, drop = FALSE])
    # Name columns by block only (easier plotting)
    colnames(S) = blk_of(cn)

    # Robust correlation (pairwise complete)
    C = suppressWarnings(stats::cor(S, method = method, use = "pairwise.complete.obs"))
    if (absolute) C <- abs(C)

    # Order blocks for display
    ord_blocks = blks
    if (isTRUE(cluster) && ncol(C) > 1) {
      # cluster by 1 - |r|
      D = try(as.dist(1 - abs(C)), silent = TRUE)
      if (!inherits(D, "try-error")) {
        ord = try(stats::hclust(D)$order, silent = TRUE)
        if (!inherits(ord, "try-error")) ord_blocks <- colnames(C)[ord]
      }
    }

    C = C[ord_blocks, ord_blocks, drop = FALSE]
    df = as.data.frame(as.table(C), stringsAsFactors = FALSE)
    colnames(df) = c("block_x", "block_y", "r")

    df$component = sprintf("LC %d", k)
    df
  })

  dfl = Filter(Negate(is.null), dfl)
  if (!length(dfl)) {
    stop("Need at least two LV block-columns for one component to draw a heatmap.", call. = FALSE)
  }

  df = do.call(rbind, dfl)
  df$block_x = factor(df$block_x, levels = blks)
  df$block_y = factor(df$block_y, levels = blks)

  p = ggplot2::ggplot(df, ggplot2::aes(block_x, block_y, fill = r)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.15) +
    {
      if (isTRUE(label)) {
        ggplot2::geom_text(ggplot2::aes(label = formatC(r, format = "f", digits = digits)), size = 3)
      } else {
        ggplot2::geom_blank()
      }
    } +
    {
      if (absolute) {
        ggplot2::scale_fill_viridis_c(limits = c(0, 1))
      } else {
        ggplot2::scale_fill_gradient2(low = scales::muted("blue", l = 30, c = 80), high = scales::muted("red", l = 50, c = 100), limits = c(-1, 1), midpoint = 0)
      }
      # else ggplot2::scale_fill_gradient2(low = "blue", high = "red", limits = c(-1, 1), midpoint = 0)
    } +
    ggplot2::facet_wrap(~component) +
    ggplot2::coord_equal() +
    ggplot2::labs(
      x = NULL, y = NULL, fill = if (absolute) "|r|" else "r",
      title = paste0("Cross-block LV correlation heatmap", title_suffix),
      subtitle = sprintf("Method: %s; one matrix per component", method)
    ) +
    ggplot2::theme_minimal(base_size = 11, base_family = font) +
    ggplot2::theme(
      panel.spacing = grid::unit(0.7, "lines"),
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
    )

  p
}

# -------- Network --------------------------------------------------------------

.mbspls_plot_network_from_model = function(model, cutoff = 0.3, method = "spearman",
  T_override = NULL, title_suffix = "", font = "sans") {
  requireNamespace("igraph")
  requireNamespace("ggraph")
  requireNamespace("ggplot2")

  scores = T_override %||% model$T_mat
  if (is.null(scores) || NCOL(scores) < 2) {
    stop("Need at least two latent variables to draw a network.", call. = FALSE)
  }

  if (is.null(colnames(scores))) {
    colnames(scores) = unlist(lapply(seq_len(model$ncomp), function(k) {
      paste0("LV", sprintf("%02d", k), "_", names(model$blocks))
    }))
  }

  C = stats::cor(scores, method = method, use = "pairwise.complete.obs")
  diag(C) = 0
  idx = which(abs(C) >= cutoff, arr.ind = TRUE)
  idx = idx[idx[, 1] < idx[, 2], , drop = FALSE]
  if (nrow(idx) == 0) {
    max_cor = max(abs(C[upper.tri(C)]))
    stop(sprintf("No LV pairs exceed cutoff (%.2f). Maximum correlation is %.3f",
      cutoff, max_cor), call. = FALSE)
  }

  edges = data.frame(
    from = rownames(C)[idx[, 1]],
    to = colnames(C)[idx[, 2]],
    r = C[idx],
    stringsAsFactors = FALSE
  )
  g = igraph::graph_from_data_frame(edges, directed = FALSE)

  edge_guide_fun = get0("guide_edge_colourbar", asNamespace("ggraph"))
  if (is.null(edge_guide_fun)) {
    edge_guide_fun = get0("guide_edge_colorbar", asNamespace("ggraph"))
  }
  guide_obj = if (is.null(edge_guide_fun)) ggplot2::guide_colourbar() else edge_guide_fun()

  ggraph::ggraph(g, layout = "fr") +
    ggraph::geom_edge_link(ggplot2::aes(width = abs(.data$r), colour = .data$r)) +
    ggraph::scale_edge_width(range = c(0.3, 3)) +
    ggraph::scale_edge_colour_gradient2(
      limits = c(-1, 1), midpoint = 0,
      low = "blue", mid = "grey", high = "red",
      guide = guide_obj
    ) +
    ggraph::geom_node_point(size = 4, colour = "grey30") +
    ggraph::geom_node_text(ggplot2::aes(label = name), repel = TRUE, size = 3) +
    ggplot2::theme_void(base_family = font) +
    ggplot2::labs(title = sprintf("LV network |r| \u2265 %.2f (%s)%s", cutoff, method, title_suffix))
}

# -------- Variance, Scree, Scores ---------------------------------------------

.mbspls_plot_variance_from_model = function(
  model,
  show_total = TRUE,
  viridis_option = "D",
  layout = c("grouped", "facet"),
  show_values = TRUE,
  accuracy = 1,
  flip = FALSE,
  ev_block_override = NULL,
  ev_comp_override = NULL,
  title_suffix = "",
  font = "sans"
) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for this plot.")
  }
  if (!requireNamespace("scales", quietly = TRUE)) {
    stop("Package 'scales' is required for labeling.")
  }

  layout = match.arg(layout)

  ev_blk = ev_block_override %||% model$ev_block
  if (is.null(ev_blk)) {
    stop("No block-wise variance info in model (train the operator).")
  }

  ev_blk = as.matrix(ev_blk)
  if (!is.numeric(ev_blk)) {
    stop("ev_block must be numeric.")
  }

  rn = rownames(ev_blk)
  if (is.null(rn) || length(rn) != nrow(ev_blk)) {
    rn = paste0("LC_", sprintf("%02d", seq_len(nrow(ev_blk))))
  }
  cn = colnames(ev_blk)
  if (is.null(cn) || length(cn) != ncol(ev_blk)) {
    cn = names(model$blocks)
    if (is.null(cn) || length(cn) != ncol(ev_blk)) {
      cn = paste0("Block_", seq_len(ncol(ev_blk)))
    }
  }
  rownames(ev_blk) = rn
  colnames(ev_blk) = cn

  df_long = data.frame(
    component = factor(rep(rn, each = length(cn)), levels = rn),
    block = factor(rep(cn, times = length(rn)), levels = cn),
    explained = as.numeric(ev_blk),
    check.names = FALSE
  )

  ev_table = data.frame(component = rn, check.names = FALSE)
  for (j in seq_along(cn)) ev_table[[cn[j]]] = ev_blk[, j]

  tot_vec = ev_comp_override %||% model$ev_comp
  if (!is.null(tot_vec)) {
    if (!is.null(names(tot_vec)) && length(unique(names(tot_vec))) == length(tot_vec)) {
      reord = as.numeric(tot_vec[rn])
      if (length(reord) == length(rn)) tot_vec <- reord
    }
    if (length(tot_vec) == length(rn)) {
      ev_table$total = as.numeric(tot_vec)
    }
  }

  ev_table_percent = ev_table
  if (ncol(ev_table_percent) > 1) {
    ev_table_percent[-1] = lapply(ev_table_percent[-1],
      scales::percent, accuracy = accuracy)
  }

  p_base =
    if (layout == "grouped") {
      ggplot2::ggplot(df_long, ggplot2::aes(x = component, y = explained, fill = block)) +
        ggplot2::geom_col(position = ggplot2::position_dodge2(width = 0.9, preserve = "single"))
    } else {
      ggplot2::ggplot(df_long, ggplot2::aes(x = block, y = explained, fill = block)) +
        ggplot2::geom_col() +
        ggplot2::facet_wrap(~component, nrow = 1)
    }

  if (isTRUE(show_values)) {
    if (layout == "grouped") {
      p_base = p_base +
        ggplot2::geom_text(
          ggplot2::aes(label = scales::percent(explained, accuracy = accuracy)),
          position = ggplot2::position_dodge2(width = 0.9, preserve = "single"),
          vjust = -0.3, size = 3
        )
    } else {
      p_base = p_base +
        ggplot2::geom_text(
          ggplot2::aes(label = scales::percent(explained, accuracy = accuracy)),
          vjust = -0.3, size = 3
        )
    }
  }

  p = p_base +
    ggplot2::scale_y_continuous(labels = scales::label_percent(accuracy = accuracy),
      expand = ggplot2::expansion(mult = c(0, 0.12))) +
    ggplot2::scale_fill_viridis_d(option = viridis_option) +
    ggplot2::labs(y = "Variance explained", x = NULL,
      title = paste0("MB-sPLS: block-wise variance per component", title_suffix)) +
    ggplot2::guides(fill = ggplot2::guide_legend(title = "block")) +
    ggplot2::theme_minimal(base_size = 11, base_family = font) +
    ggplot2::theme(legend.position = "right")

  if (isTRUE(show_total) && "total" %in% names(ev_table)) {
    df_tot = data.frame(component = factor(rn, levels = rn),
      total = as.numeric(ev_table$total))
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

  if (isTRUE(flip)) p <- p + ggplot2::coord_flip()

  attr(p, "ev_table") = ev_table
  attr(p, "ev_table_percent") = ev_table_percent
  p
}

.mbspls_plot_scree_from_model = function(model, cumulative = FALSE,
  obj_override = NULL, title_suffix = "", font = "sans") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for this plot.")
  }
  obj = obj_override %||% model$obj_vec
  if (is.null(obj)) {
    stop("No per-component objective found (train the operator).")
  }

  rn = names(obj)
  if (is.null(rn)) rn <- paste0("LC_", sprintf("%02d", seq_along(obj)))
  df = data.frame(
    comp = factor(rn, levels = rn),
    obj  = as.numeric(obj),
    cum  = cumsum(as.numeric(obj))
  )
  ylab = if (cumulative) "Cumulative objective" else "Latent correlation (MAC/Frobenius)"

  ggplot2::ggplot(df, ggplot2::aes(.data$comp, if (cumulative) .data$cum else .data$obj, group = 1)) +
    ggplot2::geom_line() + ggplot2::geom_point(size = 2) +
    ggplot2::labs(x = NULL, y = ylab,
      title = paste0(if (!cumulative) {
        "MB-sPLS scree (objective per component)"
      } else {
        "MB-sPLS cumulative objective"
      },
      title_suffix)) +
    ggplot2::theme_minimal(base_size = 11, base_family = font)
}

.mbspls_plot_scores_from_model = function(model,
  component = 1,
  standardize = TRUE,
  density = c("none", "contour", "hex"),
  annotate = TRUE,
  T_override = NULL,
  title_suffix = "",
  font = "sans") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for this plot.")
  }
  density = match.arg(density)

  Tmat = T_override %||% model$T_mat
  if (is.null(Tmat) || ncol(Tmat) < 2) {
    stop("Need at least two latent variables to draw score plots.")
  }
  if (component < 1 || component > model$ncomp) {
    stop("`component` out of range.")
  }

  lv_cols = paste0("LV", component, "_", names(model$blocks))
  missing = setdiff(lv_cols, colnames(Tmat))
  if (length(missing)) {
    stop("Score columns missing from T_mat: ", paste(missing, collapse = ", "))
  }

  S = as.data.frame(Tmat[, lv_cols, drop = FALSE])
  names(S) = names(model$blocks)

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
    slope = if (!is.null(fit)) unname(coef(fit)[2]) else NA_real_
    data.frame(block_x = pr[1], block_y = pr[2],
      n = n, r = r, ccc = ccc, slope = slope,
      stringsAsFactors = FALSE)
  }))

  p = ggplot2::ggplot(df, ggplot2::aes(.data$x, .data$y)) +
    ggplot2::geom_point(alpha = 0.55, size = 1.5)

  if (density == "contour") {
    p = p + ggplot2::stat_density_2d(
      ggplot2::aes(level = after_stat(level)), linewidth = 0.3
    )
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
      label = sprintf("r = %.2f\nccc = %.2f\nslope = %.2f\nn = %d",
        r, ccc, slope, n)
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
        subtitle = "Z-scored per block; dashed: y = x; solid: LS fit; r = Pearson, ccc = Lin’s concordance"
      ) +
      ggplot2::theme_minimal(base_size = 11, base_family = font)
  } else {
    p +
      ggplot2::facet_grid(.data$block_y ~ .data$block_x, scales = "free") +
      ggplot2::labs(
        x = sprintf("Scores: LV%d (block)", component),
        y = sprintf("Scores: LV%d (block)", component),
        title = sprintf("MB-sPLS cross-block score agreement - LC%d%s", component, title_suffix),
        subtitle = "Dashed: y = x; solid: LS fit; r = Pearson, ccc = Lin’s concordance"
      ) +
      ggplot2::theme_minimal(base_size = 11, base_family = font)
  }
}

# -------- Bootstrap component (validation on NEW data) -------------------------

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
      subtitle = sprintf("Statistic: %s • %.0f%% CI • n_boot = %d",
        tolower(perf), df$conf[1] * 100, df$nboot[1]),
      x = NULL,
      y = "Latent correlation (MAC/Frobenius)",
      caption = "Red cross = observed correlation on original test data; filled dot = bootstrap mean"
    ) +
    ggplot2::theme_minimal(base_size = 11, base_family = font)
}

# -------- Payload reader -------------------------------------------------------

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

  stop("No evaluation payload found. Call predict() with a log_env set (or use val_task=), ",
    "or pass payload= explicitly.")
}
