#' Autoplot for GraphLearner with MB-sPLS
#'
#' @description
#' MB-sPLS-specific \code{ggplot2::autoplot()} \code{type}s for a *trained*
#' \code{GraphLearner} where the trained model is available at \code{gl$model$mbspls}.
#'
#' @param object A trained \code{GraphLearner}.
#' @param type Character string, one of:
#'   \itemize{
#'     \item \code{"mbspls_weights"} – bar plot of strongest (|w|) weights per component×block. Args: \code{top_n}, \code{palette}.
#'     \item \code{"mbspls_heatmap"} – heat map of cross-correlations of latent scores. Arg: \code{method}.
#'     \item \code{"mbspls_network"} – LV network for edges with \code{|r| >= cutoff}. Args: \code{cutoff}, \code{method}.
#'     \item \code{"mbspls_variance"} – non-stacked bars of block-wise variance explained; optional total line.
#'     \item \code{"mbspls_scree"} – objective per component or cumulative.
#'     \item \code{"mbspls_scores"} – score plots across all block pairs for a chosen LV. Arg: \code{component = 1}.
#'     \item \code{"bootstrap_variable"} – bootstrap confidence intervals for variable-wise correlations. Args: \code{variable_name}, \code{component}.
#'     \item \code{"bootstrap_component"} – bootstrap confidence intervals for component-wise correlations. Arg: \code{component}.
#'   }
#' @param ... Additional arguments forwarded to the chosen \code{type}.
#'   The following extras are recognized by performance/score plots:
#'   \itemize{
#'     \item \code{new_task} (\pkg{mlr3} \code{Task}): if supplied, compute EVs/MAC
#'           and scores *on the new data* using the learned weights (calls
#'           \code{mbspls_eval_new_data()} internally) and plot test-side metrics.
#'     \item \code{mbspls_id} (\code{character(1)}): id of the MB-sPLS node when multiple are present.
#'   }
#'
#' @return A \pkg{ggplot2}/\pkg{ggraph} object.
#'
#' @examples
#' # gl <- GraphLearner$new(...); gl$train(task)
#' # autoplot(gl, type = "mbspls_variance")
#' # autoplot(gl, type = "mbspls_variance", new_task = task_test)
#' # autoplot(gl, type = "mbspls_scree")
#' # autoplot(gl, type = "mbspls_scores", component = 1, new_task = task_test)
#'
#' @importFrom ggplot2 autoplot
#' @export
#' @method autoplot GraphLearner
autoplot.GraphLearner <- function(object,
  type = c("mbspls_weights", "mbspls_heatmap", "mbspls_network",
           "mbspls_variance", "mbspls_scree", "mbspls_scores",
           "mbspls_bootstrap_component", "mbspls_bootstrap_comp"),
  ...) {
  type <- match.arg(type)
  mod  <- .locate_mbspls_model(object)

  dots <- list(...)
  new_task  <- dots$new_task %||% NULL
  mbspls_id <- dots$mbspls_id %||% NULL
  dots$new_task <- dots$mbspls_id <- NULL

  needs_eval <- !is.null(new_task) && type %in% c(
    "mbspls_variance", "mbspls_scree", "mbspls_scores",
    "mbspls_heatmap", "mbspls_network",
    "mbspls_bootstrap_component",
    "mbspls_bootstrap_comp"
  )

  eval_payload <- NULL
  if (isTRUE(needs_eval)) {
    eval_payload <- .mbspls_eval_newdata_dispatch(object, new_task, mbspls_id)
  }

  title_suffix <- if (isTRUE(needs_eval)) " (new data)" else ""

  p <- switch(
    type,
    mbspls_weights = do.call(.mbspls_plot_weights_from_model, c(list(model = mod), dots)),
    mbspls_heatmap = do.call(.mbspls_plot_heatmap_from_model,
                      c(list(model = mod, T_override = if (!is.null(eval_payload)) eval_payload$T_mat else NULL,
                             title_suffix = title_suffix), dots)),
    mbspls_network = do.call(.mbspls_plot_network_from_model,
                      c(list(model = mod, T_override = if (!is.null(eval_payload)) eval_payload$T_mat else NULL,
                             title_suffix = title_suffix), dots)),
    mbspls_variance = do.call(.mbspls_plot_variance_from_model,
                      c(list(model = mod,
                             ev_block_override = if (!is.null(eval_payload)) eval_payload$ev_block else NULL,
                             ev_comp_override  = if (!is.null(eval_payload)) eval_payload$ev_comp  else NULL,
                             title_suffix = title_suffix), dots)),
    mbspls_scree = do.call(.mbspls_plot_scree_from_model,
                      c(list(model = mod,
                             obj_override = if (!is.null(eval_payload)) eval_payload$mac_comp else NULL,
                             title_suffix = title_suffix), dots)),
    mbspls_scores = do.call(.mbspls_plot_scores_from_model,
                      c(list(model = mod, T_override = if (!is.null(eval_payload)) eval_payload$T_mat else NULL,
                             title_suffix = title_suffix), dots)),
    mbspls_bootstrap_component = do.call(.mbspls_plot_bootstrap_component,
                      c(list(model = object, payload = eval_payload, mbspls_id = mbspls_id), dots)),
  )
  p
}

# Internal: use whichever helper exists
.mbspls_eval_newdata_dispatch <- function(gl, task, mbspls_id = NULL) {
  fn <- get0("mbspls_eval_new_data", mode = "function")
  if (is.null(fn)) fn <- get0("mbspls_eval_on_new", mode = "function")
  if (is.null(fn)) stop("Need mbspls_eval_new_data() or mbspls_eval_on_new() in your package.")
  fn(gl, task, mbspls_id)
}

.mbspls_get_eval_payload <- function(model = NULL, payload = NULL, mbspls_id = NULL) {
  if (!is.null(payload)) return(payload)

  if (inherits(model, "GraphLearner")) {
    ids <- names(model$graph$pipeops)
    if (is.null(mbspls_id)) {
      cand <- ids[vapply(model$graph$pipeops, inherits, logical(1), "PipeOpMBsPLS")]
      if (!length(cand)) stop("No PipeOpMBsPLS node found in the graph.")
      mbspls_id <- cand[1]
    }
    po <- model$graph$pipeops[[mbspls_id]]
    env <- po$param_set$values$log_env
    if (inherits(env, "environment") && !is.null(env$last)) return(env$last)
  }
  if (!is.null(model$log_env) && !is.null(model$log_env$last)) return(model$log_env$last)
  if (!is.null(model$last)) return(model$last)

  stop("No evaluation payload found. Call predict() with a log_env set (or use new_task=), or pass payload= explicitly.")
}

# Internal: locate trained MB-sPLS model inside a GraphLearner$model
.locate_mbspls_model <- function(gl) {
  if (!inherits(gl, "GraphLearner"))
    stop("Expected a GraphLearner.", call. = FALSE)
  mod <- gl$model
  if (is.null(mod))
    stop("GraphLearner appears to be untrained (model is NULL).", call. = FALSE)
  if (!is.null(mod$mbspls))
    return(mod$mbspls)
  stop("Could not find a trained MB-sPLS model in gl$model. The PipeOp in the graph is untrained.", call. = FALSE)
}

.mbspls_plot_weights_from_model <- function(
  model,
  top_n = NULL,
  palette = "Dark2",
  label_width = 24,
  sep_color = "grey40",
  sep_size = 0.8,
  weights_view = c("weights", "bootstrap"),
  summary_fun = NULL,
  filter_min_frequency = NULL,
  alpha_by_stability = FALSE,
  weights_source = c("raw", "stability_filtered")
) {
  requireNamespace("ggplot2")
  requireNamespace("dplyr")
  requireNamespace("tibble")
  requireNamespace("RColorBrewer")
  requireNamespace("stringr")
  has_patchwork <- requireNamespace("patchwork", quietly = TRUE)

  weights_view   <- match.arg(weights_view)
  weights_source <- match.arg(weights_source)

  # optional stability/selection frequency table
  .freq_tbl <- {
    sf <- model$weights_selectfreq
    if (!is.null(sf)) {
      if (requireNamespace("data.table", quietly = TRUE) && data.table::is.data.table(sf)) {
        sf <- as.data.frame(sf)
      }
      if (!nrow(sf)) sf <- NULL
    }
    sf
  }

  # pick which weights to plot
  W_use <- model$weights
  if (weights_source == "stability_filtered" && !is.null(model$weights_stability_filtered)) {
    W_use <- model$weights_stability_filtered
  }

  # consistent ordering in facets
  block_levels <- names(model$blocks)
  comp_levels  <- paste0("LC_", sprintf("%02d", seq_len(model$ncomp)))

  # reusable helpers -----------------------------------------------------------
  .pal_vals <- function(palette) {
    if (length(palette) == 1L) RColorBrewer::brewer.pal(3, palette)[c(3, 1)]
    else rep_len(palette, 2)[1:2]
  }
  .base_size_from_n <- function(n) {
    # scale gently with number of features shown; clamp to [7, 14]
    bs <- 13 - 0.07 * (n - 20)
    max(7, min(14, bs))
  }
  .nice_label <- function(x) gsub("_", " ", x, fixed = TRUE)

  # one LC per plot; stitch with patchwork if >1 LCs --------------------------
  build_weights_long <- function() {
    dplyr::bind_rows(lapply(seq_along(W_use), function(k) {
      dplyr::bind_rows(lapply(names(W_use[[k]]), function(b) {
        tibble::tibble(
          component = sprintf("LC_%01d", k),
          block     = b,
          feature   = names(W_use[[k]][[b]]),
          weight    = as.numeric(W_use[[k]][[b]])
        )
      }))
    })) |>
      dplyr::mutate(abs_w = abs(weight)) |>
      dplyr::filter(abs_w > 0)
  }

  build_boot_long <- function(vb) {
    dplyr::bind_rows(lapply(seq_along(vb), function(k) {
      dplyr::bind_rows(lapply(names(vb[[k]]), function(b) {
        fb <- vb[[k]][[b]]  # list(feature -> numeric vector)
        dplyr::bind_rows(lapply(names(fb), function(f) {
          tibble::tibble(
            component = sprintf("LC_%02d", k),
            block     = b,
            feature   = f,
            value     = as.numeric(fb[[f]])
          )
        }))
      }))
    })) |>
      dplyr::mutate(abs_v = abs(value)) |>
      dplyr::filter(abs_v > 0)
  }

  # Shared plotting core for a single LC (weights view) -----------------------
  plot_one_lc_weights <- function(df_lc, comp_label) {
    # attach frequency (optional)
    if (!is.null(.freq_tbl)) {
      df_lc <- dplyr::left_join(df_lc, .freq_tbl,
                                by = c("component", "block", "feature"))
    }
    if (!"freq" %in% names(df_lc)) df_lc$freq <- NA_real_

    if (is.numeric(filter_min_frequency)) {
      df_lc <- dplyr::filter(df_lc, is.na(freq) | freq >= filter_min_frequency)
    }

    # order & per-panel factor for correct within-facet order
    df_lc$block     <- factor(df_lc$block,     levels = block_levels)
    df_lc$component <- factor(df_lc$component, levels = comp_levels)

    df_lc <- df_lc |>
      dplyr::arrange(block, abs_w) |>
      dplyr::mutate(
        feature_label = stringr::str_wrap(feature, width = label_width),
        axis_id       = paste(block, feature, sep = "___")
      )
    df_lc$axis_id <- factor(df_lc$axis_id, levels = unique(df_lc$axis_id))
    lab_map <- setNames(as.character(df_lc$feature_label), as.character(df_lc$axis_id))

    # alpha by stability
    if (isTRUE(alpha_by_stability)) {
      af <- df_lc$freq
      af[!is.finite(af)] <- 1
      df_lc$alpha_freq <- pmax(pmin(af, 1), 0)
    } else {
      df_lc$alpha_freq <- 1
    }

    # pretty facet labels (block & LC)
    df_lc$block_lab <- factor(.nice_label(as.character(df_lc$block)),
                              levels = .nice_label(block_levels))
    comp_lab        <- .nice_label(as.character(comp_label))
    df_lc$component_lab <- factor(comp_lab, levels = comp_lab)

    # separators between blocks: one thin rule at the top of each block panel
    block_lvls <- levels(df_lc$block_lab)
    sep_df <- if (length(block_lvls) > 1L) {
      expand.grid(
        block_lab    = factor(block_lvls[-length(block_lvls)], levels = block_lvls),
        component_lab= factor(comp_lab, levels = comp_lab),
        KEEP.OUT.ATTRS = FALSE
      )
    } else data.frame(block_lab = factor(), component_lab = factor())
    if (nrow(sep_df)) sep_df$xintercept <- -Inf

    # font scaling
    base_size <- .base_size_from_n(nrow(df_lc))

    ggplot2::ggplot(
      df_lc,
      ggplot2::aes(axis_id, weight, fill = weight > 0, alpha = alpha_freq)
    ) +
      ggplot2::geom_col(width = 0.85, show.legend = FALSE) +
      ggplot2::scale_alpha_identity() +
      ggplot2::geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey70") +
      # facet-aware separator (vertical in data space; horizontal after coord_flip)
      (if (nrow(sep_df)) ggplot2::geom_vline(
         data = sep_df,
         ggplot2::aes(xintercept = xintercept),
         colour = sep_color, linewidth = sep_size, inherit.aes = FALSE
       ) else NULL) +
      ggplot2::facet_grid(
        rows = ggplot2::vars(block_lab),
        cols = ggplot2::vars(component_lab),
        scales = "free_y",
        space  = "free_y",
        switch = "y"
      ) +
      ggplot2::scale_fill_manual(values = .pal_vals(palette)) +
      ggplot2::scale_x_discrete(labels = function(x) lab_map[as.character(x)]) +
      ggplot2::coord_flip() +
      ggplot2::labs(
        x = NULL, y = "Weight",
        title = sprintf("Sparse weights per block — %s", comp_lab)
      ) +
      ggplot2::theme_minimal(base_size = base_size) +
      ggplot2::theme(
        panel.grid.major.y = ggplot2::element_blank(),
        panel.grid.minor   = ggplot2::element_blank(),
        panel.spacing      = grid::unit(0, "pt"),
        strip.placement    = "outside",
        strip.background   = ggplot2::element_rect(fill = NA, colour = NA),
        strip.text.y.left  = ggplot2::element_text(angle = 0, face = "bold"),
        strip.text.x       = ggplot2::element_text(face = "bold"),
        axis.text.y        = ggplot2::element_text(size = base_size * 0.8)
      )
  }

  # Shared plotting core for a single LC (bootstrap view) ---------------------
  plot_one_lc_boot <- function(df_lc, comp_label) {
    # join stability freq (optional)
    if (!is.null(.freq_tbl)) {
      df_lc <- dplyr::left_join(df_lc, .freq_tbl,
                                by = c("component", "block", "feature"))
    }
    if (!"freq" %in% names(df_lc)) df_lc$freq <- NA_real_
    if (is.numeric(filter_min_frequency)) {
      df_lc <- dplyr::filter(df_lc, is.na(freq) | freq >= filter_min_frequency)
    }

    df_lc$block     <- factor(df_lc$block,     levels = block_levels)
    df_lc$component <- factor(df_lc$component, levels = comp_levels)

    if (isTRUE(alpha_by_stability)) {
      af <- df_lc$freq
      af[!is.finite(af)] <- 1
      df_lc$alpha_freq <- pmax(pmin(af, 1), 0)
    } else {
      df_lc$alpha_freq <- 1
    }

    # pretty labels
    df_lc$block_lab <- factor(.nice_label(as.character(df_lc$block)),
                              levels = .nice_label(block_levels))
    comp_lab        <- .nice_label(as.character(comp_label))

    # separators between blocks per component (after coord_flip shows as horizontal rule)
    block_lvls <- levels(df_lc$block_lab)
    sep_df <- if (length(block_lvls) > 1L) {
      data.frame(
        block_lab  = factor(block_lvls[-length(block_lvls)], levels = block_lvls),
        xintercept = -Inf
      )
    } else data.frame(block_lab = factor(), xintercept = numeric())

    base_size <- .base_size_from_n(nrow(df_lc))

    ggplot2::ggplot(
      df_lc,
      ggplot2::aes(stringr::str_wrap(feature, width = label_width), value, alpha = alpha_freq)
    ) +
      ggplot2::geom_violin(width = 0.8, fill = "grey40", colour = NA, show.legend = FALSE) +
      ggplot2::scale_alpha_identity() +
      ggplot2::geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey70") +
      # facet-aware separator (vertical in data space → horizontal after coord_flip)
      (if (nrow(sep_df)) ggplot2::geom_vline(
         data = sep_df,
         ggplot2::aes(xintercept = xintercept),
         colour = sep_color, linewidth = sep_size, inherit.aes = FALSE
       ) else NULL) +
      ggplot2::facet_grid(
        rows = ggplot2::vars(block_lab),
        cols = NULL,
        scales = "free_y",
        space  = "free_y",
        switch = "y"
      ) +
      ggplot2::scale_x_discrete(labels = function(x) stringr::str_wrap(x, width = label_width)) +
      ggplot2::coord_flip() +
      ggplot2::labs(
        x = NULL,
        y = if (is.function(summary_fun) && identical(summary_fun, abs)) "Bootstrapped |weight|" else "Bootstrapped weight",
        title = sprintf("Bootstrap distributions of weights — %s", comp_lab)
      ) +
      ggplot2::theme_minimal(base_size = base_size) +
      ggplot2::theme(
        panel.grid.major.y = ggplot2::element_blank(),
        panel.grid.minor   = ggplot2::element_blank(),
        panel.spacing      = grid::unit(0, "pt"),
        strip.placement    = "outside",
        strip.background   = ggplot2::element_rect(fill = NA, colour = NA),
        strip.text.y.left  = ggplot2::element_text(angle = 0, face = "bold"),
        strip.text.x       = ggplot2::element_text(face = "bold"),
        axis.text.y        = ggplot2::element_text(size = base_size * 0.8)
      )
  }

  # Branches ------------------------------------------------------------------
  if (weights_view == "weights") {
    long <- build_weights_long()

    if (!is.null(top_n)) {
      long <- long |>
        dplyr::group_by(component, block) |>
        dplyr::slice_max(abs_w, n = max(1L, as.integer(top_n)), with_ties = FALSE) |>
        dplyr::ungroup()
    }

    # loop over components and compose horizontally
    comps <- unique(long$component)
    plots <- lapply(comps, function(comp) {
      plot_one_lc_weights(long[long$component == comp, , drop = FALSE], comp)
    })
    if (length(plots) == 1L || !has_patchwork) {
      return(plots[[1L]])
    } else {
      return(patchwork::wrap_plots(plots, nrow = 1))
    }
  }

  # bootstrap view
  vb <- model$weights_boot_vectors
  if (is.null(vb)) {
    stop("weights_boot_vectors not found in model state. ",
         "Enable bootstrap_test=TRUE and boot_store_vectors=TRUE during training.")
  }
  boot_long <- build_boot_long(vb)

  if (is.function(summary_fun)) {
    boot_long$value <- summary_fun(boot_long$value)
  }

  comps <- unique(boot_long$component)
  plots <- lapply(comps, function(comp) {
    plot_one_lc_boot(boot_long[boot_long$component == comp, , drop = FALSE], comp)
  })
  if (length(plots) == 1L || !has_patchwork) {
    return(plots[[1L]])
  } else {
    return(patchwork::wrap_plots(plots, nrow = 1))
  }
}

.mbspls_plot_heatmap_from_model <- function(model, method = "spearman",
                                            T_override = NULL, title_suffix = "") {
  requireNamespace("ggplot2")
  scores <- T_override %||% model$T_mat
  if (is.null(scores) || NCOL(scores) < 2)
    stop("Need at least two latent variables to draw a heat map.", call. = FALSE)

  if (is.null(colnames(scores))) {
    colnames(scores) <- unlist(lapply(seq_len(model$ncomp), function(k)
      paste0("LV", sprintf("%02d", k), "_", names(model$blocks))))
  }
  C <- stats::cor(scores, method = method)
  rn <- rownames(C); cn <- colnames(C)
  meltC <- expand.grid(row = rn, col = cn, stringsAsFactors = FALSE)
  meltC$value <- as.vector(C)
  meltC$row <- factor(meltC$row, levels = rn)
  meltC$col <- factor(meltC$col, levels = cn)

  ggplot2::ggplot(meltC, ggplot2::aes(row, col, fill = value)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient2(limits = c(-1, 1), midpoint = 0,
                                  low = "blue", mid = "white", high = "red") +
    ggplot2::theme_minimal() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5, hjust = 1),
                   panel.grid  = ggplot2::element_blank()) +
    ggplot2::labs(title = sprintf("Cross-correlation of latent scores (%s)%s", method, title_suffix),
                  x = NULL, y = NULL, fill = "r")
}

.mbspls_plot_network_from_model <- function(model, cutoff = 0.3, method = "spearman",
                                            T_override = NULL, title_suffix = "") {
  requireNamespace("igraph"); requireNamespace("ggraph"); requireNamespace("ggplot2")

  scores <- T_override %||% model$T_mat
  if (is.null(scores) || NCOL(scores) < 2)
    stop("Need at least two latent variables to draw a network.", call. = FALSE)

  if (is.null(colnames(scores))) {
    colnames(scores) <- unlist(lapply(seq_len(model$ncomp), function(k)
      paste0("LV", sprintf("%02d", k), "_", names(model$blocks))))
  }

  C <- stats::cor(scores, method = method)
  diag(C) <- 0
  idx <- which(abs(C) >= cutoff, arr.ind = TRUE)
  idx <- idx[idx[, 1] < idx[, 2], , drop = FALSE]
  if (nrow(idx) == 0) {
    max_cor <- max(abs(C[upper.tri(C)]))
    stop(sprintf("No LV pairs exceed cutoff (%.2f). Maximum correlation is %.3f",
                 cutoff, max_cor), call. = FALSE)
  }

  edges <- data.frame(
    from = rownames(C)[idx[, 1]],
    to   = colnames(C)[idx[, 2]],
    r    = C[idx],
    stringsAsFactors = FALSE
  )
  g <- igraph::graph_from_data_frame(edges, directed = FALSE)

  # --- pick a guide object safely (handle both spellings & fallback) ---
  edge_guide_fun <- get0("guide_edge_colourbar", asNamespace("ggraph"))
  if (is.null(edge_guide_fun))
    edge_guide_fun <- get0("guide_edge_colorbar", asNamespace("ggraph"))
  guide_obj <- if (is.null(edge_guide_fun)) ggplot2::guide_colourbar() else edge_guide_fun()

  ggraph::ggraph(g, layout = "fr") +
    ggraph::geom_edge_link(ggplot2::aes(width = abs(r), colour = r)) +
    ggraph::scale_edge_width(range = c(0.3, 3)) +
    ggraph::scale_edge_colour_gradient2(
      limits = c(-1, 1), midpoint = 0,
      low = "blue", mid = "grey", high = "red",
      guide = guide_obj   # <- object, not string
    ) +
    ggraph::geom_node_point(size = 4, colour = "grey30") +
    ggraph::geom_node_text(ggplot2::aes(label = name), repel = TRUE, size = 3) +
    ggplot2::theme_void() +
    ggplot2::labs(title = sprintf("LV network |r| \u2265 %.2f (%s)%s", cutoff, method, title_suffix))
}

.mbspls_plot_variance_from_model <- function(
  model,
  show_total = TRUE,
  viridis_option = "D",
  layout = c("grouped", "facet"),
  show_values = TRUE,
  accuracy = 1,
  flip = FALSE,
  ev_block_override = NULL,
  ev_comp_override  = NULL,
  title_suffix = ""
) {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("Package 'ggplot2' is required for this plot.")
  if (!requireNamespace("scales", quietly = TRUE))
    stop("Package 'scales' is required for labeling.")

  layout <- match.arg(layout)
  ev_blk <- ev_block_override %||% model$ev_block
  if (is.null(ev_blk))
    stop("No block-wise variance info in model (train the operator).")

  rn <- rownames(ev_blk) %||% paste0("LC_", sprintf("%02d", nrow(ev_blk)))
  cn <- colnames(ev_blk) %||% names(model$blocks)
  n_blocks <- length(cn)

  df_long <- data.frame(
    component = factor(rep(rn, each = n_blocks), levels = rn),
    block     = factor(rep(cn, times = length(rn)), levels = cn),
    explained = as.numeric(ev_blk),
    check.names = FALSE
  )

  # Pretty tables as attributes
  ev_table <- reshape2::dcast(df_long, component ~ block, value.var = "explained")
  tot_vec  <- ev_comp_override %||% model$ev_comp
  if (!is.null(tot_vec)) ev_table$total <- as.numeric(tot_vec)
  ev_table_percent <- ev_table
  ev_table_percent[-1] <- lapply(ev_table_percent[-1],
                                 scales::percent, accuracy = accuracy)

  p_base <-
    if (layout == "grouped") {
      ggplot2::ggplot(df_long, ggplot2::aes(x = component, y = explained, fill = block)) +
        ggplot2::geom_col(
          position = ggplot2::position_dodge2(width = 0.9, preserve = "single")  # grouped bars
        )
    } else {
      ggplot2::ggplot(df_long, ggplot2::aes(x = block, y = explained, fill = block)) +
        ggplot2::geom_col() +
        ggplot2::facet_wrap(~ component, nrow = 1)
    }

  if (isTRUE(show_values)) {
    if (layout == "grouped") {
      p_base <- p_base +
        ggplot2::geom_text(
          ggplot2::aes(label = scales::percent(explained, accuracy = accuracy)),
          position = ggplot2::position_dodge2(width = 0.9, preserve = "single"),
          vjust = -0.3, size = 3
        )
    } else {
      p_base <- p_base +
        ggplot2::geom_text(
          ggplot2::aes(label = scales::percent(explained, accuracy = accuracy)),
          vjust = -0.3, size = 3
        )
    }
  }

  p <- p_base +
    ggplot2::scale_y_continuous(labels = scales::label_percent(accuracy = accuracy),
                                expand = ggplot2::expansion(mult = c(0, 0.12))) +
    ggplot2::scale_fill_viridis_d(option = viridis_option) +
    ggplot2::labs(y = "Variance explained", x = NULL,
                  title = paste0("MB-sPLS: block-wise variance per component", title_suffix)) +
    ggplot2::guides(fill = ggplot2::guide_legend(title = "block")) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(legend.position = "right")

  if (isTRUE(show_total) && !is.null(tot_vec)) {
    df_tot <- data.frame(component = factor(rn, levels = rn),
                         total = as.numeric(tot_vec))
    if (layout == "grouped") {
      p <- p +
        ggplot2::geom_line(data = df_tot, ggplot2::aes(component, total, group = 1),
                           inherit.aes = FALSE) +
        ggplot2::geom_point(data = df_tot, ggplot2::aes(component, total),
                            inherit.aes = FALSE, size = 1.9)
    } else {
      p <- p + ggplot2::geom_point(data = df_tot,
                                   ggplot2::aes(x = Inf, y = total),
                                   inherit.aes = FALSE, size = 1.9, alpha = 0.9)
    }
  }
  if (isTRUE(flip)) p <- p + ggplot2::coord_flip()

  attr(p, "ev_table") <- ev_table
  attr(p, "ev_table_percent") <- ev_table_percent
  p
}


.mbspls_plot_scree_from_model <- function(model, cumulative = FALSE,
                                          obj_override = NULL, title_suffix = "") {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("Package 'ggplot2' is required for this plot.")
  obj <- obj_override %||% model$obj_vec
  if (is.null(obj))
    stop("No per-component objective found (train the operator).")

  rn <- names(obj)
  if (is.null(rn)) rn <- paste0("LC_", sprintf("%02d", seq_along(obj)))
  df <- data.frame(
    comp = factor(rn, levels = rn),
    obj  = as.numeric(obj),
    cum  = cumsum(as.numeric(obj))
  )
  ylab <- if (cumulative) "Cumulative objective" else "Latent correlation (MAC/Frobenius)"

  ggplot2::ggplot(df, ggplot2::aes(comp, if (cumulative) cum else obj, group = 1)) +
    ggplot2::geom_line() + ggplot2::geom_point(size = 2) +
    ggplot2::labs(x = NULL, y = ylab,
                  title = paste0(if (!cumulative) "MB-sPLS scree (objective per component)"
                                  else "MB-sPLS cumulative objective",
                                  title_suffix)) +
    ggplot2::theme_minimal(base_size = 11)
}


.mbspls_plot_scores_from_model <- function(model,
                                           component = 1,
                                           standardize = TRUE,
                                           density = c("none", "contour", "hex"),
                                           annotate = TRUE,
                                           T_override = NULL,
                                           title_suffix = "") {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("Package 'ggplot2' is required for this plot.")
  density <- match.arg(density)

  Tmat <- T_override %||% model$T_mat
  if (is.null(Tmat) || ncol(Tmat) < 2)
    stop("Need at least two latent variables to draw score plots.")
  if (component < 1 || component > model$ncomp)
    stop("`component` out of range.")

  lv_cols <- paste0("LV", component, "_", names(model$blocks))
  missing <- setdiff(lv_cols, colnames(Tmat))
  if (length(missing))
    stop("Score columns missing from T_mat: ", paste(missing, collapse = ", "))

  S <- as.data.frame(Tmat[, lv_cols, drop = FALSE])
  names(S) <- names(model$blocks)

  if (isTRUE(standardize)) {
    S[] <- lapply(S, function(v) { s <- stats::sd(v); if (is.finite(s) && s > 0) (v - mean(v)) / s else v })
  }

  blks  <- names(model$blocks)
  pairs <- utils::combn(blks, 2, simplify = FALSE)
  df <- do.call(rbind, lapply(pairs, function(pr) {
    data.frame(x = S[[pr[1]]], y = S[[pr[2]]],
               block_x = pr[1], block_y = pr[2],
               stringsAsFactors = FALSE)
  }))

  ccc_fun <- function(x, y) {
    mx <- mean(x); my <- mean(y)
    vx <- stats::var(x); vy <- stats::var(y)
    sxy <- stats::cov(x, y)
    if (vx <= 0 || vy <= 0) return(NA_real_)
    (2 * sxy) / (vx + vy + (mx - my)^2)
  }
  panel_stats <- do.call(rbind, lapply(pairs, function(pr) {
    xx <- S[[pr[1]]]; yy <- S[[pr[2]]]
    ok <- is.finite(xx) & is.finite(yy); xx <- xx[ok]; yy <- yy[ok]
    n  <- length(xx)
    r  <- suppressWarnings(stats::cor(xx, yy, method = "pearson"))
    ccc <- ccc_fun(xx, yy)
    fit <- if (n >= 2) stats::lm(yy ~ xx) else NULL
    slope <- if (!is.null(fit)) unname(coef(fit)[2]) else NA_real_
    data.frame(block_x = pr[1], block_y = pr[2],
               n = n, r = r, ccc = ccc, slope = slope,
               stringsAsFactors = FALSE)
  }))

  p <- ggplot2::ggplot(df, ggplot2::aes(x, y)) +
    ggplot2::geom_point(alpha = 0.55, size = 1.5)

  if (density == "contour") {
    p <- p + ggplot2::stat_density_2d(
      ggplot2::aes(level = after_stat(level)), linewidth = 0.3
    )
  } else if (density == "hex") {
    if (!requireNamespace("hexbin", quietly = TRUE))
      stop("Package 'hexbin' is required for density = 'hex'.")
    p <- p + ggplot2::stat_bin_hex(bins = 20, alpha = 0.8)
  }

  p <- p +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2, linewidth = 0.4) +
    ggplot2::geom_smooth(method = "lm", se = FALSE, linewidth = 0.5, formula = y ~ x)

  if (isTRUE(annotate)) {
    lab <- within(panel_stats, {
      label <- sprintf("r = %.2f\nccc = %.2f\nslope = %.2f\nn = %d",
                       r, ccc, slope, n)
    })
    p <- p + ggplot2::geom_text(
      data = lab, ggplot2::aes(x = -Inf, y = Inf, label = label),
      hjust = -0.05, vjust = 1.05, size = 3, inherit.aes = FALSE
    )
  }

  if (isTRUE(standardize)) {
    p +
      ggplot2::facet_grid(block_y ~ block_x, scales = "fixed") +
      ggplot2::coord_equal() +
      ggplot2::labs(
        x = sprintf("Scores: LV%d (block)", component),
        y = sprintf("Scores: LV%d (block)", component),
        title = sprintf("MB-sPLS cross-block score agreement — LV%d%s", component, title_suffix),
        subtitle = "Z-scored per block; dashed: y = x; solid: LS fit; r = Pearson, ccc = Lin’s concordance"
      ) +
      ggplot2::theme_minimal(base_size = 11)
  } else {
    p +
      ggplot2::facet_grid(block_y ~ block_x, scales = "free") +
      ggplot2::labs(
        x = sprintf("Scores: LV%d (block)", component),
        y = sprintf("Scores: LV%d (block)", component),
        title = sprintf("MB-sPLS cross-block score agreement — LV%d%s", component, title_suffix),
        subtitle = "Dashed: y = x; solid: LS fit; r = Pearson, ccc = Lin’s concordance"
      ) +
      ggplot2::theme_minimal(base_size = 11)
  }
}

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
# Internal: pull the evaluation payload (what predict() writes)
.mbspls_get_eval_payload <- function(object = NULL, payload = NULL, mbspls_id = NULL) {
  if (!is.null(payload)) return(payload)

  # If a GraphLearner is passed: read the MB-sPLS node's log_env$last
  if (inherits(object, "GraphLearner")) {
    ids <- names(object$graph$pipeops)
    if (is.null(mbspls_id)) {
      cand <- ids[vapply(object$graph$pipeops, inherits, logical(1), "PipeOpMBsPLS")]
      if (!length(cand))
        stop("No PipeOpMBsPLS node found in the graph.")
      mbspls_id <- cand[1]
    }
    po <- object$graph$pipeops[[mbspls_id]]
    env <- po$param_set$values$log_env
    if (inherits(env, "environment") && !is.null(env$last)) {
      return(env$last)
    }
  }

  # If an mbspls state list is passed and (unexpectedly) has a log_env pointer
  if (!is.null(object$log_env) && !is.null(object$log_env$last)) {
    return(object$log_env$last)
  }
  if (!is.null(object$last)) {
    return(object$last)
  }

  stop("No evaluation payload found. Call predict() with a log_env set (or use new_task=), ",
       "or pass payload= explicitly.")
}

# -------------------------------------------------------------------
# Bootstrap plot: all variables, faceted by block, for a component
# -------------------------------------------------------------------
# Bootstrap CIs per component (MAC/Frobenius on NEW data)
.mbspls_plot_bootstrap_component <- function(
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
  errorbar_width = 0.22
) {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("Package 'ggplot2' is required for bootstrap plots.")

  # Pull prediction payload (written by predict() with val_test='bootstrap')
  pay <- .mbspls_get_eval_payload(model, payload, mbspls_id)

  bt <- pay$val_bootstrap
  if (is.null(bt))
    stop("No component-level bootstrap in payload (val_bootstrap is NULL). ",
         "Enable validation with val_test='bootstrap' for predict().")

  # Expected columns (as per your logging)
  bt <- as.data.frame(bt, stringsAsFactors = FALSE)
  req <- c("component", "observed_correlation", "boot_mean", "boot_se",
           "boot_p_value", "boot_ci_lower", "boot_ci_upper",
           "confidence_level", "n_boot")
  miss <- setdiff(req, names(bt))
  if (length(miss)) stop("val_bootstrap missing columns: ", paste(miss, collapse = ", "))

  comp_lab <- paste0("LC_", sprintf("%02d", bt$component))
  df <- data.frame(
    component = factor(comp_lab, levels = comp_lab),
    mean  = bt$boot_mean,
    lwr   = bt$boot_ci_lower,
    upr   = bt$boot_ci_upper,
    obs   = bt$observed_correlation,
    pval  = bt$boot_p_value,
    conf  = bt$confidence_level[1],
    nboot = bt$n_boot[1]
  )

  perf <- pay$perf_metric %||% "MAC"

  # Build plot
  p <- ggplot2::ggplot(df, ggplot2::aes(component, mean))

  # Violin & box show the distribution shape and summary (centered at each component)
  if (isTRUE(show_violin)) {
    # Reconstruct a per-component bootstrap vector when available
    # If you saved full replicate vectors per component (e.g. pay$val_boot_vectors[[k]]),
    # you can replace the synthetic normal below with the true replicates for perfect fidelity.
    if (!is.null(pay$val_boot_vectors)) {
      boot_long <- do.call(rbind, lapply(seq_along(pay$val_boot_vectors), function(i) {
        data.frame(component = comp_lab[i], boot = as.numeric(pay$val_boot_vectors[[i]]))
      }))
      boot_long$component <- factor(boot_long$component, levels = comp_lab)
      p <- p +
        ggplot2::geom_violin(data = boot_long,
                             ggplot2::aes(y = boot, x = component),
                             fill = "grey40", alpha = violin_alpha, colour = NA, width = 0.8,
                             inherit.aes = FALSE)
    } else {
      # Fallback: draw a narrow violin using a normal approx (mean & sd)
      # (kept subtle; replace with true replicates when you expose them)
      approx_long <- do.call(rbind, lapply(seq_len(nrow(df)), function(i) {
        m <- df$mean[i]; s <- df$`mean`[i]*0 # placeholder to avoid R CMD check NOTE
        # Estimate sd from CI if SE is present; otherwise derive from (upr-lwr)
        se <- bt$boot_se[i]
        sd_est <- if (is.finite(se) && se > 0) se else max((df$upr[i] - df$lwr[i]) / (2 * 1.96), 1e-6)
        y <- m + stats::rnorm(500L, 0, sd_est)
        data.frame(component = df$component[i], boot = y)
      }))
      p <- p +
        ggplot2::geom_violin(data = approx_long,
                             ggplot2::aes(y = boot, x = component),
                             fill = "grey40", alpha = violin_alpha, colour = NA, width = 0.8,
                             inherit.aes = FALSE)
    }
  }

  if (isTRUE(show_box)) {
    p <- p +
      ggplot2::geom_boxplot(width = box_width, outlier.shape = NA, fill = NA)
  }

  # Mean ± CI from bootstrap
  if (isTRUE(show_ci)) {
    p <- p +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = lwr, ymax = upr),
                             width = errorbar_width) +
      ggplot2::geom_point(size = point_size)
  } else {
    p <- p + ggplot2::geom_point(size = point_size)
  }

  # Observed correlation on original test set (open ring)
  if (isTRUE(show_observed)) {
    p <- p + ggplot2::geom_point(ggplot2::aes(y = obs),
                   shape = 4, colour = "red", 
                   size = point_size + 2, stroke = 1.2)
  }

  # P-value labels above the higher of (mean, obs)
  if (isTRUE(show_pvalue)) {
    # Place label above the CI upper whisker for readability
    df$y_lab <- df$upr
    df$lab   <- sprintf("p = %s",
                        ifelse(is.finite(df$pval),
                               formatC(df$pval, format = "f", digits = 3), "NA"))
    p <- p + ggplot2::geom_text(
      data = df,
      ggplot2::aes(x = component, y = y_lab, label = lab),
      inherit.aes = FALSE,
      vjust = -0.8, size = 3.2
    )
  }

  p +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.02, 0.18))) +
    ggplot2::labs(
      title    = "Bootstrap validation (component-wise)",
      subtitle = sprintf("Statistic: %s • %.0f%% CI • n_boot = %d",
                         tolower(perf), df$conf[1] * 100, df$nboot[1]),
      x = NULL,
      y = "Latent correlation (MAC/Frobenius)",
      caption = "Red cross = observed correlation on original test data; filled dot = bootstrap mean"
    ) +
    ggplot2::theme_minimal(base_size = 11)
}
