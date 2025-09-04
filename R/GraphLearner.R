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
#'     \item \code{"mbspls_network"} – LC network for edges with \code{|r| >= cutoff}. Args: \code{cutoff}, \code{method}.
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
    mbspls_weights = do.call(.mbspls_plot_weights_from_model,
                      c(list(model = mod, gl = object), dots)),
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

# Returns a numeric vector of length ncomp with entries in {+1, -1},
# or NULL if no flip operator/model is found.
.mbspls_get_flip_signs <- function(gl, flip_id = NULL) {
  if (!inherits(gl, "GraphLearner")) return(NULL)
  if (is.null(flip_id)) {
    ids <- names(gl$graph$pipeops)
    hit <- ids[vapply(gl$graph$pipeops,
                      function(po) inherits(po, "PipeOpMBsPLSFlipWeights"),
                      logical(1))]
    flip_id <- if (length(hit)) hit[1] else NULL
  }
  if (is.null(flip_id)) return(NULL)
  node_model <- tryCatch(gl$model[[flip_id]], error = function(e) NULL)
  if (is.null(node_model)) return(NULL)
  s <- tryCatch(node_model$signs, error = function(e) NULL)
  if (is.null(s)) return(NULL)
  as.numeric(s)
}

.mbspls_align_signs <- function(signs, ncomp) {
  if (is.null(signs)) return(rep(1, ncomp))
  if (!is.null(names(signs))) {
    s <- sapply(seq_len(ncomp), function(k) signs[[sprintf("LC_%02d", k)]] %||% 1)
  } else {
    s <- as.numeric(signs)
    if (length(s) < ncomp) s <- c(s, rep(1, ncomp - length(s)))
    s <- s[seq_len(ncomp)]
  }
  s[!is.finite(s)] <- 1
  s
}

.pal_vals <- function(palette) {
  pal <- RColorBrewer::brewer.pal(3, palette)
  c(`TRUE` = pal[1],  # positive (green-ish in Dark2)
    `FALSE`= pal[3])  # negative (purple-ish in Dark2)
}

.base_size_from_n <- function(n) {
  # scale gently with number of features shown; clamp to [7, 14]
  bs <- 13 - 0.07 * (n - 20)
  max(7, min(14, bs))
}

.nice_label <- function(x) gsub("_", " ", x, fixed = TRUE)

#' @importFrom rlang .data
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
  weights_source = c("raw", "stability_filtered"),
  font = "sans",
  # NEW: flip support & GL context
  gl = NULL,
  flip_signs = NULL,
  flip_id = NULL
) {
  requireNamespace("ggplot2")
  requireNamespace("dplyr")
  requireNamespace("tibble")
  requireNamespace("RColorBrewer")
  requireNamespace("stringr")
  has_patchwork <- requireNamespace("patchwork", quietly = TRUE)

  weights_view   <- match.arg(weights_view)
  weights_source <- match.arg(weights_source)

  # optional selection freq as data.frame
  .freq_tbl <- {
    sf <- model$weights_selectfreq
    if (!is.null(sf) && requireNamespace("data.table", quietly = TRUE) && data.table::is.data.table(sf)) {
      sf <- as.data.frame(sf)
    }
    if (!is.null(sf) && !nrow(sf)) sf <- NULL
    sf
  }

  # choose which weights to plot
  W_use <- model$weights
  if (weights_source == "stability_filtered" && !is.null(model$weights_stability_filtered)) {
    W_use <- model$weights_stability_filtered
  }

  # ---------- NEW: apply flip signs to weights & boot draws ----------
  flip_signs <- flip_signs %||% .mbspls_get_flip_signs(gl, flip_id)
  S <- .mbspls_align_signs(flip_signs, ncomp = length(W_use))
  if (any(S != 1)) {
    for (k in seq_along(W_use)) if (S[k] != 1) {
      for (b in names(W_use[[k]])) W_use[[k]][[b]] <- S[k] * W_use[[k]][[b]]
    }
  }
  # -------------------------------------------------------------------

  block_levels <- names(model$blocks)
  comp_levels  <- paste0("LC_", sprintf("%02d", seq_len(model$ncomp)))

  # Fallback names for unnamed weight vectors (prevents 'feature' not found)
  .fallback_featnames <- function(block_name, w_vec) {
    nms <- names(w_vec)
    if (!is.null(nms)) return(nms)
    # try model$blocks mapping first:
    fb <- model$blocks[[block_name]]
    if (!is.null(fb) && length(fb) == length(w_vec)) return(fb)
    # last resort: synthetic names
    paste0(block_name, "_", seq_along(w_vec))
  }

  build_weights_long <- function() {
    tibble::as_tibble(dplyr::bind_rows(lapply(seq_along(W_use), function(k) {
      dplyr::bind_rows(lapply(names(W_use[[k]]), function(b) {
        wv <- W_use[[k]][[b]]
        tibble::tibble(
          component = sprintf("LC_%02d", k),
          block     = b,
          feature   = .fallback_featnames(b, wv),   # NEW: always present
          weight    = as.numeric(wv)
        )
      }))
    })))
  }

  build_boot_long <- function(vb) {
    dplyr::bind_rows(lapply(seq_along(vb), function(k) {
      mult <- S[k] %||% 1   # NEW: flip bootstrap draws as well
      dplyr::bind_rows(lapply(names(vb[[k]]), function(b) {
        fb <- vb[[k]][[b]]
        dplyr::bind_rows(lapply(names(fb), function(f) {
          tibble::tibble(
            component = sprintf("LC_%02d", k),
            block     = b,
            feature   = f,
            value     = mult * as.numeric(fb[[f]])
          )
        }))
      }))
    }))
  }

  # Plot core (weights view)
  plot_one_lc_weights <- function(df_lc, comp_label) {
    # join stability freq (optional)
    if (!is.null(.freq_tbl)) {
      df_lc <- dplyr::left_join(
        df_lc,
        .freq_tbl[, c("component","block","feature","freq")],
        by = c("component", "block", "feature")
      )
    }
    if (!"freq" %in% names(df_lc)) df_lc$freq <- NA_real_

    # order columns & compute abs
    df_lc <- df_lc |>
      dplyr::mutate(abs_w = abs(.data$weight)) |>
      dplyr::filter(.data$abs_w > 0)

    if (!nrow(df_lc)) stop("No non-zero weights to plot (after filtering).")

    df_lc$block     <- factor(df_lc$block,     levels = block_levels)
    df_lc$component <- factor(df_lc$component, levels = comp_levels)

    df_lc <- df_lc |>
      dplyr::arrange(.data$block, .data$abs_w) |>
      dplyr::mutate(
        feature_label = stringr::str_wrap(.data$feature, width = label_width),  # NEW: .data
        axis_id       = paste(.data$block, .data$feature, sep = "___")
      )
    df_lc$axis_id <- factor(df_lc$axis_id, levels = unique(df_lc$axis_id))
    lab_map <- setNames(as.character(df_lc$feature_label), as.character(df_lc$axis_id))

    # alpha by stability
    if (isTRUE(alpha_by_stability)) {
      af <- df_lc$freq; af[!is.finite(af)] <- 1
      df_lc$alpha_freq <- pmax(pmin(af, 1), 0)
    } else df_lc$alpha_freq <- 1

    # pretty facet labels
    .nice_label <- function(x) gsub("_", " ", x, fixed = TRUE)
    df_lc$block_lab <- factor(.nice_label(as.character(df_lc$block)),
                              levels = .nice_label(block_levels))
    comp_lab        <- .nice_label(as.character(comp_label))
    df_lc$component_lab <- factor(comp_lab, levels = comp_lab)

    # separators
    block_lvls <- levels(df_lc$block_lab)
    sep_df <- if (length(block_lvls) > 1L) {
      expand.grid(
        block_lab     = factor(block_lvls[-length(block_lvls)], levels = block_lvls),
        component_lab = factor(comp_lab, levels = comp_lab),
        KEEP.OUT.ATTRS = FALSE
      )
    } else data.frame(block_lab = factor(), component_lab = factor())

    # font scaling
    .base_size_from_n <- function(n) max(7, min(14, 13 - 0.07 * (n - 20)))
    base_size <- .base_size_from_n(nrow(df_lc))

    ggplot2::ggplot(
      df_lc,
      ggplot2::aes(.data$axis_id, .data$weight, fill = .data$weight > 0, alpha = .data$alpha_freq)
    ) +
      ggplot2::geom_col(width = 0.85, show.legend = FALSE) +
      ggplot2::scale_alpha_identity() +
      ggplot2::geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey70") +
      (if (nrow(sep_df)) ggplot2::geom_vline(
         data    = sep_df,
         mapping = ggplot2::aes(xintercept = -Inf),
         colour  = sep_color, linewidth = sep_size, inherit.aes = FALSE
       ) else NULL) +
      ggplot2::facet_grid(
        rows = ggplot2::vars(block_lab),
        cols = ggplot2::vars(component_lab),
        scales = "free_y",
        space  = "free_y",
        switch = "y",
        labeller = ggplot2::labeller(
          block_lab = function(x) stringr::str_wrap(x, width = label_width)
        )
      ) +
      ggplot2::scale_fill_manual(values = {
        pal <- RColorBrewer::brewer.pal(3, palette)
        c(`TRUE` = pal[1], `FALSE` = pal[3])
      }) +
      ggplot2::scale_x_discrete(labels = function(x) lab_map[as.character(x)]) +
      ggplot2::coord_flip() +
      ggplot2::labs(x = NULL, y = "Weight",
                    title = sprintf("Sparse weights per block — %s", comp_lab)) +
      ggplot2::theme_minimal(base_size = base_size, base_family = font) +
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

  # ---- Branching: weights vs bootstrap ----
  if (weights_view == "weights") {
    long <- build_weights_long()
    if (!nrow(long)) stop("No weights available to plot.")
    if (!is.null(top_n)) {
      long <- long |>
        dplyr::group_by(.data$component, .data$block) |>
        dplyr::mutate(abs_w = abs(.data$weight)) |>
        dplyr::slice_max(.data$abs_w, n = max(1L, as.integer(top_n)), with_ties = FALSE) |>
        dplyr::ungroup()
    }
    comps <- unique(long$component)
    plots <- lapply(comps, function(comp) {
      plot_one_lc_weights(long[long$component == comp, , drop = FALSE], comp)
    })
    if (length(plots) == 1L || !has_patchwork) return(plots[[1L]])
    return(patchwork::wrap_plots(plots, nrow = 1))
  }

  # bootstrap view
  vb <- model$weights_boot_vectors
  if (is.null(vb)) {
    stop("weights_boot_vectors not found. Train with bootstrap_test=TRUE and boot_store_vectors=TRUE.")
  }
  boot_long <- build_boot_long(vb)
  if (is.function(summary_fun)) boot_long$value <- summary_fun(boot_long$value)

  comps <- unique(boot_long$component)
  plots <- lapply(comps, function(comp) {
    # reuse your existing 'plot_one_lc_boot' (unchanged), or inline same .data fixes
    plot_one_lc_boot(boot_long[boot_long$component == comp, , drop = FALSE], comp, font = font)
  })
  if (length(plots) == 1L || !has_patchwork) return(plots[[1L]])
  patchwork::wrap_plots(plots, nrow = 1)
}

.mbspls_plot_heatmap_from_model <- function(model, method = "spearman",
                                            T_override = NULL, title_suffix = "", font = "sans") {
  requireNamespace("ggplot2")
  scores <- T_override %||% model$T_mat
  if (is.null(scores) || NCOL(scores) < 2)
    stop("Need at least two latent variables to draw a heat map.", call. = FALSE)

  if (is.null(colnames(scores))) {
    colnames(scores) <- unlist(lapply(seq_len(model$ncomp), function(k)
      paste0("LC", sprintf("%02d", k), "_", names(model$blocks))))
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
    ggplot2::theme_minimal(base_family = font) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5, hjust = 1),
                   panel.grid  = ggplot2::element_blank()) +
    ggplot2::labs(title = sprintf("Cross-correlation of latent scores (%s)%s", method, title_suffix),
                  x = NULL, y = NULL, fill = "r")
}

.mbspls_plot_network_from_model <- function(model, cutoff = 0.3, method = "spearman",
                                            T_override = NULL, title_suffix = "", font = "sans") {
  requireNamespace("igraph"); requireNamespace("ggraph"); requireNamespace("ggplot2")

  scores <- T_override %||% model$T_mat
  if (is.null(scores) || NCOL(scores) < 2)
    stop("Need at least two latent variables to draw a network.", call. = FALSE)

  if (is.null(colnames(scores))) {
    colnames(scores) <- unlist(lapply(seq_len(model$ncomp), function(k)
      paste0("LV", sprintf("%02d", k), " ", names(model$blocks))))
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
    ggplot2::theme_void(base_family = font) +
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
  title_suffix = "",
  font = "sans"
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
    ggplot2::theme_minimal(base_size = 11, base_family = font) +
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
                                          obj_override = NULL, title_suffix = "", font = "sans") {
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
    ggplot2::theme_minimal(base_size = 11, base_family = font)
}


.mbspls_plot_scores_from_model <- function(model,
                                           component = 1,
                                           standardize = TRUE,
                                           density = c("none", "contour", "hex"),
                                           annotate = TRUE,
                                           T_override = NULL,
                                           title_suffix = "",
                                           font = "sans") {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("Package 'ggplot2' is required for this plot.")
  density <- match.arg(density)

  Tmat <- T_override %||% model$T_mat
  if (is.null(Tmat) || ncol(Tmat) < 2)
    stop("Need at least two latent variables to draw score plots.")
  if (component < 1 || component > model$ncomp)
    stop("`component` out of range.")

  lv_cols <- paste0("LC", component, "_", names(model$blocks))
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
        x = sprintf("Scores: LC%d (block)", component),
        y = sprintf("Scores: LC%d (block)", component),
        title = sprintf("MB-sPLS cross-block score agreement — LC%d%s", component, title_suffix),
        subtitle = "Z-scored per block; dashed: y = x; solid: LS fit; r = Pearson, ccc = Lin’s concordance"
      ) +
      ggplot2::theme_minimal(base_size = 11, base_family = font)
  } else {
    p +
      ggplot2::facet_grid(block_y ~ block_x, scales = "free") +
      ggplot2::labs(
        x = sprintf("Scores: LC%d (block)", component),
        y = sprintf("Scores: LC%d (block)", component),
        title = sprintf("MB-sPLS cross-block score agreement — LC%d%s", component, title_suffix),
        subtitle = "Dashed: y = x; solid: LS fit; r = Pearson, ccc = Lin’s concordance"
      ) +
      ggplot2::theme_minimal(base_size = 11, base_family = font)
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
  errorbar_width = 0.22,
  font = "sans"
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
  if (isTRUE(show_violin) && !is.null(pay$val_boot_vectors)) {
    # Add violin plot
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
    stop("No bootstrap samples available for violin plot.")
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
    ggplot2::theme_minimal(base_size = 11, base_family = font)
}
