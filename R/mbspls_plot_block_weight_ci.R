#' Plot LC block weights with 95% CIs from models or a single model's bootstraps
#'
#' @param blocks named list: per-block character vectors of feature names
#' @param models either a list of mbspls models (for source="weights"),
#'               or a single model (or list of length 1) with $weights_boot_draws
#'               (for source="bootstrap")
#' @param component integer: LC to plot (e.g., 1)
#' @param source one of c("weights","bootstrap"):
#'               "weights"  -> aggregate across the supplied models' weights;
#'               "bootstrap"-> aggregate across bootstrap replicates of one model
#' @param ci_filter one of c("none","excludes_zero","overlaps_zero").
#'                  "excludes_zero" keeps only features whose 95% CI does not cross 0.
#' @param top_n integer: show only top n features per block (by absolute weight). 
#'              If NULL (default), show all features.
#' @param add_block_rule logical: draw a thin rule between block facets (default FALSE;
#'                       turn on only if needed; some ggplot2 versions are picky here)
#' @param font character: font family for the plot (default "Sans")
#' @param alpha_by_stability logical (bootstrap only): if TRUE, use
#'   model$weights_selectfreq$freq for the requested component as bar alpha;
#'   ignored for source="weights"
#' @return a ggplot object
#' 
#' @export 
mbspls_plot_block_weight_ci <- function(
  blocks,
  models,
  component = 1,
  source = c("weights", "bootstrap"),
  ci_filter = c("none", "excludes_zero", "overlaps_zero"),
  top_n = NULL,
  add_block_rule = TRUE,
  font = "sans",
  alpha_by_stability = TRUE
) {
  requireNamespace("ggplot2")
  requireNamespace("dplyr")
  requireNamespace("tibble")
  requireNamespace("stringr")
  requireNamespace("RColorBrewer")
  requireNamespace("grid")

  source    <- match.arg(source)
  ci_filter <- match.arg(ci_filter)
  comp_lab  <- sprintf("LC_%02d", component)

  ci_filter_label <- switch(ci_filter,
    none = "No CI filtering",
    excludes_zero = "Non-Zero CIs",
    overlaps_zero = "Zero-Overlap Only"
  )

  # ---------------- helpers ----------------
  .nice_label <- function(x) gsub("_", " ", x, fixed = TRUE)
  .wrap       <- function(x, width = 24) stringr::str_wrap(x, width = width)
  .base_size_from_n <- function(n) max(7, min(14, 13 - 0.07 * (n - 20)))

  block_levels <- names(blocks)
  if (is.null(block_levels) || !length(block_levels))
    stop("`blocks` must be a *named* list of character vectors.")

  mean_block_weights_LC <- list()
  sd_block_weights_LC   <- list()
  freq_tbl              <- NULL

  # ------------- source = 'weights' (across models) -------------
  if (source == "weights") {
    is_single_model <- is.list(models) && !is.null(models$weights)
    model_list <- if (is_single_model) list(models) else models
    if (!length(model_list)) stop("No model(s) supplied.")
    if (isTRUE(alpha_by_stability)) {
      message("alpha_by_stability is only available for source='bootstrap'; ignoring.")
    }

    block_mats <- lapply(block_levels, function(b) {
      cols <- lapply(seq_along(model_list), function(i) {
        wi <- model_list[[i]]$weights[[comp_lab]][[b]]
        if (is.null(wi)) stop("Block '", b, "' not found in model ", i, ".")
        wi <- wi[blocks[[b]]]
        wi[is.na(wi)] <- 0
        as.numeric(wi)
      })
      mat <- do.call(cbind, cols)
      rownames(mat) <- blocks[[b]]
      mat
    })
    names(block_mats) <- block_levels

    # Sign-align columns within each block to the first model
    block_mats <- lapply(block_mats, function(mat) {
      if (is.null(dim(mat))) mat <- matrix(mat, ncol = 1)
      if (ncol(mat) > 1) {
        ref <- mat[, 1]
        for (j in 2:ncol(mat)) {
          cc <- suppressWarnings(stats::cor(ref, mat[, j], use = "complete.obs"))
          if (is.finite(cc) && cc < 0) mat[, j] <- -mat[, j]
        }
      }
      mat
    })

    mean_block_weights_LC <- lapply(block_mats, function(mat) rowMeans(mat, na.rm = TRUE))
    sd_block_weights_LC   <- lapply(block_mats, function(mat) apply(mat, 1, stats::sd, na.rm = TRUE))

  # ------------- source = 'bootstrap' (one model) -------------
  } else {
    model_obj <- if (is.list(models) && !is.null(models$weights)) {
      models
    } else if (is.list(models) && length(models) == 1) {
      models[[1]]
    } else models

    if (is.null(model_obj$weights_boot_draws))
      stop("For source='bootstrap', provide a single model with $weights_boot_draws.")

    # Optional selection frequency table for alpha mapping
    if (isTRUE(alpha_by_stability)) {
      freq_tbl <- model_obj$weights_selectfreq
      if (!is.null(freq_tbl)) {
        if (inherits(freq_tbl, "data.table")) freq_tbl <- as.data.frame(freq_tbl)
        if (all(c("component","block","feature","freq") %in% names(freq_tbl))) {
          freq_tbl <- freq_tbl[freq_tbl$component == comp_lab,
                               c("block","feature","freq"), drop = FALSE]
        } else {
          freq_tbl <- NULL
        }
      }
    }

    boot_df <- as.data.frame(model_obj$weights_boot_draws)
    boot_df <- boot_df[boot_df$component == comp_lab & boot_df$block %in% block_levels, , drop = FALSE]
    if (!nrow(boot_df))
      stop("No bootstrap draws for ", comp_lab, " in the supplied model.")

    # Reference weights for per-replicate sign alignment
    ref_tbl <- dplyr::bind_rows(lapply(block_levels, function(b) {
      w <- model_obj$weights[[comp_lab]][[b]]
      if (is.null(w)) stop("Block '", b, "' not found in model weights for ", comp_lab, ".")
      tibble::tibble(block = b, feature = names(w), refw = as.numeric(w))
    }))
    boot_df <- dplyr::left_join(boot_df, ref_tbl, by = c("block", "feature"))

    sign_df <- boot_df |>
      dplyr::group_by(replicate, block) |>
      dplyr::summarise(
        corr = suppressWarnings(stats::cor(weight, refw, use = "complete.obs")),
        .groups = "drop"
      ) |>
      dplyr::mutate(sign = dplyr::if_else(is.na(corr) | corr >= 0, 1, -1))

    boot_df <- dplyr::left_join(
      boot_df, sign_df[, c("replicate", "block", "sign")],
      by = c("replicate", "block")
    ) |>
      dplyr::mutate(weight_aligned = sign * weight)
    
    boot_summary <- boot_df |>
      dplyr::group_by(block, feature) |>
      dplyr::summarise(
        mean   = mean(weight_aligned, na.rm = TRUE),
        q025   = stats::quantile(weight_aligned, 0.025, na.rm = TRUE, names = FALSE),
        q975   = stats::quantile(weight_aligned, 0.975, na.rm = TRUE, names = FALSE),
        .groups = "drop"
      )

    for (b in block_levels) {
      feats <- blocks[[b]]
      sb <- boot_summary[boot_summary$block == b, , drop = FALSE]
      mu_map <- setNames(sb$mean, sb$feature)
      q025_map <- setNames(sb$q025, sb$feature)
      q975_map <- setNames(sb$q975, sb$feature)
      mu    <- unname(mu_map[feats]); mu[is.na(mu)]       <- 0
      q025v <- unname(q025_map[feats]); q025v[is.na(q025v)] <- 0
      q975v <- unname(q975_map[feats]); q975v[is.na(q975v)] <- 0
      mean_block_weights_LC[[b]] <- mu
      sd_block_weights_LC[[b]]   <- list(q025 = q025v, q975 = q975v)
    }
  }

  # ---------------- build long df ----------------
  df <- dplyr::bind_rows(lapply(block_levels, function(b) {
  feats <- blocks[[b]]
  mu    <- mean_block_weights_LC[[b]]
  quantiles <- sd_block_weights_LC[[b]]
  q025v <- quantiles$q025
  q975v <- quantiles$q975
  if (length(mu) != length(feats) || length(q025v) != length(feats) || length(q975v) != length(feats)) {
    stop(sprintf("Length mismatch in block '%s': %d features, %d means, %d q025, %d q975.",
           b, length(feats), length(mu), length(q025v), length(q975v)))
  }
  tibble::tibble(block = b, feature = feats, mean = as.numeric(mu), q025 = as.numeric(q025v), q975 = as.numeric(q975v))
  }))

  df <- df |>
  dplyr::mutate(
    ci_low  = q025,
    ci_high = q975,
    signpos = mean >= 0,
    abs_m   = abs(mean)
  )

  # Apply top_n filtering per block
  if (!is.null(top_n) && is.numeric(top_n) && top_n > 0) {
    df <- df |>
      dplyr::group_by(block) |>
      dplyr::slice_max(abs_m, n = top_n, with_ties = FALSE) |>
      dplyr::ungroup()
  }

  # Attach stability-based alpha (bootstrap only)
  if (isTRUE(alpha_by_stability) && !is.null(freq_tbl) && nrow(freq_tbl)) {
    df <- dplyr::left_join(df, freq_tbl, by = c("block","feature"))
    df$alpha_freq <- df$freq
    df$alpha_freq[!is.finite(df$alpha_freq)] <- 1
    df$alpha_freq <- pmax(pmin(df$alpha_freq, 1), 0)
  } else {
    df$alpha_freq <- 1
  }

  # CI filters
  if (ci_filter == "excludes_zero") {
    df <- dplyr::filter(df, ci_low >= 0 | ci_high <= 0)
    # filter all 0s
    df <- dplyr::filter(df, abs_m > 1e-3)
  } else if (ci_filter == "overlaps_zero") {
    df <- dplyr::filter(df, ci_low <= 0 & ci_high >= 0)
  }
  if (!nrow(df)) stop("No features passed the '", ci_filter_label, "' filter.")

  # Ordering & labels (after filtering)
  df <- df |>
    dplyr::arrange(match(block, block_levels), abs_m) |>
    dplyr::mutate(
      axis_id       = paste(block, feature, sep = "___"),
      block_lab     = factor(.nice_label(block), levels = .nice_label(block_levels)),
      component_lab = factor(sprintf("LC %d", component), levels = sprintf("LC %d", component))
    )
  df$axis_id   <- factor(df$axis_id, levels = unique(df$axis_id))
  df$block_lab <- droplevels(df$block_lab)
  lab_map <- setNames(.wrap(df$feature, 24), as.character(df$axis_id))

  # Optional separator data (disabled by default to avoid PANEL issues)
  sep_layer <- NULL
  if (isTRUE(add_block_rule)) {
    block_lvls <- levels(df$block_lab)
    sep_df <- if (length(block_lvls) > 1L) {
      data.frame(
        block_lab     = factor(block_lvls[-length(block_lvls)], levels = block_lvls),
        component_lab = factor(sprintf("LC %d", component), levels = sprintf("LC %d", component)),
        xintercept    = -Inf
      )
    } else data.frame(block_lab = factor(), component_lab = factor(), xintercept = numeric())

    if (nrow(sep_df)) {
      sep_layer <- ggplot2::geom_vline(
        data    = sep_df,
        mapping = ggplot2::aes(xintercept = xintercept),
        colour  = "grey40",
        linewidth = 0.8,
        inherit.aes = FALSE
      )
    }
  }

  # Colors & base size
  pal     <- RColorBrewer::brewer.pal(3, "Dark2")
  col_pos <- pal[1]
  col_neg <- pal[3]
  base_sz <- .base_size_from_n(nrow(df))

  # ---------------- plot ----------------
  subtitle_text <- paste0(
    if (source == "weights") "Across models" else "Across bootstrap replicates",
    " · 95% Bootstrap Intervall",
    if (!is.null(top_n)) paste0(" · Top ", top_n, " features per block") else "",
    if (ci_filter != "none") paste0(" · Filter: ", ci_filter_label) else ""
  )

  p <- ggplot2::ggplot(df, ggplot2::aes(x = axis_id, y = mean, fill = signpos, alpha = alpha_freq)) +
    ggplot2::geom_col(width = 0.85, show.legend = FALSE) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = ci_low, ymax = ci_high),
      width = 0.2, linewidth = 0.25, colour = "grey15"
    ) +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey70") +
    sep_layer +
    ggplot2::facet_grid(
      rows = ggplot2::vars(block_lab),
      cols = ggplot2::vars(component_lab),
      scales = "free_y",
      space  = "free_y",
      switch = "y"
    ) +
    ggplot2::scale_fill_manual(values = c(`TRUE` = col_pos, `FALSE` = col_neg)) +
    ggplot2::scale_alpha_identity() +
    ggplot2::scale_x_discrete(labels = function(x) lab_map[as.character(x)]) +
    ggplot2::coord_flip() +
    ggplot2::labs(
      x = NULL, y = "Weight",
      title = sprintf("Sparse weights per block - LC %d", component),
      subtitle = subtitle_text
    ) +
    ggplot2::theme_minimal(base_size = base_sz, base_family = font) +
    ggplot2::theme(
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.minor   = ggplot2::element_blank(),
      panel.spacing      = grid::unit(0, "pt"),
      strip.placement    = "outside",
      strip.background   = ggplot2::element_rect(fill = NA, colour = NA),
      strip.text.y.left  = ggplot2::element_text(angle = 0, face = "bold"),
      strip.text.x       = ggplot2::element_text(face = "bold"),
      axis.text.y        = ggplot2::element_text(size = base_sz * 0.8)
    )

  return(p)
}
