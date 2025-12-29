#' Plot LC block weights with 95% CIs from a GraphLearner (or legacy inputs)
#'
#' @param x Either:
#'   - a trained GraphLearner built with `mbspls_graph_learner()` (recommended), or
#'   - a list like `list(mbspls = <pipeop or state>, mbspls_bootstrap_select = <pipeop or state>)`,
#'     or your previous `c(glearner$model$mbspls, glearner$model$mbspls_bootstrap_select)`.
#' @param source One of c("weights","bootstrap").
#'   * "bootstrap": uses aligned summaries from selection state (`weights_ci` + `weights_selectfreq`).
#'   * "weights"  : aggregates across multiple MB-sPLS fits (means + Wald CIs).
#' @param ci_filter One of c("none","excludes_zero","overlaps_zero").
#'   * "excludes_zero": keep if (ci_low >= 0 | ci_high <= 0) AND |mean| > 1e-3.
#' @param top_n Integer or NULL. Keep top-N features per blockxcomponent by |mean|.
#' @param add_block_rule Logical; thin rule between block facets (default FALSE; safe implementation).
#' @param font Character; font family (default "sans").
#' @param alpha_by_stability Logical; for source="bootstrap", map bar alpha to selection frequency.
#'
#' @return ggplot object (all components facetted in columns)
#' @export
mbspls_plot_block_weight_ci = function(
  x,
  source = c("weights", "bootstrap"),
  ci_filter = c("none", "excludes_zero", "overlaps_zero"),
  top_n = NULL,
  add_block_rule = TRUE, # now FALSE by default to avoid separator pitfalls
  font = "sans",
  alpha_by_stability = TRUE
) {
  source = match.arg(source)
  ci_filter = match.arg(ci_filter)

  requireNamespace("ggplot2")
  requireNamespace("dplyr")
  requireNamespace("tibble")
  requireNamespace("stringr")
  requireNamespace("RColorBrewer")
  requireNamespace("grid")

  `%||%` = function(x, y) if (is.null(x)) y else x
  .nice = function(s) gsub("_", " ", s, fixed = TRUE)
  .wrap = function(s, w = 24) stringr::str_wrap(s, width = w)
  .q = function(v, p) stats::quantile(v, probs = p, na.rm = TRUE, names = FALSE)
  .base_size_from_n = function(n) max(7, min(14, 13 - 0.07 * (n - 20)))

  # -- extract states ----------------------------------------------------------
  .get_state = function(obj) if (!is.null(obj$state)) obj$state else obj
  .blocks_named = function(b) {
    if (is.list(b) && !is.null(names(b))) {
      b
    } else {
      stop("Blocks mapping not found; ensure 'x' is a trained GraphLearner or a state list.")
    }
  }

  st_fit = NULL
  st_sel = NULL
  if (inherits(x, "GraphLearner")) {
    mdl = x$model
    if (is.null(mdl)) stop("GraphLearner is untrained (model is NULL).")
    po_fit = mdl$mbspls %||% (x$graph$pipeops$mbspls)
    po_sel = mdl$mbspls_bootstrap_select %||% (x$graph$pipeops$mbspls_bootstrap_select)
    if (is.null(po_fit)) stop("Cannot locate 'mbspls' in the graph/model.")
    st_fit = .get_state(po_fit)
    if (source == "bootstrap") {
      if (is.null(po_sel)) stop("Cannot locate 'mbspls_bootstrap_select' in the model for bootstrap plotting.")
      st_sel = .get_state(po_sel)
    }
  } else if (is.list(x)) {
    if (!is.null(x$mbspls)) st_fit <- .get_state(x$mbspls)
    if (!is.null(x$mbspls_bootstrap_select)) st_sel <- .get_state(x$mbspls_bootstrap_select)
    if (is.null(st_fit) && length(x) >= 1L) st_fit <- .get_state(x[[1]])
    if (is.null(st_sel) && length(x) >= 2L && source == "bootstrap") st_sel <- .get_state(x[[2]])
    if (is.null(st_fit)) stop("Could not extract MB-sPLS state from the provided list.")
    if (is.null(st_sel) && source == "bootstrap") {
      stop("Could not extract bootstrap-select state from the provided list.")
    }
  } else {
    stop("Unsupported input 'x'.")
  }

  blocks = .blocks_named(st_fit$blocks)
  block_levels = names(blocks)

  # -- build long table across ALL components ----------------------------------
  df = NULL
  freq_tbl = NULL
  align_tag = NULL
  comp_levels = sprintf("LC_%02d", seq_len(st_fit$ncomp %||% length(st_fit$weights) %||% 1L))

  if (source == "bootstrap") {
    ci_tbl = st_sel$weights_ci
    if (!is.null(ci_tbl) && nrow(ci_tbl)) {
      ci_df = as.data.frame(ci_tbl)
      # ensure character, not factors
      ci_df$component = as.character(ci_df$component)
      ci_df$block = as.character(ci_df$block)
      ci_df$feature = as.character(ci_df$feature)
      # limit to known blocks; keep components present in table
      ci_df = ci_df[ci_df$block %in% block_levels, , drop = FALSE]
      present_comp = intersect(unique(ci_df$component), comp_levels)
      if (!length(present_comp)) stop("No bootstrap CI rows for any component in selection state.")
      comp_levels = present_comp

      # grid (component x block x features) to fill missing rows with zeros
      grid_df = dplyr::bind_rows(lapply(comp_levels, function(k) {
        dplyr::bind_rows(lapply(block_levels, function(b) {
          tibble::tibble(component = k, block = b, feature = blocks[[b]])
        }))
      }))

      ci_df2 = dplyr::left_join(grid_df, ci_df,
        by = c("component", "block", "feature"))

      repl0 = function(z) {
        z[is.na(z)] = 0
        z
      }
      df = tibble::tibble(
        component = ci_df2$component,
        block     = ci_df2$block,
        feature   = ci_df2$feature,
        mean      = as.numeric(repl0(ci_df2$boot_mean)),
        ci_low    = as.numeric(repl0(ci_df2$ci_lower)),
        ci_high   = as.numeric(repl0(ci_df2$ci_upper))
      )

      if (isTRUE(alpha_by_stability) && !is.null(st_sel$weights_selectfreq)) {
        freq_tbl = as.data.frame(st_sel$weights_selectfreq)
        freq_tbl$component = as.character(freq_tbl$component)
        freq_tbl = freq_tbl[freq_tbl$block %in% block_levels &
          freq_tbl$component %in% comp_levels,
        c("component", "block", "feature", "freq"), drop = FALSE]
      }
      align_tag = st_sel$alignment_method %||% NULL

    } else {
      # fallback to draws if summaries not present
      d = st_sel$weights_boot_draws
      if (is.null(d) || !nrow(d)) {
        stop("No 'weights_ci' or 'weights_boot_draws' found for bootstrap plotting.")
      }
      d = as.data.frame(d)
      d$component = as.character(d$component)
      d$block = as.character(d$block)
      d$feature = as.character(d$feature)
      d = d[d$block %in% block_levels, , drop = FALSE]
      comp_levels = intersect(unique(d$component), comp_levels)
      if (!length(comp_levels)) stop("No bootstrap draws for any component in the supplied model.")

      df = d |>
        dplyr::group_by(component, block, feature) |>
        dplyr::summarise(
          mean = mean(weight, na.rm = TRUE),
          ci_low = .q(weight, 0.025),
          ci_high = .q(weight, 0.975),
          .groups = "drop"
        )
      if (isTRUE(alpha_by_stability) && !is.null(st_sel$weights_selectfreq)) {
        freq_tbl = as.data.frame(st_sel$weights_selectfreq)
        freq_tbl$component = as.character(freq_tbl$component)
        freq_tbl = freq_tbl[freq_tbl$block %in% block_levels &
          freq_tbl$component %in% comp_levels,
        c("component", "block", "feature", "freq"), drop = FALSE]
      }
      align_tag = st_sel$alignment_method %||% "aligned_draws"
    }

  } else { # source == "weights"
    # Aggregate across a list of models; if only one, show mean with NA CI
    get_one_weight_vec = function(st_like, k_lab, b) {
      w = st_like$weights[[k_lab]][[b]]
      if (is.null(w)) {
        stats::setNames(rep(0, length(blocks[[b]])), blocks[[b]])
      } else {
        w = w[blocks[[b]]]
        w[is.na(w)] = 0
        w
      }
    }

    model_list = NULL
    if (inherits(x, "GraphLearner")) {
      model_list = list(st_fit)
    } else if (is.list(x)) {
      grab_fit = function(elem) {
        if (inherits(elem, "GraphLearner")) {
          mdl = elem$model
          po_fit = mdl$mbspls %||% elem$graph$pipeops$mbspls
          .get_state(po_fit)
        } else if (!is.null(elem$weights)) {
          elem
        } else if (!is.null(elem$mbspls)) {
          .get_state(elem$mbspls)
        } else {
          elem
        }
      }
      model_list = lapply(x, grab_fit)
    }
    if (is.null(model_list) || !length(model_list)) {
      stop("No MB-sPLS model(s) provided for source='weights'.")
    }

    comp_levels = sprintf("LC_%02d", seq_len(st_fit$ncomp %||% length(st_fit$weights) %||% 1L))

    df = dplyr::bind_rows(lapply(comp_levels, function(k_lab) {
      dplyr::bind_rows(lapply(block_levels, function(b) {
        cols = lapply(model_list, function(sfi) get_one_weight_vec(sfi, k_lab, b))
        mat = do.call(cbind, lapply(cols, as.numeric))
        rownames(mat) = names(cols[[1]])
        if (ncol(mat) > 1) { # sign-align columns to the first
          ref = mat[, 1]
          for (j in 2:ncol(mat)) {
            cc = suppressWarnings(stats::cor(ref, mat[, j], use = "complete.obs"))
            if (is.finite(cc) && cc < 0) mat[, j] <- -mat[, j]
          }
        }
        mu = rowMeans(mat, na.rm = TRUE)
        sdv = apply(mat, 1, stats::sd, na.rm = TRUE)
        n = ncol(mat)
        half = if (n > 1) 1.96 * sdv / sqrt(n) else NA_real_
        tibble::tibble(component = k_lab, block = b, feature = rownames(mat),
          mean = as.numeric(mu),
          ci_low = as.numeric(mu - half),
          ci_high = as.numeric(mu + half))
      }))
    }))
    freq_tbl = NULL
    align_tag = "across_models"
  }

  # -- attach stability alpha (bootstrap only) ---------------------------------
  if (isTRUE(alpha_by_stability) && !is.null(freq_tbl) && nrow(freq_tbl)) {
    df = dplyr::left_join(df, freq_tbl, by = c("component", "block", "feature"))
    df$alpha_freq = df$freq
    df$alpha_freq[!is.finite(df$alpha_freq)] = 1
    df$alpha_freq = pmax(pmin(df$alpha_freq, 1), 0)
  } else {
    df$alpha_freq = 1
  }

  # -- CI filter (exact rule for excludes_zero) --------------------------------
  df = df |>
    dplyr::mutate(abs_m = abs(mean))

  if (ci_filter == "excludes_zero") {
    df = df |>
      dplyr::filter(ci_low >= 0 | ci_high <= 0) |>
      dplyr::filter(abs_m > 1e-3)
  } else if (ci_filter == "overlaps_zero") {
    df = df |>
      dplyr::filter(ci_low <= 0 & ci_high >= 0)
  }
  if (!nrow(df)) stop("No features passed the '", ci_filter, "' filter.")

  # -- top-N per blockxcomponent ----------------------------------------------
  if (!is.null(top_n) && is.numeric(top_n) && top_n > 0) {
    df = df |>
      dplyr::group_by(component, block) |>
      dplyr::slice_max(abs_m, n = top_n, with_ties = FALSE) |>
      dplyr::ungroup()
  }

  # keep only components that still have data
  comp_levels = intersect(comp_levels, unique(df$component))
  if (!length(comp_levels)) stop("No components left after filtering.")

  # -- ordering & labels -------------------------------------------------------
  df = df |>
    dplyr::mutate(
      signpos = mean >= 0,
      block_lab = factor(.nice(block), levels = .nice(block_levels)),
      component_lab = factor(gsub("^LC_0?", "LC ", component),
        levels = gsub("^LC_0?", "LC ", comp_levels)),
      axis_id = paste(block, feature, sep = "___")
    ) |>
    dplyr::arrange(match(block, block_levels), match(component, comp_levels), abs_m)

  df$axis_id = factor(df$axis_id, levels = unique(df$axis_id))
  lab_map = setNames(.wrap(as.character(df$feature), 24), as.character(df$axis_id))

  # -- optional safe separator (no -Inf) ---------------------------------------
  sep_layer = NULL
  if (isTRUE(add_block_rule)) {
    # draw a vertical line at the *left edge* of each block panel using annotate
    # (safe on discrete axes + coord_flip). We simply draw a thin segment at x = 0.5.
    sep_layer = ggplot2::annotate("segment",
      x = 0.5, xend = 0.5, y = -Inf, yend = Inf,
      colour = "grey70", linewidth = 0.3
    )
  }

  pal = RColorBrewer::brewer.pal(3, "Dark2")
  col_pos = pal[1]
  col_neg = pal[3]
  base_sz = .base_size_from_n(nrow(df))

  subtitle_bits = c(
    if (source == "weights") "Across models" else "Across bootstrap replicates",
    "95% CI",
    if (!is.null(top_n)) sprintf("Top %d per blockxcomponent", as.integer(top_n)) else NULL,
    if (ci_filter != "none") {
      switch(ci_filter,
        excludes_zero = "Filter: CI excludes 0 & |mean|>1e-3",
        overlaps_zero = "Filter: CI overlaps 0",
        "Filter: none")
    },
    if (!is.null(align_tag) && source == "bootstrap") paste0("Aligned: ", align_tag)
  )
  subtitle_text = paste(subtitle_bits, collapse = " . ")

  ggplot2::ggplot(df, ggplot2::aes(x = axis_id, y = mean, fill = signpos, alpha = alpha_freq)) +
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
      space = "free_y",
      switch = "y"
    ) +
    ggplot2::scale_fill_manual(values = c(`TRUE` = col_pos, `FALSE` = col_neg)) +
    ggplot2::scale_alpha_identity() +
    ggplot2::scale_x_discrete(labels = function(x) lab_map[as.character(x)]) +
    ggplot2::coord_flip(clip = "off") +
    ggplot2::labs(
      x = NULL, y = "Weight",
      title = "MB-sPLS sparse weights per block",
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
}
