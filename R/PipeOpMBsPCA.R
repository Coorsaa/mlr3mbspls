#' Multi-Block Sparse PCA (MB-sPCA) PipeOp
#'
#' @title Multi-Block Sparse PCA (MB-sPCA) Transformer
#'
#' @description
#' `PipeOpMBsPCA` extracts up to `ncomp` **sparse** principal components from
#' multiple, predefined feature blocks. Each component is estimated with a
#' GSMV-style algorithm (`cpp_mbspca_one_lv()`), and after extraction a
#' block-wise rank-1 deflation is applied so later components capture novel
#' structure. Optionally, a permutation test
#' (`perm_test_component_mbspca()`) can stop extraction early if a component
#' explains no more variance than expected under the null.
#'
#' The operator appends one column per *block × component* (name:
#' `PC<k>_<block>`) to the feature table.
#'
#' @section Input and Output:
#' * **Input**: [`Task`] with a table of numeric features (non-numeric or
#'   constant features inside a block are dropped automatically).
#' * **Output**: the input task with additional columns containing the latent
#'   scores for each block & component.
#'
#' @section State:
#' The `$state` stores:
#' * `blocks`: the (sanitised) named list of block → feature names used.
#' * `ncomp`: number of retained components.
#' * `weights`: list(`PCk` → list(`block` → numeric named vector)).
#' * `loadings`: same shape as `weights`, block-wise loadings used for deflation.
#' * `ev_block`: matrix `[components × blocks]` of variance explained by each PC
#'   within each block.
#' * `ev_comp`: numeric vector of total variance explained by each PC.
#' * `T_mat`: numeric matrix of appended latent scores (column names match the
#'   appended features).
#'
#' @section Parameters (ParamSet):
#' * `blocks` (`uty`, **required**; tag `"train"`): named list mapping block IDs
#'   to character vectors of feature names.
#' * `ncomp` (`int`, default `1`; tag `"train"`): number of components to target
#'   (may be shortened by permutation early-stopping).
#' * `permutation_test` (`lgl`, default `FALSE`; tag `"train"`): enable early-stop test.
#' * `n_perm` (`int`, default `500`; tag `"train"`): permutations for the test.
#' * `perm_alpha` (`dbl` in `[0,1]`, default `0.05`; tag `"train"`): test α.
#' * `c_<block>` (one `dbl` per block, lower `1`, upper `sqrt(#features in block)`,
#'   default `sqrt(#features)`; tags `c("train","tune")`): √(L¹-budget) for that block.
#' * `c_matrix` (`uty`, default `NULL`; tags `c("train","tune")`): optional
#'   matrix (`blocks × components`) overriding single-value `c_<block>` parameters.
#'
#' @section Methods:
#' * `$plot_variance()`: stacked bar chart of `ev_block`.  
#' * `$plot_loadings(top_n = 20, palette = "Dark2")`: bar plots of top sparse
#'   loadings per block/component.  
#' * `$plot_scree(type = c("component","cumulative"))`: scree/cumulative EV.  
#' * `$plot_score_heatmap(method = "spearman")`: heat-map of score correlations.  
#' * `$plot_score_network(cutoff = 0.3, method = "spearman")`: network of score
#'   correlations with |r| ≥ cutoff.
#'
#' @section Construction:
#' `PipeOpMBsPCA$new(id = "mbspca", blocks, param_vals = list())`
#'
#' @param id (`character(1)`) Identifier of the PipeOp.
#' @param blocks (`named list`)\cr
#'   Map of `block → character()` feature names.
#' @param param_vals (`list`)\cr
#'   Initial parameter values passed to the `ParamSet`.
#'
#' @details
#' During training, non-numeric or constant features are silently removed from
#' each block. If `c_matrix` is supplied, its number of columns determines the
#' maximum number of components (overrides `ncomp`). Deflation is performed
#' block-wise after each component.
#'
#' @return
#' * **Training**: appends latent score columns and sets `$state` as described.
#' * **Prediction**: appends latent score columns computed from stored weights.
#' * **Plot methods**: return a `ggplot` object.
#'
#' @examples
#' \dontrun{
#' library(mlr3); library(mlr3pipelines)
#' task = tsk("mtcars")
#' blocks = list(eng = c("disp","hp","drat"),
#'               body = c("wt","qsec"))
#' po = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 2))
#' g = as_graph(po)
#' g$train(task)
#' print(po$plot_scree())
#' }
#'
#' @seealso [mlr3pipelines::PipeOp], [mlr3tuning], `TunerSeqMBsPCA`
#' @importFrom rlang "%||%"
#' @export
PipeOpMBsPCA = R6::R6Class(
  "PipeOpMBsPCA",
  inherit = mlr3pipelines::PipeOpTaskPreproc,

  public = list(

    #' @field blocks Named list mapping block names to feature columns (kept for predict()).
    blocks = NULL,                      # named list is kept for predict()

    #' @description Create a new PipeOpMBsPCA.
    #' @param id character(1). Identifier (default: "mbspca").
    #' @param blocks named list. Block → character() of feature names (required).
    #' @param param_vals list. Initial ParamSet values.
    initialize = function(id = "mbspca",
                          blocks,
                          param_vals = list()) {

      checkmate::assert_list(blocks, min.len = 1L, names = "unique")

      ## ── build ParamSet ─────────────────────────────────────────────
      ps_base <- list(
        blocks      = paradox::p_uty(tags = "train", default = blocks),
        ncomp       = paradox::p_int(lower = 1L, default = 1L, tags = "train"),
        permutation_test = paradox::p_lgl(default = FALSE, tags = "train"),
        n_perm           = paradox::p_int(lower = 1L, default = 500L, tags = "train"),
        perm_alpha       = paradox::p_dbl(lower = 0, upper = 1,
                                          default = 0.05, tags = "train"),
        c_matrix         = paradox::p_uty(tags = c("train", "tune"),
                                          default = NULL)
      )

      ## one sparsity hyper‑parameter per block (√L¹ budget)
      for (bn in names(blocks)) {
        p <- length(blocks[[bn]])
        ps_base[[paste0("c_", bn)]] <- paradox::p_dbl(
          lower   = 1,
          upper   = sqrt(p),
          default = max(1, sqrt(p)),
          tags    = c("train", "tune")
        )
      }

      super$initialize(
        id         = id,
        param_set  = do.call(paradox::ps, ps_base),
        param_vals = param_vals
      )

      self$packages <- "mlr3mbspls"      # ensure namespace on workers
      self$blocks   <- blocks
    },

    #──────────────────────────────── plots ────────────────────────────
    #' @description Stacked bar plot of block-wise variance explained per component (training).
    #' @return A ggplot object.
    plot_variance = function() {
      if (!requireNamespace("ggplot2", quietly = TRUE))
        stop("Package 'ggplot2' is required for this plot.")
      st <- self$state
      if (is.null(st$ev_block))
        stop("No variance information stored – did you train the operator?")
      dt <- data.table::as.data.table(st$ev_block)
      dt[, component := factor(rownames(st$ev_block),
                               levels = rownames(st$ev_block))]
      dt <- data.table::melt(dt, id.vars = "component",
                             variable.name = "block",
                             value.name = "explained")
      ggplot2::ggplot(dt,
        ggplot2::aes(component, explained, fill = block)) +
        ggplot2::geom_col(position = "stack") +
        ggplot2::scale_y_continuous(labels = scales::percent_format(1)) +
        ggplot2::labs(y = "Variance explained", x = NULL) +
        ggplot2::theme_minimal(11)
    },

    #' @description Bar-plot of sparse loadings (top features) per block & component.
    #' @param top_n integer(1). Number of features to show per block/component (default 20).
    #' @param palette character(1) or character(2). Brewer palette name, or 2 custom fill colors (pos/neg).
    #' @return A ggplot object.
    plot_loadings = function(top_n = 20, palette = "Dark2") {
      if (!requireNamespace("ggplot2", quietly = TRUE))
        stop("Package 'ggplot2' is required for this plot.")
      if (!requireNamespace("dplyr", quietly = TRUE))
        stop("Package 'dplyr' is required for this plot.")

      st <- self$state
      if (is.null(st$weights))
        stop("PipeOp must be trained first.")

      comp_names  <- names(st$weights)
      block_names <- names(st$weights[[1]])

      long <- purrr::imap_dfr(st$weights, \(Wk, comp)
        purrr::imap_dfr(Wk, \(wb, block)
          tibble::tibble(
            component = comp,
            block     = block,
            feature   = names(wb),
            weight    = as.numeric(wb)
          )))

      if (length(palette) == 1L) {
        if (!palette %in% rownames(RColorBrewer::brewer.pal.info))
          stop("Unknown palette '", palette, "'.")
      }
      pal_vals <- if (length(palette) == 1L)
        RColorBrewer::brewer.pal(3, palette)[c(3, 1)] else palette[1:2]

      long %>%
        dplyr::mutate(abs_w = abs(weight)) %>%
        dplyr::filter(abs_w > 0) %>%
        dplyr::group_by(component, block) %>%
        dplyr::slice_max(abs_w, n = max(1, top_n), with_ties = FALSE) %>%
        dplyr::ungroup() %>%
        dplyr::arrange(component, block, abs_w) %>%
        dplyr::mutate(
          feature_unique = paste(component, block, feature, sep = "_"),
          feature_unique = factor(feature_unique, levels = feature_unique)
        ) %>%
        ggplot2::ggplot(ggplot2::aes(feature_unique, weight,
                                     fill = weight > 0)) +
        ggplot2::geom_col(show.legend = FALSE) +
        ggplot2::facet_grid(block ~ component, scales = "free") +
        ggplot2::scale_fill_manual(values = pal_vals) +
        ggplot2::scale_x_discrete(labels = function(x) {
          sapply(strsplit(x, "_"), function(z) paste(z[-(1:2)], collapse = "_"))
        }) +
        ggplot2::coord_flip() +
        ggplot2::labs(x = NULL, y = "Weight",
                      title = "Sparse loadings (top features)") +
        ggplot2::theme_bw(base_size = 10) +
        ggplot2::theme(strip.text.y = ggplot2::element_text(angle = 0),
                       axis.text.y  = ggplot2::element_text(size = 8))
    },

    #' @description Scree or cumulative variance-explained plot (training).
    #' @param type character(1). One of "component" or "cumulative".
    #' @return A ggplot object.
    plot_scree = function(type = c("component", "cumulative")) {
      if (!requireNamespace("ggplot2", quietly = TRUE))
        stop("Package 'ggplot2' is required for this plot.")
      type <- match.arg(type)
      st   <- self$state
      ev   <- st$ev_comp
      if (is.null(ev)) stop("Train the operator first.")
      if (is.null(names(ev)))
        names(ev) <- paste0("PC", seq_along(ev))

      df <- data.frame(PC = factor(names(ev), levels = names(ev)),
                       EV = ev,
                       CUM = cumsum(ev))

      ggplot2::ggplot(df,
        ggplot2::aes(PC, if (type == "component") EV else CUM,
                     group = 1)) +
        ggplot2::geom_line() + ggplot2::geom_point(size = 2) +
        ggplot2::scale_y_continuous(labels = scales::percent) +
        ggplot2::labs(y = if (type == "component")
                          "Variance explained"
                       else
                          "Cumulative variance explained",
                      x = NULL, title = "MB‑sPCA variance profile") +
        ggplot2::theme_minimal(11)
    },

    #' @description Heat-map of score correlations across appended PC score columns.
    #' @param method character(1). Correlation method, e.g. "spearman" (default) or "pearson".
    #' @return A ggplot object.
    plot_score_heatmap = function(method = "spearman") {
      if (!requireNamespace("ggplot2", quietly = TRUE))
        stop("Package 'ggplot2' is required for this plot.")
      st <- self$state
      if (is.null(st$T_mat) || ncol(st$T_mat) < 2)
        stop("Need at least two PCs to draw a heat‑map.")

      C <- cor(st$T_mat, method = method)
      diag(C) <- NA
      lvls <- colnames(C)

      meltC <- data.frame(
        row = factor(rep(lvls, each = length(lvls)), levels = lvls),
        col = factor(rep(lvls, times = length(lvls)), levels = lvls),
        value = as.vector(C)
      )

      ggplot2::ggplot(meltC, ggplot2::aes(row, col, fill = value)) +
        ggplot2::geom_tile() +
        ggplot2::scale_fill_gradient2(limits = c(-1, 1), midpoint = 0,
                                      low = "blue", mid = "white",
                                      high = "red", na.value = "grey90") +
        ggplot2::theme_minimal() +
        ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90,
                                                           vjust = 0.5,
                                                           hjust = 1),
                       panel.grid = ggplot2::element_blank()) +
        ggplot2::labs(title = sprintf("PC score correlations (%s)", method),
                      x = NULL, y = NULL, fill = "r")
    },

    #' @description Network of score correlations between PCs with |r| above a cutoff.
    #' @param cutoff numeric(1). Absolute correlation threshold in `[0, 1]` (default 0.3).
    #' @param method character(1). Correlation method, e.g. "spearman" (default) or "pearson".
    #' @return A ggplot object.
    plot_score_network = function(cutoff = 0.3, method = "spearman") {
      requireNamespace("igraph"); requireNamespace("ggraph")
      st <- self$state
      if (is.null(st$T_mat) || ncol(st$T_mat) < 2)
        stop("Need at least two PCs to draw a network.")

      C <- cor(st$T_mat, method = method); diag(C) <- 0
      idx <- which(abs(C) >= cutoff, arr.ind = TRUE)
      idx <- idx[idx[, 1] < idx[, 2], , drop = FALSE]  # upper triangle

      if (!nrow(idx))
        stop(sprintf("No PC pairs exceed |r| >= %.2f", cutoff))

      edges <- data.frame(
        from = rownames(C)[idx[, 1]],
        to   = colnames(C)[idx[, 2]],
        r    = C[idx],
        stringsAsFactors = FALSE
      )
      g <- igraph::graph_from_data_frame(edges, directed = FALSE)

      ggraph::ggraph(g, layout = "fr") +
        ggraph::geom_edge_link(ggplot2::aes(width = abs(r), colour = r)) +
        ggraph::scale_edge_width(range = c(0.3, 3)) +
        ggraph::scale_edge_colour_gradient2(limits = c(-1, 1),
                                            midpoint = 0,
                                            low = "blue", mid = "grey70",
                                            high = "red") +
        ggraph::geom_node_point(size = 4, colour = "grey20") +
        ggraph::geom_node_text(ggplot2::aes(label = name), repel = TRUE,
                               size = 3) +
        ggplot2::theme_void() +
        ggplot2::labs(title = sprintf("PC network |r| ≥ %.2f (%s)",
                                      cutoff, method))
    }
  ),

  private = list(

    # ─────────────────────────── train helper ────────────────────────
    .train_dt = function(dt, levels, target = NULL) {

      pv <- utils::modifyList(
        paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "train"),
        keep.null = TRUE
      )

      blocks  <- pv$blocks
      n_block <- length(blocks)

      ## 0) sanity‑filter columns: numeric, non‑constant ---------------
      blocks <- lapply(blocks, function(cols) {
        cols <- intersect(cols, names(dt))                   # still present?
        cols <- cols[vapply(cols, \(cl) is.numeric(dt[[cl]]),
                           logical(1))]                     # numeric only
        cols <- cols[vapply(cols,
                            \(cl) stats::var(dt[[cl]], na.rm = TRUE) > 0,
                            logical(1))]                     # non‑constant
        cols
      })
      blocks <- Filter(length, blocks)           # drop empty blocks
      if (!length(blocks))
        stop("No block contains at least one usable numeric column.")

      ## 1) materialise block matrices ---------------------------------
      X <- lapply(blocks, \(cols) {
        M <- as.matrix(dt[, ..cols]); storage.mode(M) <- "double"; M
      })

      ## 2) handle c‑matrix vs single‑value per block ------------------
      if (!is.null(pv$c_matrix)) {
        cm <- pv$c_matrix
        if (!is.null(rownames(cm)))
          cm <- cm[names(blocks), , drop = FALSE]
        pv$ncomp <- ncol(cm)                      # override
      } else {
        cm <- NULL
      }

      ncomp <- pv$ncomp
      W_all <- P_all <- vector("list", ncomp)
      ev_blk <- matrix(0, nrow = ncomp, ncol = length(blocks))
      colnames(ev_blk) <- names(blocks)
      ev_cmp <- numeric(ncomp)

      ## copy of X that we deflate iteratively
      X_res <- X
      ss_tot_block <- vapply(X, \(M) sum(M^2), numeric(1))
      ss_tot_all   <- sum(ss_tot_block)

      for (k in seq_len(ncomp)) {

        ## sparsity vector for this PC
        c_k <- if (is.null(cm))
                  vapply(names(blocks),
                         \(bn) pv[[paste0("c_", bn)]],
                         numeric(1))
               else
                  cm[, k]

        ## fit one component ------------------------------------------
        fit <- cpp_mbspca_one_lv(X_res, c_k,
                                 max_iter = 60L, tol = 1e-4)
        Wk  <- fit$W

        ## block scores & loadings
        Tk <- matrix(0, nrow = nrow(X_res[[1]]), ncol = length(blocks))
        Pk <- vector("list", length(blocks))
        ss_exp_tot <- 0
        for (b in seq_along(blocks)) {
          tb <- X_res[[b]] %*% Wk[[b]]
          norm2 <- drop(crossprod(tb))
          Tk[, b] <- tb
          if (norm2 > 1e-12) {
            pb <- crossprod(X_res[[b]], tb) / norm2
            Pk[[b]] <- as.numeric(pb)
            ss_b <- norm2 * sum(pb^2)
            ev_blk[k, b] <- ss_b / ss_tot_block[b]
            ss_exp_tot   <- ss_exp_tot + ss_b
          } else {
            Pk[[b]] <- numeric(ncol(X_res[[b]]))
          }
        }
        ev_cmp[k] <- ss_exp_tot / ss_tot_all

        W_all[[k]] <- Wk
        P_all[[k]] <- Pk

        ## permutation early‑stop? ------------------------------------
        if (isTRUE(pv$permutation_test)) {
          p_val <- perm_test_component_mbspca(X_res, Wk, c_k,
                                              n_perm = pv$n_perm,
                                              alpha  = pv$perm_alpha)
          if (p_val > pv$perm_alpha && k != 1L) {   # always keep PC‑1
            W_all   <- W_all[seq_len(k - 1)]
            P_all   <- P_all[seq_len(k - 1)]
            ev_blk  <- ev_blk[seq_len(k - 1), , drop = FALSE]
            ev_cmp  <- ev_cmp[seq_len(k - 1)]
            ncomp   <- k - 1
            break
          }
        }

        ## deflate residual matrices ----------------------------------
        for (b in seq_along(blocks)) {
          tb <- Tk[, b]; denom <- drop(crossprod(tb))
          if (denom > 1e-12) {
            pb <- Pk[[b]]
            X_res[[b]] <- X_res[[b]] - tcrossprod(tb, pb)
          }
        }
      }

      ## ── build output latent score table ---------------------------
      coln <- unlist(lapply(seq_len(ncomp),
                            \(k) paste0("PC", k, "_", names(blocks))))
      T_mat <- do.call(cbind, lapply(seq_len(ncomp), \(k) {
        do.call(cbind, lapply(seq_along(blocks), \(b)
          X[[b]] %*% W_all[[k]][[b]]))
      }))
      colnames(T_mat) <- coln
      dt_lat <- data.table::as.data.table(T_mat)

      ## ── add names so downstream plots work ──────────────────────────
      pad_and_name <- function(x, feat) {
        if (length(x) == 0) x <- numeric(length(feat))
        else if (length(x) != length(feat))
          stop("weight/feature length mismatch")
        stats::setNames(as.numeric(x), feat)
      }

      comp_ids <- paste0("PC", seq_len(ncomp))

      for (k in seq_len(ncomp)) {
        for (b in seq_along(blocks)) {
          feat <- blocks[[b]]
          W_all[[k]][[b]] <- pad_and_name(W_all[[k]][[b]], feat)
          P_all[[k]][[b]] <- pad_and_name(P_all[[k]][[b]], feat)
        }
      }
      names(W_all) <- names(P_all) <- comp_ids
      rownames(ev_blk) <- comp_ids
      names(ev_cmp)    <- comp_ids

      ## store state --------------------------------------------------
      self$state <- list(
        blocks   = blocks,
        ncomp    = ncomp,
        weights  = W_all,
        loadings = P_all,
        ev_block = ev_blk,
        ev_comp  = ev_cmp,
        T_mat    = T_mat
      )
      dt_lat
    },

    # ────────────────────────── predict helper ───────────────────────
    .predict_dt = function(dt, levels, target = NULL) {

      st <- self$state
      blocks <- st$blocks
      ## add missing columns (zeros) so matrix dimensions match
      miss <- setdiff(unlist(blocks), names(dt))
      if (length(miss)) dt[, (miss) := 0 ]

      X_cur <- lapply(blocks, \(cols) {
        M <- as.matrix(dt[, ..cols]); storage.mode(M) <- "double"; M
      })

      lat_list <- vector("list", st$ncomp)

      for (k in seq_len(st$ncomp)) {
        Wk <- st$weights[[k]]
        Tk <- do.call(cbind, lapply(seq_along(blocks),
                                    \(b) X_cur[[b]] %*% Wk[[b]]))
        colnames(Tk) <- paste0("PC", k, "_", names(blocks))
        lat_list[[k]] <- Tk

        ## deflate for next component
        if (k < st$ncomp) {
          Pk <- st$loadings[[k]]
          for (b in seq_along(blocks)) {
            tb <- Tk[, b]; denom <- drop(crossprod(tb))
            if (denom > 1e-12)
              X_cur[[b]] <- X_cur[[b]] -
                            tcrossprod(tb, Pk[[b]])
          }
        }
      }
      data.table::as.data.table(do.call(cbind, lat_list))
    }
  )
)
