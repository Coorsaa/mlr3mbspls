#' MB‑sPLS Bootstrap Selection with Two Methods ("ci" | "frequency")
#'
#' @title PipeOp \code{mbspls_bootstrap_select}
#'
#' @description
#' Performs post‑hoc **bootstrap**, aligns replicate components (two alignment modes),
#' summarises per‑feature weights, then **selects features** via:
#' \itemize{
#'   \item \code{selection_method = "ci"} (default): keep if CI excludes 0 AND |mean| > 1e-3;
#'   \item \code{selection_method = "frequency"}: keep if non‑zero frequency ≥ \code{frequency_threshold}.
#' }
#' Blocks with no kept features **vanish** for that component; components with no
#' non‑empty blocks are dropped. Remaining components are **renumbered**.
#' Final weights are the **mean aligned bootstrap weights**. Training scores are
#' recomputed with **deflation** from those weights and replace upstream LV columns.
#'
#' **Important:** This operator uses the original block features to recompute stable
#' LV scores and then **drops those block features and all upstream LV columns**, so
#' only the kept stable LV columns remain downstream. No fallback to upstream LVs.
#'
#' @section Parameters:
#' @param log_env Environment shared with upstream \code{po("mbspls")} (required).
#' @param bootstrap Run bootstrap selection (default \code{TRUE}).
#' @param B Bootstrap replicates (default \code{500}).
#' @param alpha CI alpha (default \code{0.05} → 95\% CI).
#' @param align \code{"block_sign"} (default) or \code{"score_correlation"}.
#' @param selection_method \code{"ci"} (default) or \code{"frequency"}.
#' @param frequency_threshold Only for \code{"frequency"}; default \code{0.60}.
#' @param stratify_by_block Optional dummy‑encoded block name for stratified bootstrap (e.g., "Studygroup").
#' @param workers \#Unix workers for \code{mclapply}; default cores−1.
#' 
#' @return Replaces task LV columns with kept components' LV columns (renumbered).
#' Stores \code{weights_ci}, \code{weights_selectfreq}, \code{weights_stable},
#' \code{loadings_stable}, \code{kept_components}, \code{kept_blocks_per_comp}.
#'
#' @import data.table
#' @importFrom R6 R6Class
#' @importFrom lgr lgr
#' @importFrom paradox ps p_uty p_fct p_lgl p_int p_dbl
#' @importFrom mlr3pipelines PipeOpTaskPreproc
#' @importFrom parallel detectCores mclapply
#' @importFrom stats cor sd var quantile setNames
#' @export
PipeOpMBsPLSBootstrapSelect <- R6::R6Class(
  "PipeOpMBsPLSBootstrapSelect",
  inherit = mlr3pipelines::PipeOpTaskPreproc,
  public = list(
    #' @description Initialize the PipeOpMBsPLSBootstrapSelect.
    #' @param id character(1). Identifier of the resulting object.
    #' @param param_vals named list. List of hyperparameter settings.
    initialize = function(id = "mbspls_bootstrap_select", param_vals = list()) {
      ncore_default <- 1L
      try({ ncore_default <- max(1L, parallel::detectCores(logical = TRUE) - 1L) }, silent = TRUE)

      ps <- paradox::ps(
        log_env   = paradox::p_uty(tags = c("train","predict"), default = NULL),

        bootstrap = paradox::p_lgl(default = TRUE, tags = "train"),
        B         = paradox::p_int(lower = 1L, default = 500L, tags = "train"),
        alpha     = paradox::p_dbl(lower = 0, upper = 1, default = 0.05, tags = "train"),

        align     = paradox::p_fct(levels = c("score_correlation","block_sign"),
                                   default = "block_sign", tags = "train"),

        selection_method    = paradox::p_fct(levels = c("ci","frequency"), default = "ci", tags = "train"),
        frequency_threshold = paradox::p_dbl(lower = 0, upper = 1, default = 0.60, tags = "train"),

        stratify_by_block   = paradox::p_uty(default = NULL, tags = "train"),
        workers             = paradox::p_int(lower = 1L, default = ncore_default, tags = "train")
      )

      super$initialize(id = id, param_set = ps, param_vals = param_vals)
      self$packages <- "mlr3mbspls"
    }
  ),

  private = list(

    # ---------- utils ----------
    .lv_column_map = function(dt_names) {
      lv_cols <- grep("^LV\\d+_", dt_names, value = TRUE)
      if (!length(lv_cols)) return(list(K = 0L, blocks = character(0), map = list()))
      comps  <- as.integer(sub("^LV(\\d+)_.*$", "\\1", lv_cols, perl = TRUE))
      blocks <- sub("^LV\\d+_", "", lv_cols)
      K      <- max(comps)
      bset   <- unique(blocks)
      map <- lapply(seq_len(K), function(k) {
        sel <- (comps == k)
        stats::setNames(lv_cols[sel], blocks[sel])
      })
      list(K = K, blocks = bset, map = map)
    },

    .get_env_state = function(pv) {
      env <- pv$log_env
      if (!inherits(env, "environment")) stop("Provide a shared 'log_env' with po('mbspls').")
      st  <- env$mbspls_state
      if (is.null(st)) stop("mbspls_state not found in log_env; place this PipeOp directly after po('mbspls').")
      st
    },

    .recompute_scores_deflated = function(X_list, W_list, all_blocks) {
      B <- length(all_blocks); K <- length(W_list)
      X_cur <- lapply(X_list, identity)
      T_tabs <- vector("list", K)
      P_all  <- vector("list", K)

      for (k in seq_len(K)) {
        Tk <- matrix(0, nrow(X_cur[[1]]), B)
        Pk <- vector("list", B); names(Pk) <- all_blocks

        for (bi in seq_along(all_blocks)) {
          b <- all_blocks[bi]
          w_b <- W_list[[k]][[b]]
          if (is.null(w_b)) {
            Tk[, bi] <- 0
            Pk[[b]]  <- setNames(rep(0, ncol(X_cur[[b]])), colnames(X_cur[[b]]))
          } else {
            cols <- colnames(X_cur[[b]])
            if (!is.null(names(w_b))) {
              wv <- as.numeric(w_b[cols]); wv[is.na(wv)] <- 0
            } else {
              wv <- as.numeric(w_b)
              if (length(wv) != length(cols)) {
                wv <- numeric(length(cols))
              }
            }
            storage.mode(wv) <- "double"
            Tk[, bi] <- X_cur[[b]] %*% wv
            denom <- sum(Tk[, bi] * Tk[, bi])
            if (denom <= 0) {
              Pk[[b]] <- setNames(rep(0, ncol(X_cur[[b]])), colnames(X_cur[[b]]))
            } else {
              pb <- drop(crossprod(X_cur[[b]], Tk[, bi]) / denom)
              names(pb) <- colnames(X_cur[[b]])
              Pk[[b]] <- pb
            }
          }
        }
        colnames(Tk) <- paste0("LV", k, "_", all_blocks)
        T_tabs[[k]] <- data.table::as.data.table(Tk)
        P_all[[k]]  <- Pk

        if (k < K) { # deflate
          for (bi in seq_along(all_blocks)) {
            b <- all_blocks[bi]
            t_b <- Tk[, bi, drop = FALSE]
            pb  <- Pk[[b]]
            X_cur[[b]] <- X_cur[[b]] - t_b %*% t(as.matrix(pb))
          }
        }
      }
      list(T_mat = do.call(cbind, T_tabs), P = P_all)
    },

    # --------------- bootstrap core (alignment + acceptance + summary) --------
    .bootstrap_align_and_summarise = function(
      X_list, W_ref, blocks, ncomp,
      sparsity, corr_method = "pearson", perf_metric = "mac",
      B = 500L, alpha = 0.05, align = "score_correlation",
      workers = 1L, stratify_block = NULL
    ) {
      MIN_SCORE_COR <- 0.10  # acceptance gate

      # match blocks
      bn <- intersect(names(X_list), names(W_ref[[1]]))
      if (!length(bn)) stop("bootstrap: cannot match blocks between X_list and reference weights")
      X_list <- X_list[bn]
      W_ref  <- lapply(W_ref, function(wk) wk[bn])
      comp_lab <- sprintf("LC_%02d", seq_len(ncomp))
      N <- nrow(X_list[[1]]); stopifnot(all(vapply(X_list, nrow, 1L) == N))

      pad_to_order <- function(w_boot, w_ref_named) {
        all_feat <- names(w_ref_named)
        out <- numeric(length(all_feat)); names(out) <- all_feat
        if (!is.null(names(w_boot))) out[names(w_boot)] <- w_boot
        out
      }
      ref_concat <- lapply(seq_len(ncomp), function(k)
        unlist(lapply(bn, function(b) as.numeric(W_ref[[k]][[b]])))
      )
      concat_boot <- function(Wi, k) {
        unlist(lapply(bn, function(b) {
          w_b <- Wi[[b]]
          if (is.null(names(w_b)) && length(w_b) == length(W_ref[[k]][[b]]))
            names(w_b) <- names(W_ref[[k]][[b]])
          pad_to_order(w_b, W_ref[[k]][[b]])
        }))
      }

      T_ref <- lapply(seq_len(ncomp), function(k)
        lapply(bn, function(b) as.numeric(X_list[[b]] %*% W_ref[[k]][[b]]))
      )

      strata <- NULL
      if (!is.null(stratify_block) && stratify_block %in% names(X_list)) {
        Xs <- X_list[[stratify_block]]
        labs_idx <- max.col(Xs, ties.method = "first")
        labs <- colnames(Xs)[labs_idx]
        strata <- factor(labs, levels = unique(labs))
        lgr$info("Stratified bootstrap by '%s' with %d strata.", stratify_block, nlevels(strata))
      }

      fit_once <- function(Xb) {
        if (!is.null(sparsity) && identical(sparsity$type, "c_matrix")) {
          cpp_mbspls_multi_lv_cmatrix(
            X_blocks  = Xb, c_matrix = sparsity$c_matrix,
            max_iter  = 600L, tol = 1e-4,
            spearman  = (corr_method == "spearman"),
            do_perm   = FALSE, n_perm = 0L, alpha = 0.05,
            frobenius = (perf_metric == "frobenius")
          )
        } else {
          c_vec <- if (!is.null(sparsity) && !is.null(sparsity$c_vec)) sparsity$c_vec else
            stop("bootstrap: sparsity$c_vec missing")
          cpp_mbspls_multi_lv(
            X_blocks      = Xb,
            c_constraints = as.numeric(c_vec[bn]),
            K             = ncomp,
            max_iter      = 600L,
            spearman      = (corr_method == "spearman"),
            do_perm       = FALSE, n_perm = 0L, alpha = 0.05,
            frobenius     = (perf_metric == "frobenius")
          )
        }
      }

      cor_safe <- function(x, y) {
        if (length(unique(x)) < 2 || length(unique(y)) < 2) return(NA_real_)
        suppressWarnings(stats::cor(x, y, method = corr_method))
      }

      one_rep <- function(r) {
        idx <- if (!is.null(strata)) {
          splits <- split(seq_len(N), strata, drop = FALSE)
          idx_cat <- unlist(lapply(splits, function(ix) if (length(ix)) sample(ix, length(ix), TRUE) else integer(0)), use.names = FALSE)
          if (!length(idx_cat)) return(NULL)
          sample(idx_cat)
        } else sample.int(N, replace = TRUE)

        Xb <- lapply(X_list, function(X) X[idx, , drop = FALSE])
        fit_r <- try(fit_once(Xb), silent = TRUE)
        if (inherits(fit_r, "try-error")) return(NULL)
        W_r <- fit_r$W; Kfit <- length(W_r); if (!Kfit) return(NULL)
        W_r <- lapply(W_r, function(wk) { if (is.null(names(wk))) names(wk) <- bn; wk[bn] })

        # replicate scores (pre-align)
        T_boot <- lapply(seq_len(Kfit), function(i)
          lapply(bn, function(b) as.numeric(Xb[[b]] %*% W_r[[i]][[b]]))
        )

        # match comps by mean |cor| across blocks
        K <- ncomp
        S <- matrix(0, K, K)
        for (i in seq_len(min(Kfit, K))) for (k in seq_len(K)) {
          cs <- vapply(seq_along(bn), function(bi) {
            r_ <- cor_safe(T_boot[[i]][[bi]], T_ref[[k]][[bi]][idx])
            if (is.finite(r_)) abs(r_) else NA_real_
          }, numeric(1))
          cs <- cs[is.finite(cs)]
          S[i, k] <- if (length(cs)) mean(cs) else 0
        }
        cost <- matrix(1, K, K)
        if (min(Kfit, K) > 0) cost[seq_len(min(Kfit, K)), ] <- 1 - S[seq_len(min(Kfit, K)), ]
        perm <- if (requireNamespace("clue", quietly = TRUE)) {
          as.integer(clue::solve_LSAP(cost))
        } else {
          rem <- seq_len(K); p <- rep(NA_integer_, K)
          for (i in seq_len(min(Kfit, K))) { j <- which.max(S[i, rem]); p[i] <- rem[j]; rem <- rem[-j] }
          if (length(rem)) p[is.na(p)] <- rem
          p
        }

        out_list <- list(); ptr <- 0L
        n_eff <- integer(K)

        for (i in seq_len(min(Kfit, K))) {
          k <- perm[i]; if (is.na(k)) next

          # ---------- GLOBAL ALIGNMENT: score_correlation ----------
          if (align == "score_correlation") {
            cs <- vapply(seq_along(bn), function(bi) {
              cor_safe(T_boot[[i]][[bi]], T_ref[[k]][[bi]][idx])
            }, numeric(1))
            cs <- cs[is.finite(cs)]
            if (length(cs)) {
              s_all <- sign(sum(cs))
              if (!is.finite(s_all) || s_all == 0) {
                j <- which.max(abs(cs))
                s_all <- sign(cs[j]); if (!is.finite(s_all) || s_all == 0) s_all <- +1L
              }
            } else {
              s_all <- +1L
            }
            for (bi in seq_along(bn)) T_boot[[i]][[bi]] <- s_all * T_boot[[i]][[bi]]
            for (b in bn) {
              w_ref_b  <- W_ref[[k]][[b]]
              w_boot_b <- W_r[[i]][[b]]
              if (is.null(names(w_boot_b)) && length(w_boot_b) == length(w_ref_b))
                names(w_boot_b) <- names(w_ref_b)
              W_r[[i]][[b]] <- s_all * pad_to_order(w_boot_b, w_ref_b)
            }

          } else if (align == "block_sign") {
            for (bi in seq_along(bn)) {
              b <- bn[bi]
              w_ref_b  <- W_ref[[k]][[b]]
              w_boot_b <- W_r[[i]][[b]]
              if (is.null(names(w_boot_b)) && length(w_boot_b) == length(w_ref_b))
                names(w_boot_b) <- names(w_ref_b)
              s_b <- sign(sum(pad_to_order(w_boot_b, w_ref_b) * as.numeric(w_ref_b)))
              if (!is.finite(s_b) || s_b == 0) s_b <- +1L
              T_boot[[i]][[bi]] <- s_b * T_boot[[i]][[bi]]
              W_r[[i]][[b]]     <- s_b * pad_to_order(w_boot_b, w_ref_b)
            }

          } else {
            stop(sprintf("Unknown align mode: %s", align))
          }
          # ---------- END ALIGNMENT ----------

          # acceptance gate on aligned scores
          sc <- vapply(seq_along(bn), function(bi) {
            suppressWarnings(abs(stats::cor(T_boot[[i]][[bi]], T_ref[[k]][[bi]][idx], method = corr_method)))
          }, numeric(1))
          sc <- sc[is.finite(sc)]
          if (!length(sc) || mean(sc) < MIN_SCORE_COR) next
          n_eff[k] <- n_eff[k] + 1L

          # record aligned weights (including zeros)
          for (b in bn) {
            wb <- W_r[[i]][[b]]
            ptr <- ptr + 1L
            out_list[[ptr]] <- data.table::data.table(
              replicate = r, component = sprintf("LC_%02d", k), block = b,
              feature = names(wb), weight = as.numeric(wb)
            )
          }
        }

        list(
          draws = if (length(out_list)) data.table::rbindlist(out_list, use.names = TRUE, fill = TRUE) else NULL,
          n_eff = n_eff
        )
      }

      rep_idx <- seq_len(B)
      rep_res <- if (.Platform$OS.type == "unix" && workers > 1L) {
        parallel::mclapply(rep_idx, one_rep, mc.cores = workers)
      } else lapply(rep_idx, one_rep)

      n_eff <- Reduce(`+`, lapply(rep_res, `[[`, "n_eff"), init = integer(length(comp_lab)))
      n_eff_by_component <- data.table::data.table(
        component = comp_lab, n_eff = as.integer(n_eff)
      )

      # gather draws
      dlist <- Filter(Negate(is.null), lapply(rep_res, `[[`, "draws"))
      draws <- if (length(dlist)) data.table::rbindlist(dlist, use.names = TRUE, fill = TRUE) else
        data.table::data.table()
      if (nrow(draws)) {
        draws[, component := as.character(component)]
        draws[, block := as.character(block)]
      }

      # selection frequency (non-zero among accepted)
      sel_grid <- data.table::rbindlist(lapply(seq_len(ncomp), function(k)
        data.table::rbindlist(lapply(bn, function(b)
          data.table::data.table(component = comp_lab[k], block = b, feature = names(W_ref[[k]][[b]]))
        ), use.names = TRUE, fill = TRUE)
      ), use.names = TRUE, fill = TRUE)

      inc_dt <- if (nrow(draws)) {
        tmp <- copy(draws)
        tmp[, sel := as.integer(abs(weight) > 0)]
        tmp[, .(sel = sum(sel, na.rm = TRUE)), by = .(component, block, feature)]
      } else data.table::data.table(component = character(0), block = character(0), feature = character(0), sel = integer(0))

      sel_freq <- merge(sel_grid, inc_dt, by = c("component","block","feature"), all.x = TRUE)
      sel_freq[is.na(sel), sel := 0L]
      sel_freq[, eff := n_eff[match(component, comp_lab)]]
      sel_freq[eff <= 0, eff := NA_integer_]
      sel_freq[, `:=`(freq = sel / eff, sel = NULL, eff = NULL)]
      sel_freq$component <- as.character(sel_freq$component)
      sel_freq$block     <- as.character(sel_freq$block)

      # summaries (CIs + means over aligned accepted draws)
      a <- alpha
      summary <- if (nrow(draws)) {
        draws[, {
          list(
            boot_mean = mean(weight, na.rm = TRUE),
            boot_sd   = stats::sd(weight, na.rm = TRUE),
            ci_lower  = stats::quantile(weight, probs = a/2,     na.rm = TRUE),
            ci_upper  = stats::quantile(weight, probs = 1 - a/2, na.rm = TRUE)
          )
        }, by = .(component, block, feature)]
      } else data.table::data.table(component=character(),block=character(),feature=character(),
                                    boot_mean=numeric(),boot_sd=numeric(),
                                    ci_lower=numeric(),ci_upper=numeric())

      summary$component <- as.character(summary$component)
      summary$block     <- as.character(summary$block)

      list(
        summary = summary,
        select_freq = sel_freq,
        n_eff_by_component = n_eff_by_component,
        blocks_order = bn,
        comp_labels  = comp_lab
      )
    },

    # ---------------- TRAIN ----------------
    .train_task = function(task) {
      pv <- utils::modifyList(paradox::default_values(self$param_set),
                              self$param_set$get_values(tags = "train"),
                              keep.null = TRUE)
      st_env <- private$.get_env_state(pv)

      dt_all <- task$data()
      lm <- private$.lv_column_map(names(dt_all))
      if (lm$K == 0L) stop("No LV columns found. Ensure po('mbspls') is upstream.")

      if (!isTRUE(pv$bootstrap)) {
        return(task)
      }

      lgr$info("Bootstrap selection: B=%d, align='%s', method='%s'",
               as.integer(pv$B), pv$align, pv$selection_method)

      blocks_map <- st_env$blocks
      X_blocks_train <- st_env$X_train_blocks
      if (is.null(X_blocks_train)) {
        lgr$warn("X_train_blocks not in env; rebuilding from current task for bootstrap stage.")
        X_blocks_train <- lapply(blocks_map, function(cols) {
          miss <- setdiff(cols, names(dt_all))
          if (length(miss)) for (m in miss) dt_all[, (m) := 0.0]
          m <- as.matrix(dt_all[, ..cols]); storage.mode(m) <- "double"; m
        })
      }

      bt <- private$.bootstrap_align_and_summarise(
        X_list        = X_blocks_train,
        W_ref         = st_env$weights,
        blocks        = st_env$blocks,
        ncomp         = length(st_env$weights),
        sparsity      = st_env$sparsity,
        corr_method   = st_env$corr_method %||% "pearson",
        perf_metric   = st_env$perf_metric %||% "mac",
        B             = as.integer(pv$B),
        alpha         = as.numeric(pv$alpha),
        align         = pv$align,
        workers       = as.integer(pv$workers),
        stratify_block= pv$stratify_by_block
      )

      bn      <- bt$blocks_order
      sum_df  <- as.data.frame(bt$summary)
      freq_df <- as.data.frame(bt$select_freq)
      K <- length(st_env$weights)

      W_stable <- list()
      kept_blocks_per_comp <- list()

      for (k in seq_len(K)) {
        k_lab <- sprintf("LC_%02d", k)
        Wk_out <- list()
        kept_blocks <- character(0)

        for (b in bn) {
          feats <- blocks_map[[b]]
          if (is.null(feats) || !length(feats)) next

          sb <- sum_df[sum_df$component == k_lab & sum_df$block == b,
                       c("feature","boot_mean","ci_lower","ci_upper"),
                       drop = FALSE]

          mu_map <- if (nrow(sb)) stats::setNames(sb$boot_mean, sb$feature) else setNames(numeric(0), character(0))
          lo_map <- if (nrow(sb)) stats::setNames(sb$ci_lower,  sb$feature) else setNames(numeric(0), character(0))
          hi_map <- if (nrow(sb)) stats::setNames(sb$ci_upper,  sb$feature) else setNames(numeric(0), character(0))

          mu <- as.numeric(mu_map[feats]); mu[is.na(mu)] <- 0
          lo <- as.numeric(lo_map[feats]); lo[is.na(lo)] <- 0
          hi <- as.numeric(hi_map[feats]); hi[is.na(hi)] <- 0

          if (pv$selection_method == "ci") {
            keep <- ((lo >= 0) | (hi <= 0)) & (abs(mu) > 1e-3)
          } else {
            fb <- freq_df[freq_df$component == k_lab & freq_df$block == b, c("feature","freq"), drop = FALSE]
            fq_map <- if (nrow(fb)) stats::setNames(fb$freq, fb$feature) else setNames(numeric(0), character(0))
            fv <- as.numeric(fq_map[feats]); fv[is.na(fv)] <- 0
            keep <- (fv >= as.numeric(pv$frequency_threshold))
          }

          mu[!keep] <- 0
          if (any(mu != 0)) {
            Wk_out[[b]] <- stats::setNames(mu, feats)
            kept_blocks <- c(kept_blocks, b)
          }
        }

        if (length(Wk_out)) {
          W_stable[[length(W_stable) + 1L]] <- Wk_out
          kept_blocks_per_comp[[length(kept_blocks_per_comp) + 1L]] <- kept_blocks
        }
      }

      names(W_stable) <- sprintf("LC_%02d", seq_along(W_stable))
      names(kept_blocks_per_comp) <- sprintf("LC_%02d", seq_along(W_stable))

      if (!length(W_stable)) {
        lgr$warn("All components empty after stability selection; removing upstream LV columns and block features.")
        # drop upstream LVs and original block features
        old_lv <- unlist(lm$map, use.names = FALSE)
        orig_feats <- intersect(unlist(blocks_map, use.names = FALSE), task$feature_names)
        keep_features <- setdiff(task$feature_names, c(old_lv, orig_feats))
        task$select(keep_features)

        self$state$kept_components <- integer(0)
        self$state$kept_blocks_per_comp <- list()
        self$state$weights_ci <- sum_df
        self$state$weights_selectfreq <- freq_df

        st_env$weights_stable       <- list()
        st_env$loadings_stable      <- list()
        st_env$ncomp                <- 0L
        st_env$T_mat_train          <- matrix(0, nrow = nrow(X_blocks_train[[1]]), ncol = 0)
        st_env$T_mat_train_kept     <- st_env$T_mat_train
        st_env$weights_ci           <- sum_df
        st_env$weights_selectfreq   <- freq_df
        st_env$kept_blocks_per_comp <- list()
        st_env$alignment_method     <- pv$align
        st_env$selection_method     <- pv$selection_method
        st_env$frequency_threshold  <- pv$frequency_threshold
        pv$log_env$mbspls_state     <- st_env

        return(task)
      }

      # recompute training scores with deflation using W_stable
      X_train <- st_env$X_train_blocks %||% X_blocks_train
      rec <- private$.recompute_scores_deflated(X_train, W_stable, names(blocks_map))
      T_all_dt  <- rec$T_mat
      P_all  <- rec$P

      # keep only non-empty block columns
      keep_cols <- character(0)
      for (newk in seq_along(W_stable)) {
        for (b in kept_blocks_per_comp[[newk]]) {
          keep_cols <- c(keep_cols, paste0("LV", newk, "_", b))
        }
      }
      T_all <- as.matrix(T_all_dt)
      if (length(keep_cols)) {
        keep_cols <- intersect(keep_cols, colnames(T_all))
        T_keep <- T_all[, keep_cols, drop = FALSE]
      } else {
        T_keep <- matrix(0, nrow = nrow(T_all), ncol = 0)
      }

      # ------- Drop upstream LVs and original block features; then append stable LVs
      old_lv <- unlist(lm$map, use.names = FALSE)
      orig_feats <- intersect(unlist(blocks_map, use.names = FALSE), task$feature_names)
      keep_features <- setdiff(task$feature_names, c(old_lv, orig_feats))
      task$select(keep_features)
      if (ncol(T_keep)) task$cbind(data.table::as.data.table(T_keep))
      lgr$info("Bootstrap-select (train): dropped %d original block features and %d upstream LV columns; kept %d stable LV columns.",
               length(orig_feats), length(old_lv), ncol(T_keep))

      # persist to state + env
      self$state$kept_components      <- seq_along(W_stable)
      self$state$kept_blocks_per_comp <- kept_blocks_per_comp
      self$state$weights_ci           <- sum_df
      self$state$weights_selectfreq   <- freq_df
      self$state$weights_stable       <- W_stable
      self$state$loadings_stable      <- P_all
      self$state$alignment_method     <- pv$align
      self$state$selection_method     <- pv$selection_method
      self$state$frequency_threshold  <- pv$frequency_threshold

      st_env$weights_stable       <- W_stable
      st_env$loadings_stable      <- P_all
      st_env$ncomp                <- length(W_stable)
      st_env$T_mat_train          <- as.matrix(T_all)
      st_env$T_mat_train_kept     <- as.matrix(T_keep)
      st_env$weights_ci           <- sum_df
      st_env$weights_selectfreq   <- freq_df
      st_env$kept_blocks_per_comp <- kept_blocks_per_comp
      st_env$alignment_method     <- pv$align
      st_env$selection_method     <- pv$selection_method
      st_env$frequency_threshold  <- pv$frequency_threshold
      pv$log_env$mbspls_state     <- st_env

      task
    },

    # ---------------- PREDICT ----------------
    .predict_task = function(task) {
      st <- self$state

      # if no stable weights, drop upstream LVs and block features, then return
      if (is.null(st$weights_stable) || !length(st$weights_stable)) {
        dt_all <- task$data()
        lm <- private$.lv_column_map(names(dt_all))
        env <- self$param_set$values$log_env
        blocks_map <- env$mbspls_state$blocks
        old_lv <- if (lm$K) unlist(lm$map, use.names = FALSE) else character(0)
        orig_feats <- intersect(unlist(blocks_map, use.names = FALSE), task$feature_names)
        keep_features <- setdiff(task$feature_names, c(old_lv, orig_feats))
        task$select(keep_features)
        return(task)
      }

      dt <- task$data()
      env <- self$param_set$values$log_env
      st_env <- env$mbspls_state
      blocks_map <- st_env$blocks
      all_blocks <- names(blocks_map)

      # Build X_test by blocks (fill missing features with zeros)
      X_test <- lapply(blocks_map, function(cols) {
        miss <- setdiff(cols, names(dt))
        if (length(miss)) for (m in miss) dt[, (m) := 0.0]
        m <- as.matrix(dt[, ..cols])
        storage.mode(m) <- "double"
        m
      })

      # Use stable weights + (if available) stable loadings to deflate
      W_stable <- st$weights_stable
      P_train  <- st$loadings_stable

      if (is.null(P_train) || !length(P_train)) {
        rec <- private$.recompute_scores_deflated(X_test, W_stable, all_blocks)
        T_pred_all_dt <- rec$T_mat
      } else {
        K <- length(W_stable)
        X_cur <- lapply(X_test, identity)
        tabs <- vector("list", K)

        for (k in seq_len(K)) {
          Tk <- matrix(0, nrow(X_cur[[1]]), length(all_blocks))
          for (bi in seq_along(all_blocks)) {
            b <- all_blocks[bi]
            w_b <- W_stable[[k]][[b]]
            if (is.null(w_b)) {
              Tk[, bi] <- 0
            } else {
              cols <- colnames(X_cur[[b]])
              if (!is.null(names(w_b))) {
                wv <- as.numeric(w_b[cols]); wv[is.na(wv)] <- 0
              } else {
                wv <- as.numeric(w_b)
                if (length(wv) != length(cols)) wv <- numeric(length(cols))
              }
              storage.mode(wv) <- "double"
              Tk[, bi] <- X_cur[[b]] %*% wv
            }
          }
          colnames(Tk) <- paste0("LV", k, "_", all_blocks)
          tabs[[k]] <- data.table::as.data.table(Tk)
          if (k < K) {
            for (bi in seq_along(all_blocks)) {
              b <- all_blocks[bi]
              pb <- P_train[[k]][[b]]
              if (is.null(pb)) next
              X_cur[[b]] <- X_cur[[b]] - Tk[, bi, drop = FALSE] %*% t(as.matrix(pb))
            }
          }
        }
        T_pred_all_dt <- do.call(cbind, tabs)
      }

      # keep only non-empty block columns from kept blocks
      keep_cols <- character(0)
      for (newk in seq_along(st$kept_blocks_per_comp)) {
        kb <- st$kept_blocks_per_comp[[newk]]
        if (length(kb)) keep_cols <- c(keep_cols, paste0("LV", newk, "_", kb))
      }

      T_pred_all <- as.matrix(T_pred_all_dt)
      T_pred_keep <- if (length(keep_cols)) {
        keep_cols <- intersect(keep_cols, colnames(T_pred_all))
        T_pred_all[, keep_cols, drop = FALSE]
      } else {
        matrix(0, nrow = nrow(T_pred_all), ncol = 0)
      }

      # ------- Drop upstream LVs and original block features; then append stable LVs
      dt_all <- task$data()
      lm <- private$.lv_column_map(names(dt_all))
      old_lv <- if (lm$K) unlist(lm$map, use.names = FALSE) else character(0)
      orig_feats <- intersect(unlist(blocks_map, use.names = FALSE), task$feature_names)
      keep_features <- setdiff(task$feature_names, c(old_lv, orig_feats))
      task$select(keep_features)
      if (ncol(T_pred_keep)) task$cbind(data.table::as.data.table(T_pred_keep))
      lgr$info("Bootstrap-select (predict): dropped %d original block features and %d upstream LV columns; kept %d stable LV columns.",
               length(orig_feats), length(old_lv), ncol(T_pred_keep))

      task
    }
  )
)

