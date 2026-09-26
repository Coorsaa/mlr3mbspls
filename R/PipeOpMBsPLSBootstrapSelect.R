#' MB-sPLS Bootstrap Selection with Two Methods ("ci" | "frequency")
#'
#' @title Post-hoc bootstrap feature selection
#'
#' @description
#' Performs a post-hoc **descriptive bootstrap** of an upstream
#' \code{po("mbspls")} fit and selects stable features. Each replicate resamples
#' the stored (column-centred) training blocks, re-centres the resampled rows on
#' their own column means and refits MB-sPLS with the upstream sparsity. The
#' replicate components are then matched to the training components and
#' sign-aligned block by block (see \code{align}), and the aligned weights are
#' summarised per feature. Features are selected via:
#' \itemize{
#'   \item \code{selection_method = "ci"} (default): keep if the percentile
#'     interval excludes 0 (\code{ci_lower > 0} or \code{ci_upper < 0}) AND
#'     |bootstrap mean| > \code{magnitude_threshold};
#'   \item \code{selection_method = "frequency"}: keep if the non-zero
#'     frequency among accepted replicates is \code{>= frequency_threshold}.
#' }
#' Blocks with no kept features **vanish** for that component, and components
#' with no kept feature in any block emit no LV columns. Component numbering is
#' **preserved**: \code{LVk_<block>} and \code{LC_0k} always refer to upstream
#' component \code{k}, so \code{LV2_*} can appear without \code{LV1_*};
#' \code{kept_components} lists the components that remain. Stable weights keep
#' one (possibly all-zero) entry per upstream component, so they stay aligned
#' with \code{weights_ci} and \code{weights_selectfreq} by label.
#'
#' If no component keeps any feature, the operator stops with an informative
#' error when \code{stability_only = FALSE} (a task without features would only
#' make the downstream learner fail). With \code{stability_only = TRUE} it warns,
#' stores the bootstrap summaries, and publishes no stable weights.
#'
#' Final weights are either the **original training weights** or the **mean
#' aligned bootstrap weights** of the selected features, controlled via
#' \code{stable_weight_source}. Training scores are recomputed with
#' **deflation** from those weights and replace the upstream LV columns.
#'
#' **Important:** Unless \code{stability_only = TRUE}, this operator uses the
#' original block features to recompute stable LV scores and then **drops those
#' block features and all upstream LV columns**, so only the kept stable LV
#' columns remain downstream, consistently at training and prediction time.
#' Prediction blocks are centred with the upstream training means
#' (\code{center} in the upstream state) before scoring.
#'
#' @section Bootstrap design:
#' Rows are resampled with replacement. If the task has an mlr3 \code{group}
#' column role and \code{bootstrap_groups} is \code{NULL}, whole groups of that
#' role are resampled instead (a cluster bootstrap), because rows of the same
#' group are not exchangeable. An explicit \code{bootstrap_groups} vector
#' overrides the role and must be equal to or coarser than it (e.g. families
#' over subjects). To bootstrap rows of a grouped task, remove the group role.
#' With \code{stratify_by_block}, units are resampled within strata. The
#' resampling design is validated once per fit; each replicate draws from its
#' own RNG stream when \code{seed_bootstrap} is set.
#'
#' The number of exchangeability units (rows or groups) is checked before any
#' replicate is drawn. Training stops with an error if no unit can vary between
#' replicates: a single unit overall, or a single unit in every stratum. Every
#' replicate would then equal the training data, and every interval would have
#' zero width. This typically happens when a group role that defines the outer
#' resampling (e.g. \code{site} for leave-site-out CV) leaves only one group in
#' a training set. A warning is raised, and \code{few_exchangeability_units} is
#' set in the state, when there are fewer than 10 units or fewer distinct
#' bootstrap samples than replicates. With \eqn{G} units there are
#' \code{choose(2G - 1, G)} distinct samples, multiplied over strata.
#' Intervals and frequencies from so few units are unreliable.
#'
#' The percentile intervals in \code{weights_ci} are type-7 sample quantiles
#' (\code{stats::quantile()} default) of the sign-aligned accepted draws at level
#' \code{1 - alpha}. They describe sampling variability of the sparse weights
#' and are not hypothesis tests.
#'
#' @section Parameters:
#' Hyperparameters are defined in the object's \code{param_set} and can be set
#' via \code{param_vals}. This PipeOp is designed to be placed downstream of
#' \code{po("mbspls")} and uses its \code{log_env}.
#'
#' * `log_env`: Environment shared with upstream \code{po("mbspls")} (required).
#' * `bootstrap`: Run bootstrap selection (default \code{TRUE}). With
#'   \code{FALSE} no selection is made and only the upstream LV columns are
#'   kept, at training and prediction time.
#' * `stability_only`: Logical; if TRUE, run the bootstrap and selection
#'   computations and store the stability outputs, but do **not** modify the
#'   task: upstream LV columns and original block features pass through
#'   unchanged at training and prediction time. The flag is recorded in the
#'   shared \code{log_env} state; \code{po("mbspls")} then evaluates its
#'   prediction payload with the raw weights under
#'   \code{predict_weights = "auto"} and rejects explicit stable requests,
#'   because its emitted LV columns are the raw-weight scores. Default
#'   \code{FALSE}.
#' * `B`: Bootstrap replicates; at least two are required when bootstrap
#'   selection is enabled (default \code{500}).
#' * `alpha`: Interval alpha strictly between zero and one (default
#'   \code{0.05}, i.e. 95\% percentile intervals). Stored in the state.
#' * `align`: Sign alignment of replicate weights to the training weights.
#'   The reported MB-sPLS criterion depends only on absolute (or squared)
#'   cross-block correlations, and a refit can differ from the training fit in
#'   its relative block orientation; both modes therefore choose one sign per
#'   component and block.
#'   \code{"block_sign"} (default) uses the sign of the inner product between
#'   the replicate and training weights of the block and falls back to the
#'   score correlation when the supports are disjoint (inner product zero).
#'   \code{"score_correlation"} uses the sign of the correlation between the
#'   replicate block score and the training block score on the resampled rows
#'   and falls back to the weight inner product. If neither is informative the
#'   replicate sign is kept; fallbacks and unresolved signs are counted in
#'   \code{alignment_diagnostics}.
#' * `selection_method`: \code{"ci"} (default) or \code{"frequency"}.
#' * `frequency_threshold`: Only for \code{"frequency"}; default \code{0.60}.
#' * `magnitude_threshold`: Non-negative numeric, used only with
#'   \code{selection_method = "ci"}: a feature is kept only if its interval
#'   excludes 0 and |bootstrap mean| exceeds this value. Default \code{1e-3}.
#' * `stable_weight_source`: Either \code{"training"} (default) or
#'   \code{"bootstrap_mean"}. With \code{"training"} the stable weights are the
#'   training weights restricted to features that are both non-zero in the
#'   training fit and bootstrap-selected: the support is the intersection, so
#'   it can remove but never add features, and a block vanishes if none of its
#'   selected features has a non-zero training weight. Selected features with a
#'   zero training weight are recorded in \code{selected_not_in_training} and
#'   logged. With \code{"bootstrap_mean"} the support equals
#'   the bootstrap-selected set and the values are the aligned bootstrap means.
#' * `stratify_by_block`: Optional name of a retained MB-sPLS block that
#'   dummy-codes a single factor (e.g. \code{"Studygroup"}); units are then
#'   resampled within its levels. Full one-hot coding (one column per level,
#'   exactly one active per row) and treatment coding (reference level = all
#'   columns inactive) are both recognised, also after centring or scaling.
#'   Every column must take exactly two values and no row may have more than
#'   one active indicator. Unknown names, blocks dropped upstream (no numeric,
#'   non-constant feature) and blocks that are not dummy-coded are errors.
#'   Strata must be constant within bootstrap groups.
#' * `bootstrap_groups`: Optional exchangeability-group vector. If named, its
#'   names must cover the task row IDs and it is aligned by row ID; otherwise it
#'   must already follow training-row order. Whole groups are sampled with
#'   replacement. If \code{NULL} (default) and the task has a \code{group}
#'   column role, the groups of that role are used. An explicit vector must be
#'   equal to or coarser than the task's group role. To resample finer units
#'   (e.g. subjects within a site role), remove the group role before this
#'   operator. The source is recorded in \code{bootstrap_group_source}.
#'   Training stops if only one group can be resampled, and warns if there are
#'   fewer than 10 groups (see the section on the bootstrap design).
#' * `min_score_cor`: Numeric in `[0, 1]`. Minimum mean absolute score correlation
#'   between a bootstrap replicate component and the corresponding training reference
#'   component required for the replicate to be accepted into the summary statistics.
#'   Replicates below this threshold are excluded to reduce noise from uninformative
#'   fits. Default `0.10`. Increase for high-noise data.
#' * `min_effective_fraction`: Numeric in `[0, 1]`. A warning is raised for
#'   components whose accepted replicates make up less than this fraction of
#'   \code{B}, and for components without any accepted replicate. These
#'   components are listed in \code{components_below_effective_floor}: their
#'   intervals and frequencies rest on few, selectively accepted replicates, or
#'   are empty. Default \code{0.5}.
#' * `seed_bootstrap`: Optional positive integer. When set, each replicate
#'   gets its own deterministic RNG stream (\code{\link[=mb_rng_streams]{mb_rng_streams()}}),
#'   so results do not depend on \code{workers} and the caller's RNG state is
#'   left unchanged. \code{NULL} (default, as for the other seed parameters of
#'   the package) uses the ambient RNG.
#' * `workers`: Integer. Requested number of parallel workers for the bootstrap loop.
#'   Requires \pkg{future} and \pkg{future.apply} when set to a value larger than 1; otherwise an explicit error is raised. Default 1L.
#'
#' @return Replaces the task's LV columns with the stable LV columns of the kept
#' components (numbering preserved). The state stores \code{weights_ci}
#' (aligned bootstrap means, SDs and percentile intervals),
#' \code{weights_selectfreq}, \code{weights_stable}, \code{loadings_stable},
#' \code{kept_components}, \code{kept_blocks_per_comp}, \code{selection}
#' (per feature: bootstrap selection, training and stable weight),
#' \code{selected_not_in_training}, \code{n_eff_by_component} (accepted
#' replicates, their fraction of \code{B} and non-converged refits per
#' component), \code{alignment_diagnostics}, \code{n_nonconverged_replicates},
#' \code{components_below_effective_floor}, the bootstrap design
#' (\code{bootstrap_grouped}, \code{bootstrap_group_source},
#' \code{n_exchangeability_units}, \code{exchangeability_units_by_stratum},
#' \code{few_exchangeability_units}, \code{stratify_by_block},
#' \code{strata_sizes}, \code{rng_streams}), the upstream training centre
#' \code{center} (subtracted from prediction blocks) and the settings
#' \code{bootstrap},
#' \code{alpha},
#' \code{alignment_method}, \code{component_matching},
#' \code{selection_method}, \code{frequency_threshold},
#' \code{magnitude_threshold}, \code{stable_weight_source},
#' \code{min_score_cor} and \code{stability_only}.
#'
#' @import data.table
#' @importFrom R6 R6Class
#' @importFrom lgr lgr
#' @importFrom paradox ps p_uty p_fct p_lgl p_int p_dbl
#' @importFrom mlr3pipelines PipeOpTaskPreproc
#' @importFrom parallel detectCores
#' @importFrom stats cor sd var quantile setNames
#' @export
PipeOpMBsPLSBootstrapSelect = R6::R6Class(
  "PipeOpMBsPLSBootstrapSelect",
  inherit = mlr3pipelines::PipeOpTaskPreproc,
  public = list(
    #' @description Initialize the PipeOpMBsPLSBootstrapSelect.
    #' @param id character(1). Identifier of the resulting object.
    #' @param param_vals named list. List of hyperparameter settings.
    initialize = function(id = "mbspls_bootstrap_select", param_vals = list()) {
      ps = paradox::ps(
        log_env = paradox::p_uty(tags = c("train", "predict"), default = NULL),

        bootstrap = paradox::p_lgl(default = TRUE, tags = "train"),
        stability_only = paradox::p_lgl(default = FALSE, tags = c("train", "predict")),
        B = paradox::p_int(lower = 1L, default = 500L, tags = "train"),
        alpha = paradox::p_dbl(lower = 0, upper = 1, default = 0.05, tags = "train"),

        align = paradox::p_fct(levels = c("score_correlation", "block_sign"),
          default = "block_sign", tags = "train"),

        selection_method = paradox::p_fct(levels = c("ci", "frequency"), default = "ci", tags = "train"),
        frequency_threshold = paradox::p_dbl(lower = 0, upper = 1, default = 0.60, tags = "train"),
        magnitude_threshold = paradox::p_dbl(lower = 0, default = 1e-3, tags = "train"),

        stable_weight_source = paradox::p_fct(
          levels  = c("training", "bootstrap_mean"),
          default = "training",
          tags    = "train"
        ),

        stratify_by_block = paradox::p_uty(default = NULL, tags = "train"),
        bootstrap_groups = paradox::p_uty(default = NULL, tags = "train"),
        min_score_cor = paradox::p_dbl(lower = 0, upper = 1, default = 0.10, tags = "train"),
        min_effective_fraction = paradox::p_dbl(lower = 0, upper = 1, default = 0.5, tags = "train"),
        seed_bootstrap = paradox::p_int(lower = 1L, default = NULL, tags = "train", special_vals = list(NULL)),
        workers = paradox::p_int(lower = 1L, default = 1L, tags = "train")
      )

      super$initialize(id = id, param_set = ps, param_vals = param_vals)
      self$packages = "mlr3mbspls"
    }
  ),

  private = list(

    # ---------- utils ----------
    .lv_column_map = function(dt_names) {
      lv_cols = grep("^LV\\d+_", dt_names, value = TRUE)
      if (!length(lv_cols)) {
        return(list(K = 0L, blocks = character(0), map = list()))
      }
      comps = as.integer(sub("^LV(\\d+)_.*$", "\\1", lv_cols, perl = TRUE))
      blocks = sub("^LV\\d+_", "", lv_cols)
      K = max(comps)
      bset = unique(blocks)
      map = lapply(seq_len(K), function(k) {
        sel = (comps == k)
        stats::setNames(lv_cols[sel], blocks[sel])
      })
      list(K = K, blocks = bset, map = map)
    },

    .finalize_scores_only = function(task, blocks_map = NULL) {
      # Defensive: drop raw block features if they are present (append leakage)
      if (!is.null(blocks_map) && length(blocks_map)) {
        raw = intersect(unlist(blocks_map, use.names = FALSE), task$feature_names)
        if (length(raw)) {
          task$select(setdiff(task$feature_names, raw))
        }
      }

      # Enforce score-space only output (what users expect from MB-sPLS transformer)
      lv = grep("^LV\\d+_", task$feature_names, value = TRUE)
      if (!length(lv)) {
        lgr$warn("%s: no LV columns present after bootstrap_select; dropping all features to avoid raw-feature leakage.",
          self$id
        )
        task$select(character(0))
        return(task)
      }

      task$select(lv)
      task
    },

    .get_env_state = function(pv, run_id = NULL) {
      env = pv$log_env
      if (!inherits(env, "environment")) {
        stop("Provide a shared 'log_env' with po('mbspls').", call. = FALSE)
      }
      .mbspls_state_from_env(env, run_id = run_id, require_train_blocks = FALSE, where = "log_env")
    },

    # Validate `stratify_by_block` against the blocks of the upstream fit.
    .check_stratify_block = function(stratify_block, X_blocks_train) {
      if (is.null(stratify_block)) {
        return(NULL)
      }
      if (!checkmate::test_string(stratify_block, min.chars = 1L)) {
        stop("`stratify_by_block` must be NULL or a single non-empty block name.", call. = FALSE)
      }
      if (!stratify_block %in% names(X_blocks_train)) {
        stop(sprintf(
          paste(
            "`stratify_by_block` = '%s' is not a block of the upstream MB-sPLS fit, so the bootstrap cannot be stratified.",
            "Retained blocks: %s. Blocks without any numeric, non-constant training feature are dropped upstream."
          ),
          stratify_block, mb_format_truncated(names(X_blocks_train))
        ), call. = FALSE)
      }
      stratify_block
    },

    # Resolve the exchangeability groups: an explicit `bootstrap_groups`
    # vector (aligned by row ID when named), else the task's group role, else
    # individual rows.
    .resolve_bootstrap_groups = function(bootstrap_groups, task) {
      task_row_ids = as.character(task$row_ids)
      source = "rows"
      if (!is.null(bootstrap_groups)) {
        group_names = names(bootstrap_groups)
        if (!is.null(group_names)) {
          if (anyNA(group_names) || any(!nzchar(group_names)) ||
            anyDuplicated(group_names)) {
            stop(
              "`bootstrap_groups` names must be unique, non-missing task row IDs.",
              call. = FALSE
            )
          }
          missing_ids = setdiff(task_row_ids, group_names)
          if (length(missing_ids)) {
            stop(sprintf(
              "`bootstrap_groups` is missing task row IDs: %s.",
              mb_format_truncated(missing_ids)
            ), call. = FALSE)
          }
          bootstrap_groups = bootstrap_groups[match(task_row_ids, group_names)]
        } else if (length(bootstrap_groups) != task$nrow) {
          stop(sprintf(
            "Unnamed `bootstrap_groups` has length %d but the training task has %d rows.",
            length(bootstrap_groups), task$nrow
          ), call. = FALSE)
        }
        if (anyNA(bootstrap_groups)) {
          stop("`bootstrap_groups` must not contain missing values.",
            call. = FALSE)
        }
        source = "explicit"
      }

      role_groups = mb_task_group_vector(task)
      if (!is.null(role_groups)) {
        role_column = task$col_roles$group
        if (is.null(bootstrap_groups)) {
          bootstrap_groups = role_groups
          source = "task_group_role"
          lgr$info("Bootstrap selection resamples whole groups of the task's group role '%s'.", role_column)
        } else {
          # Each role group must map to exactly one explicit group.
          pairs = unique(data.frame(
            role = as.character(role_groups),
            explicit = as.character(bootstrap_groups),
            stringsAsFactors = FALSE
          ))
          split_roles = unique(pairs$role[duplicated(pairs$role)])
          if (length(split_roles)) {
            stop(sprintf(
              paste(
                "`bootstrap_groups` splits rows that share a value of the task's group role '%s' (%s).",
                "An explicit grouping must be equal to or coarser than the group role."
              ),
              role_column, mb_format_truncated(split_roles)
            ), call. = FALSE)
          }
        }
      }
      list(
        groups = bootstrap_groups,
        source = source,
        column = if (identical(source, "task_group_role")) task$col_roles$group else NULL
      )
    },

    # helper to construct stable weights & kept blocks from summaries
    .build_stable_from = function(
      method, K, bn, blocks_map, sum_df, freq_df, frequency_threshold,
      W_train, weight_source = c("training", "bootstrap_mean"),
      magnitude_threshold = 1e-3
    ) {
      weight_source = match.arg(weight_source)
      W_stable_local = list()
      kept_blocks_per_comp_local = list()
      selection_rows = list()

      for (k in seq_len(K)) {
        k_lab = sprintf("LC_%02d", k)
        Wk_out = vector("list", length(bn))
        names(Wk_out) = bn
        kept_blocks = character(0)

        for (b in bn) {
          feats = blocks_map[[b]]
          if (is.null(feats) || !length(feats)) {
            # keep a zero-length named numeric to be safe
            Wk_out[[b]] = setNames(numeric(0), character(0))
            next
          }

          # means + CIs for this (k,b)
          sb = sum_df[sum_df$component == k_lab & sum_df$block == b,
            c("feature", "boot_mean", "ci_lower", "ci_upper"),
            drop = FALSE]
          mu_map = if (nrow(sb)) stats::setNames(sb$boot_mean, sb$feature) else setNames(numeric(0), character(0))
          lo_map = if (nrow(sb)) stats::setNames(sb$ci_lower, sb$feature) else setNames(numeric(0), character(0))
          hi_map = if (nrow(sb)) stats::setNames(sb$ci_upper, sb$feature) else setNames(numeric(0), character(0))

          mu = as.numeric(mu_map[feats])
          if (length(mu) == 0L) mu = numeric(length(feats))
          lo = as.numeric(lo_map[feats])
          if (length(lo) == 0L) lo = numeric(length(feats))
          hi = as.numeric(hi_map[feats])
          if (length(hi) == 0L) hi = numeric(length(feats))
          mu[is.na(mu)] = 0
          lo[is.na(lo)] = 0
          hi[is.na(hi)] = 0

          if (identical(method, "ci")) {
            keep = ((lo > 0) | (hi < 0)) & (abs(mu) > magnitude_threshold)
          } else {
            fb = freq_df[freq_df$component == k_lab & freq_df$block == b,
              c("feature", "freq"), drop = FALSE]
            fq_map = if (nrow(fb)) {
              stats::setNames(fb$freq, fb$feature)
            } else {
              stats::setNames(numeric(0), character(0))
            }
            fv = as.numeric(fq_map[feats])
            if (length(fv) == 0L) fv = numeric(length(feats))
            fv[is.na(fv)] = 0
            keep = (fv >= as.numeric(frequency_threshold))
          }

          has_train = !is.null(W_train) && length(W_train) >= k &&
            !is.null(W_train[[k]]) && !is.null(W_train[[k]][[b]])
          w_train = if (has_train) {
            as.numeric(mb_align_named_numeric(
              W_train[[k]][[b]],
              cols = feats,
              context = sprintf("W_train[['%s']][['%s']]", k_lab, b)
            ))
          } else {
            rep(NA_real_, length(feats))
          }

          ## ---- choose base values for kept features ------------------------
          if (identical(weight_source, "training")) {
            if (!has_train) {
              stop(
                sprintf(
                  "stable_weight_source='training' requested but training weights are unavailable for component '%s', block '%s'. Refit the upstream MB-sPLS PipeOp and keep its state intact.",
                  k_lab, b
                ),
                call. = FALSE
              )
            }
            # The support is the intersection of the training support and the
            # bootstrap-selected set.
            val = w_train
          } else {
            val = mu
          }

          # zero out non-selected features, regardless of source
          val[!keep] = 0

          # Always create an entry for *every* block with correct length & names
          Wk_out[[b]] = stats::setNames(val, feats)
          if (any(val != 0)) kept_blocks = c(kept_blocks, b)
          selection_rows[[length(selection_rows) + 1L]] = data.table::data.table(
            component = k_lab, block = b, feature = feats,
            bootstrap_selected = keep, training_weight = w_train,
            stable_weight = val
          )
        }

        # Keep all-zero components as placeholders so that labels stay aligned
        # with weights_ci / weights_selectfreq; they have zero scores and
        # loadings and leave the deflation path unchanged.
        W_stable_local[[length(W_stable_local) + 1L]] = Wk_out
        kept_blocks_per_comp_local[[length(kept_blocks_per_comp_local) + 1L]] = kept_blocks
      }

      names(W_stable_local) = sprintf("LC_%02d", seq_along(W_stable_local))
      names(kept_blocks_per_comp_local) = sprintf("LC_%02d", seq_along(W_stable_local))
      selection = if (length(selection_rows)) {
        data.table::rbindlist(selection_rows, use.names = TRUE)
      } else {
        data.table::data.table(
          component = character(), block = character(), feature = character(),
          bootstrap_selected = logical(), training_weight = numeric(),
          stable_weight = numeric()
        )
      }
      list(W = W_stable_local, kept = kept_blocks_per_comp_local, selection = selection)
    },

    .recompute_scores_deflated = function(X_list, W_list, all_blocks) {
      B = length(all_blocks)
      K = length(W_list)
      X_cur = lapply(X_list, identity)
      T_tabs = vector("list", K)
      P_all = vector("list", K)

      for (k in seq_len(K)) {
        Tk = matrix(0, nrow(X_cur[[1]]), B)
        Pk = vector("list", B)
        names(Pk) = all_blocks

        for (bi in seq_along(all_blocks)) {
          b = all_blocks[bi]
          w_b = W_list[[k]][[b]]
          if (is.null(w_b)) {
            stop(
              sprintf("Stable weights are missing for component %d, block '%s'. Stable score recomputation requires a complete block-wise weight structure.", k, b),
              call. = FALSE
            )
          }
          cols = colnames(X_cur[[b]])
          wv = as.numeric(mb_align_named_numeric(
            w_b,
            cols = cols,
            context = sprintf("W_list[[%d]][['%s']]", k, b)
          ))
          storage.mode(wv) = "double"
          Tk[, bi] = X_cur[[b]] %*% wv
          denom = sum(Tk[, bi] * Tk[, bi])
          if (denom <= 0) {
            Pk[[b]] = setNames(rep(0, ncol(X_cur[[b]])), cols)
          } else {
            pb = drop(crossprod(X_cur[[b]], Tk[, bi]) / denom)
            names(pb) = cols
            if (anyNA(pb) || any(!is.finite(pb))) {
              stop(
                sprintf("Recomputed loadings for component %d, block '%s' contain NA/Inf values.", k, b),
                call. = FALSE
              )
            }
            Pk[[b]] = pb
          }
        }
        colnames(Tk) = paste0("LV", k, "_", all_blocks)
        T_tabs[[k]] = data.table::as.data.table(Tk)
        P_all[[k]] = Pk

        if (k < K) { # deflate
          for (bi in seq_along(all_blocks)) {
            b = all_blocks[bi]
            t_b = Tk[, bi, drop = FALSE]
            pb = Pk[[b]]
            X_cur[[b]] = X_cur[[b]] - t_b %*% t(as.matrix(pb))
          }
        }
      }
      names(P_all) = names(W_list)
      list(T_mat = do.call(cbind, T_tabs), P = P_all)
    },

    # --------------- bootstrap core (alignment + acceptance + summary) --------
    .bootstrap_align_and_summarise = function(
      X_list, W_ref, blocks, ncomp,
      sparsity, corr_method = "pearson", perf_metric = "mac",
      B = 500L, alpha = 0.05, align = "block_sign",
      workers = 1L, stratify_block = NULL,
      groups = NULL, rng_streams = NULL,
      min_score_cor = 0.10, min_effective_fraction = 0.5,
      group_source = NULL, group_column = NULL
    ) {
      MIN_SCORE_COR = as.numeric(min_score_cor %||% 0.10) # acceptance gate
      if (!align %in% c("block_sign", "score_correlation")) {
        stop(sprintf("Unknown align mode: %s", align), call. = FALSE)
      }

      # match blocks
      bn = intersect(names(X_list), names(W_ref[[1]]))
      if (!length(bn)) stop("bootstrap: cannot match blocks between X_list and reference weights")
      X_list = X_list[bn]
      W_ref = lapply(W_ref, function(wk) wk[bn])
      comp_lab = sprintf("LC_%02d", seq_len(ncomp))
      N = nrow(X_list[[1]])
      if (!all(vapply(X_list, nrow, integer(1)) == N)) {
        stop("bootstrap: all blocks in 'X_list' must have the same number of rows.", call. = FALSE)
      }
      if (!is.null(groups) && (length(groups) != N || anyNA(groups))) {
        stop(
          "bootstrap: `groups` must contain one non-missing exchangeability identifier per training row.",
          call. = FALSE
        )
      }
      if (!is.null(rng_streams) &&
        (!is.list(rng_streams) || length(rng_streams) != B)) {
        stop("bootstrap: `rng_streams` must contain one stream per replicate.",
          call. = FALSE)
      }

      pad_to_order = function(w_boot, w_ref_named) {
        all_feat = names(w_ref_named)
        out = numeric(length(all_feat))
        names(out) = all_feat
        if (!is.null(names(w_boot))) {
          out[names(w_boot)] = w_boot
        }
        out
      }

      # Each loading vector acts on the residual after earlier components.
      # Applying later weights to the original blocks changes component matching.
      reference_scores = as.matrix(
        private$.recompute_scores_deflated(X_list, W_ref, bn)$T_mat
      )
      T_ref = lapply(seq_len(ncomp), function(k) {
        lapply(seq_along(bn), function(bi) {
          reference_scores[, (k - 1L) * length(bn) + bi]
        })
      })

      strata = NULL
      if (!is.null(stratify_block)) {
        if (!checkmate::test_string(stratify_block) || !stratify_block %in% bn) {
          stop(sprintf(
            "bootstrap: `stratify_block` must name one of the blocks %s.",
            mb_format_truncated(bn)
          ), call. = FALSE)
        }
        strata = .mb_strata_from_dummy_block(X_list[[stratify_block]], stratify_block)
        lgr$info("Stratified bootstrap by '%s' with %d strata.", stratify_block, nlevels(strata))
      }

      # Validate the resampling design once; replicates only draw from it.
      # Without groups, every row is its own unit.
      design = if (!is.null(groups)) {
        .mb_cluster_design(groups, strata)
      } else if (!is.null(strata)) {
        .mb_cluster_design(seq_len(N), strata)
      } else {
        NULL
      }
      units = .mb_bootstrap_unit_check(
        design, n_rows = N, B = B,
        group_source = group_source %||% if (is.null(groups)) "rows" else "explicit",
        group_column = group_column, stratify_block = stratify_block
      )

      fit_once = function(Xb) {
        if (!is.null(sparsity) && identical(sparsity$type, "c_matrix")) {
          cpp_mbspls_multi_lv_cmatrix(
            X_blocks = Xb, c_matrix = sparsity$c_matrix,
            max_iter = 600L, tol = 1e-4,
            spearman = (corr_method == "spearman"),
            do_perm = FALSE, n_perm = 0L, alpha = 0.05,
            frobenius = (perf_metric == "frobenius")
          )
        } else {
          c_vec = if (!is.null(sparsity) && !is.null(sparsity$c_vec)) {
            sparsity$c_vec
          } else {
            stop("bootstrap: sparsity$c_vec missing")
          }
          cpp_mbspls_multi_lv(
            X_blocks = Xb,
            c_constraints = as.numeric(c_vec[bn]),
            K = ncomp,
            max_iter = 600L,
            tol = 1e-4,
            spearman = (corr_method == "spearman"),
            do_perm = FALSE, n_perm = 0L, alpha = 0.05,
            frobenius = (perf_metric == "frobenius")
          )
        }
      }

      cor_safe = function(x, y) {
        if (length(unique(x)) < 2 || length(unique(y)) < 2) {
          return(NA_real_)
        }
        suppressWarnings(stats::cor(x, y, method = corr_method))
      }
      sign_or_na = function(v, tol) {
        if (length(v) == 1L && is.finite(v) && abs(v) > tol) sign(v) else NA_real_
      }
      eps = sqrt(.Machine$double.eps)
      n_blocks = length(bn)

      one_rep = function(r) {
        idx = if (is.null(design)) {
          sample.int(N, replace = TRUE)
        } else {
          .mb_draw_cluster(design)$indices
        }

        # The reference fit is centred on the training rows; centre each
        # replicate on its own resampled rows before refitting.
        Xb = lapply(X_list, function(X) {
          Xr = X[idx, , drop = FALSE]
          Xr - rep(colMeans(Xr), each = nrow(Xr))
        })
        fit_r = tryCatch(
          fit_once(Xb),
          error = function(e) {
            stop(
              sprintf("Bootstrap replicate %d failed while refitting MB-sPLS: %s", r, conditionMessage(e)),
              call. = FALSE
            )
          }
        )
        W_r = fit_r$W
        Kfit = length(W_r)
        if (!Kfit) {
          stop(
            sprintf("Bootstrap replicate %d returned zero extracted components.", r),
            call. = FALSE
          )
        }
        W_r = lapply(W_r, function(wk) {
          if (is.null(names(wk))) {
            names(wk) = bn
          }
          wk[bn]
        })

        # The solver returns scores from the fitted sequential deflation path.
        T_boot = lapply(seq_len(Kfit), function(i) {
          lapply(seq_along(bn), function(bi) {
            fit_r$T_mat[, (i - 1L) * n_blocks + bi]
          })
        })

        # match comps by mean |cor| across blocks
        K = ncomp
        Kc = min(Kfit, K)
        S = matrix(0, K, K)
        for (i in seq_len(Kc)) {
          for (k in seq_len(K)) {
            cs = vapply(seq_along(bn), function(bi) {
              r_ = cor_safe(T_boot[[i]][[bi]], T_ref[[k]][[bi]][idx])
              if (is.finite(r_)) abs(r_) else NA_real_
            }, numeric(1))
            cs = cs[is.finite(cs)]
            S[i, k] = if (length(cs)) mean(cs) else 0
          }
        }
        perm = .mb_assignment_max(S)

        out_list = list()
        ptr = 0L
        n_eff = integer(K)
        n_fallback = matrix(0L, K, n_blocks)
        n_unresolved = matrix(0L, K, n_blocks)
        n_nonconverged = integer(K)
        converged = fit_r$converged

        for (i in seq_len(Kc)) {
          k = perm[i]
          if (length(converged) >= i && isFALSE(as.logical(converged[[i]]))) {
            n_nonconverged[k] = n_nonconverged[k] + 1L
          }

          # ---------- PER-BLOCK SIGN ALIGNMENT ----------
          fallback = logical(n_blocks)
          unresolved = logical(n_blocks)
          for (bi in seq_along(bn)) {
            b = bn[bi]
            w_ref_b = W_ref[[k]][[b]]
            w_boot_b = W_r[[i]][[b]]
            if (is.null(names(w_boot_b)) && length(w_boot_b) == length(w_ref_b)) {
              names(w_boot_b) = names(w_ref_b)
            }
            w_pad = pad_to_order(w_boot_b, w_ref_b)
            w_ref_num = as.numeric(w_ref_b)
            s_dot = sign_or_na(
              sum(w_pad * w_ref_num),
              eps * sqrt(sum(w_pad^2) * sum(w_ref_num^2))
            )
            s_cor = sign_or_na(cor_safe(T_boot[[i]][[bi]], T_ref[[k]][[bi]][idx]), eps)
            s_primary = if (align == "block_sign") s_dot else s_cor
            s_fallback = if (align == "block_sign") s_cor else s_dot
            s_b = if (!is.na(s_primary)) {
              s_primary
            } else if (!is.na(s_fallback)) {
              fallback[bi] = TRUE
              s_fallback
            } else {
              unresolved[bi] = TRUE
              1
            }
            T_boot[[i]][[bi]] = s_b * T_boot[[i]][[bi]]
            W_r[[i]][[b]] = s_b * w_pad
          }
          # ---------- END ALIGNMENT ----------

          # acceptance gate on aligned scores
          sc = vapply(seq_along(bn), function(bi) {
            suppressWarnings(abs(stats::cor(T_boot[[i]][[bi]], T_ref[[k]][[bi]][idx], method = corr_method)))
          }, numeric(1))
          sc = sc[is.finite(sc)]
          if (!length(sc) || mean(sc) < MIN_SCORE_COR) next
          n_eff[k] = n_eff[k] + 1L
          n_fallback[k, ] = n_fallback[k, ] + fallback
          n_unresolved[k, ] = n_unresolved[k, ] + unresolved

          # record aligned weights (including zeros)
          for (b in bn) {
            wb = W_r[[i]][[b]]
            ptr = ptr + 1L
            out_list[[ptr]] = data.table::data.table(
              replicate = r, component = sprintf("LC_%02d", k), block = b,
              feature = names(wb), weight = as.numeric(wb)
            )
          }
        }

        list(
          draws = if (length(out_list)) data.table::rbindlist(out_list, use.names = TRUE, fill = TRUE) else NULL,
          n_eff = n_eff,
          n_fallback = n_fallback,
          n_unresolved = n_unresolved,
          n_nonconverged = n_nonconverged
        )
      }

      rep_idx = seq_len(B)
      if (workers > 1L) {
        .mbspls_require_suggested(
          c("future", "future.apply"),
          "PipeOpMBsPLSBootstrapSelect with workers > 1 (or set workers = 1)"
        )
      }

      rep_res = if (workers > 1L) {

        old_plan = future::plan()
        on.exit(future::plan(old_plan), add = TRUE)

        # If the current plan is sequential, switch temporarily to multisession
        # for cross-platform parallelism (incl. Windows).
        if (future::nbrOfWorkers() <= 1L) {
          future::plan(future::multisession, workers = workers)
        } else if (future::nbrOfWorkers() != workers) {
          lgr$debug("bootstrap_select: future plan already has %d workers; ignoring workers=%d.",
            future::nbrOfWorkers(), workers)
        }

        lgr$info("bootstrap_select: launching %d bootstrap replicates across %d workers.", B, workers)
        run_parallel = function() {
          future.apply::future_lapply(
            rep_idx, one_rep,
            future.seed = if (is.null(rng_streams)) TRUE else rng_streams,
            future.packages = c("mlr3mbspls")
          )
        }
        raw_res = tryCatch(
          if (is.null(rng_streams)) {
            run_parallel()
          } else {
            # future_lapply advances the caller's RNG even with explicit seeds.
            with_rng_stream_local(rng_streams[[1L]], run_parallel)
          },
          error = function(e) {
            stop(sprintf(
              "bootstrap_select: parallel bootstrap failed. Consider setting workers=1 to diagnose. Original error: %s",
              conditionMessage(e)
            ), call. = FALSE)
          }
        )
        raw_res
      } else {
        lgr$info("bootstrap_select: running %d bootstrap replicates sequentially.", B)
        if (is.null(rng_streams)) {
          lapply(rep_idx, one_rep)
        } else {
          lapply(rep_idx, function(r) {
            with_rng_stream_local(rng_streams[[r]], function() one_rep(r))
          })
        }
      }

      K = length(comp_lab)
      n_eff = Reduce(`+`, lapply(rep_res, `[[`, "n_eff"), init = integer(K))
      n_fallback = Reduce(`+`, lapply(rep_res, `[[`, "n_fallback"), init = matrix(0L, K, n_blocks))
      n_unresolved = Reduce(`+`, lapply(rep_res, `[[`, "n_unresolved"), init = matrix(0L, K, n_blocks))
      n_nonconverged = Reduce(`+`, lapply(rep_res, `[[`, "n_nonconverged"), init = integer(K))
      n_nonconverged_replicates = sum(vapply(rep_res, function(x) any(x$n_nonconverged > 0L), logical(1)))

      n_eff_by_component = data.table::data.table(
        component = comp_lab,
        n_eff = as.integer(n_eff),
        fraction_effective = as.numeric(n_eff) / B,
        n_nonconverged = as.integer(n_nonconverged)
      )
      alignment_diagnostics = data.table::data.table(
        component = rep(comp_lab, each = n_blocks),
        block = rep(bn, times = K),
        n_accepted = rep(as.integer(n_eff), each = n_blocks),
        n_fallback = as.integer(t(n_fallback)),
        n_unresolved = as.integer(t(n_unresolved))
      )

      # Components with no or few accepted replicates: their summaries are
      # empty or rest on a small, selectively accepted subset.
      floor_n = ceiling(as.numeric(min_effective_fraction) * B)
      below_floor = comp_lab[n_eff == 0L | n_eff < floor_n]
      if (length(below_floor)) {
        warning(sprintf(
          paste(
            "bootstrap_select: few accepted replicates for component(s) %s (%s of B=%d; min_score_cor=%.2f, min_effective_fraction=%.2f).",
            "Their intervals and frequencies rest on a small, selectively accepted subset, and are empty without any accepted replicate.",
            "Consider lowering min_score_cor."
          ),
          paste(below_floor, collapse = ", "),
          paste(n_eff[match(below_floor, comp_lab)], collapse = ", "),
          B, MIN_SCORE_COR, as.numeric(min_effective_fraction)
        ), call. = FALSE)
      }
      if (n_nonconverged_replicates > 0L) {
        bad = n_nonconverged > 0L
        warning(sprintf(
          "bootstrap_select: %d of %d replicate refits did not converge within 600 iterations (component: count %s). Their weights are included in the summaries.",
          n_nonconverged_replicates, B,
          paste(sprintf("%s: %d", comp_lab[bad], n_nonconverged[bad]), collapse = ", ")
        ), call. = FALSE)
      }
      n_unres_total = sum(n_unresolved)
      if (n_unres_total > 0L) {
        lgr$info("bootstrap_select: %d accepted block alignment(s) had no informative sign; the replicate sign was kept.",
          n_unres_total)
      }

      # gather draws
      dlist = Filter(Negate(is.null), lapply(rep_res, `[[`, "draws"))
      draws = if (length(dlist)) {
        data.table::rbindlist(dlist, use.names = TRUE, fill = TRUE)
      } else {
        data.table::data.table()
      }
      if (nrow(draws)) {
        draws[, component := as.character(component)]
        draws[, block := as.character(block)]
      }

      # selection frequency (non-zero among accepted)
      sel_grid = data.table::rbindlist(lapply(seq_len(ncomp), function(k) {
        data.table::rbindlist(lapply(bn, function(b) {
          data.table::data.table(component = comp_lab[k], block = b, feature = names(W_ref[[k]][[b]]))
        }), use.names = TRUE, fill = TRUE)
      }), use.names = TRUE, fill = TRUE)

      inc_dt = if (nrow(draws)) {
        tmp = copy(draws)
        EPS = 1e-12
        tmp[, sel := as.integer(abs(weight) > EPS)]
        tmp[, .(sel = sum(sel, na.rm = TRUE)), by = .(component, block, feature)]
      } else {
        data.table::data.table(component = character(0), block = character(0), feature = character(0), sel = integer(0))
      }

      sel_freq = merge(sel_grid, inc_dt, by = c("component", "block", "feature"), all.x = TRUE)
      sel_freq[is.na(sel), sel := 0L]
      sel_freq[, eff := n_eff[match(component, comp_lab)]]
      sel_freq[eff <= 0, eff := NA_integer_]
      sel_freq[, `:=`(freq = sel / eff, sel = NULL, eff = NULL)]
      sel_freq$component = as.character(sel_freq$component)
      sel_freq$block = as.character(sel_freq$block)

      # summaries (type-7 percentile intervals + means over aligned accepted draws)
      a = alpha
      summary = if (nrow(draws)) {
        draws[, {
          list(
            boot_mean = mean(weight, na.rm = TRUE),
            boot_sd = if (.N >= 2L) {
              stats::sd(weight, na.rm = TRUE)
            } else {
              NA_real_
            },
            ci_lower = if (.N >= 2L) {
              stats::quantile(weight, probs = a / 2, na.rm = TRUE, names = FALSE)
            } else {
              NA_real_
            },
            ci_upper = if (.N >= 2L) {
              stats::quantile(weight, probs = 1 - a / 2, na.rm = TRUE, names = FALSE)
            } else {
              NA_real_
            },
            replicates_effective = .N
          )
        }, by = .(component, block, feature)]
      } else {
        data.table::data.table(component = character(), block = character(), feature = character(),
          boot_mean = numeric(), boot_sd = numeric(),
          ci_lower = numeric(), ci_upper = numeric(),
          replicates_effective = integer())
      }

      summary$component = as.character(summary$component)
      summary$block = as.character(summary$block)

      list(
        summary = summary,
        select_freq = sel_freq,
        n_eff_by_component = n_eff_by_component,
        alignment_diagnostics = alignment_diagnostics,
        components_below_effective_floor = below_floor,
        n_nonconverged_replicates = as.integer(n_nonconverged_replicates),
        strata_sizes = if (is.null(strata)) NULL else c(table(strata)),
        n_exchangeability_units = units$n_units,
        exchangeability_units_by_stratum = units$units_by_stratum,
        few_exchangeability_units = units$few,
        blocks_order = bn,
        comp_labels = comp_lab
      )
    },

    # ---------------- TRAIN ----------------
    .train_task = function(task) {
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "train"),
        keep.null = TRUE)

      st_env = private$.get_env_state(pv, run_id = self$state$run_id %||% NULL)
      self$state$run_id = st_env$run_id %||% self$state$run_id %||% NULL
      blocks_map = st_env$blocks

      # Always record these flags for predict()
      self$state$stability_only = isTRUE(pv$stability_only)
      self$state$bootstrap = isTRUE(pv$bootstrap)

      if (!isTRUE(pv$bootstrap)) {
        if (isTRUE(pv$stability_only)) {
          stop(
            sprintf("%s: stability_only=TRUE requires bootstrap=TRUE because stable weights/loadings are defined only after the bootstrap selection stage.", self$id),
            call. = FALSE
          )
        }
        self$state$weights_stable = NULL
        self$state$loadings_stable = NULL
        self$state$kept_components = NULL
        self$state$kept_blocks_per_comp = NULL
        return(private$.finalize_scores_only(task, blocks_map))
      }

      if (pv$B < 2L) {
        stop(
          "Bootstrap stability selection requires at least two replicates.",
          call. = FALSE
        )
      }
      if (!is.finite(pv$alpha) || pv$alpha <= 0 || pv$alpha >= 1) {
        stop(
          "Bootstrap stability selection `alpha` must be strictly between 0 and 1.",
          call. = FALSE
        )
      }

      dt_all = task$data()
      lm = private$.lv_column_map(names(dt_all))
      if (lm$K == 0L && !isTRUE(pv$stability_only)) {
        stop("No LV columns found. Ensure po('mbspls') is upstream.")
      }

      lgr$info("Bootstrap selection: B=%d, align='%s', method='%s'",
        as.integer(pv$B), pv$align, pv$selection_method)

      X_blocks_train = st_env$X_train_blocks
      if (is.null(X_blocks_train)) {
        stop(
          "Cannot rebuild training blocks: log_env$mbspls_state$X_train_blocks is missing. Fix: set 'store_train_blocks = TRUE' in po('mbspls', ...) and refit the upstream model.",
          call. = FALSE
        )
      }
      n_env = unique(vapply(X_blocks_train, nrow, integer(1)))
      if (length(n_env) != 1L) {
        stop("Stored X_train_blocks have inconsistent row counts; refit the upstream MB-sPLS model with a valid shared log_env.", call. = FALSE)
      }
      if (n_env != nrow(dt_all)) {
        stop(
          sprintf("Stored X_train_blocks contain %d rows but the current task contains %d rows. Rebuilding from the current task is disabled because it can change the bootstrap basis. Refit the upstream MB-sPLS model on the exact training task used here.", n_env, nrow(dt_all)),
          call. = FALSE
        )
      }

      stratify_block = private$.check_stratify_block(pv$stratify_by_block, X_blocks_train)
      grouping = private$.resolve_bootstrap_groups(pv$bootstrap_groups, task)
      bootstrap_groups = grouping$groups

      rng_streams = if (is.null(pv$seed_bootstrap)) {
        NULL
      } else {
        mb_rng_streams(as.integer(pv$B), pv$seed_bootstrap)
      }

      bt = private$.bootstrap_align_and_summarise(
        X_list = X_blocks_train,
        W_ref = st_env$weights,
        blocks = st_env$blocks,
        ncomp = length(st_env$weights),
        sparsity = st_env$sparsity,
        corr_method = st_env$corr_method %||% "pearson",
        perf_metric = st_env$perf_metric %||% "mac",
        B = as.integer(pv$B),
        alpha = as.numeric(pv$alpha),
        align = pv$align,
        workers = as.integer(pv$workers),
        stratify_block = stratify_block,
        groups = bootstrap_groups,
        rng_streams = rng_streams,
        min_score_cor = as.numeric(pv$min_score_cor %||% 0.10),
        min_effective_fraction = as.numeric(pv$min_effective_fraction %||% 0.5),
        group_source = grouping$source,
        group_column = grouping$column
      )

      sum_df = as.data.frame(bt$summary)
      freq_df = as.data.frame(bt$select_freq)
      n_eff_by_component = bt$n_eff_by_component
      K = length(st_env$weights)

      # Iterate over the full block set so all components have all blocks (zero-padded if filtered)
      bn_full = names(blocks_map)

      # ---- Build BOTH stable variants for env storage
      mag_thr = as.numeric(pv$magnitude_threshold %||% 1e-3)
      built_ci = private$.build_stable_from(
        method = "ci", K = K, bn = bn_full,
        blocks_map = blocks_map, sum_df = sum_df,
        freq_df = freq_df, frequency_threshold = pv$frequency_threshold,
        W_train = st_env$weights,
        weight_source = pv$stable_weight_source,
        magnitude_threshold = mag_thr
      )
      built_freq = private$.build_stable_from(
        method = "frequency", K = K, bn = bn_full,
        blocks_map = blocks_map, sum_df = sum_df,
        freq_df = freq_df, frequency_threshold = pv$frequency_threshold,
        W_train = st_env$weights,
        weight_source = pv$stable_weight_source,
        magnitude_threshold = mag_thr
      )

      # ---- Choose which set governs the graph output (according to selection_method)
      chosen = if (pv$selection_method == "ci") built_ci else built_freq
      W_stable = chosen$W
      kept_blocks_per_comp = chosen$kept
      kept_components = unname(which(lengths(kept_blocks_per_comp) > 0L))
      nonempty_ci = any(lengths(built_ci$kept) > 0L)
      nonempty_freq = any(lengths(built_freq$kept) > 0L)

      selected_not_in_training = NULL
      if (identical(pv$stable_weight_source, "training")) {
        sel = chosen$selection
        zeroed = sel$bootstrap_selected & !is.na(sel$training_weight) & sel$training_weight == 0
        selected_not_in_training = sel[zeroed, c("component", "block", "feature"), with = FALSE]
        if (nrow(selected_not_in_training)) {
          # Expected under the documented intersection semantics of the
          # default weight source; recorded in the state, so only logged.
          lgr$info("%s", sprintf(
            "%s: %d bootstrap-selected feature(s) have zero training weight and are dropped because stable_weight_source='training' restricts the stable support to the training fit: %s. Use stable_weight_source='bootstrap_mean' to keep the bootstrap-selected support.",
            self$id, nrow(selected_not_in_training),
            mb_format_truncated(paste(
              selected_not_in_training$component,
              selected_not_in_training$block,
              selected_not_in_training$feature,
              sep = ":"
            ))
          ))
        }
      }

      # ---- Recompute TRAINING scores and loadings by deflation for every
      # variant that keeps at least one feature.
      X_train = X_blocks_train
      rec_ci = if (nonempty_ci) private$.recompute_scores_deflated(X_train, built_ci$W, names(blocks_map)) else NULL
      rec_freq = if (nonempty_freq) private$.recompute_scores_deflated(X_train, built_freq$W, names(blocks_map)) else NULL

      # ------- Persist the settings and diagnostics shared by both outcomes
      self$state$weights_ci = sum_df
      self$state$weights_selectfreq = freq_df
      self$state$selection = chosen$selection
      self$state$selected_not_in_training = selected_not_in_training
      self$state$n_eff_by_component = n_eff_by_component
      self$state$alignment_diagnostics = bt$alignment_diagnostics
      self$state$components_below_effective_floor = bt$components_below_effective_floor
      self$state$n_nonconverged_replicates = bt$n_nonconverged_replicates
      self$state$rng_streams = rng_streams
      self$state$bootstrap_grouped = !is.null(bootstrap_groups)
      self$state$bootstrap_group_source = grouping$source
      self$state$n_exchangeability_units = bt$n_exchangeability_units
      self$state$exchangeability_units_by_stratum = bt$exchangeability_units_by_stratum
      self$state$few_exchangeability_units = bt$few_exchangeability_units
      self$state$stratify_by_block = stratify_block
      self$state$strata_sizes = bt$strata_sizes
      self$state$alpha = as.numeric(pv$alpha)
      self$state$alignment_method = pv$align
      self$state$component_matching = "exact_assignment"
      self$state$selection_method = pv$selection_method
      self$state$frequency_threshold = pv$frequency_threshold
      self$state$magnitude_threshold = mag_thr
      self$state$stable_weight_source = pv$stable_weight_source
      self$state$min_score_cor = as.numeric(pv$min_score_cor %||% 0.10)
      self$state$min_effective_fraction = as.numeric(pv$min_effective_fraction %||% 0.5)
      self$state$stability_only = isTRUE(pv$stability_only)
      self$state$center = st_env$center

      st_env$weights_ci = sum_df
      st_env$weights_selectfreq = freq_df
      st_env$alignment_method = pv$align
      st_env$selection_method = pv$selection_method
      st_env$frequency_threshold = pv$frequency_threshold
      st_env$magnitude_threshold = mag_thr
      st_env$alpha = as.numeric(pv$alpha)
      st_env$stable_weight_source = pv$stable_weight_source
      st_env$stability_only = isTRUE(pv$stability_only)
      st_env$kept_components = kept_components
      st_env$ncomp_stable = length(kept_components)
      st_env$kept_blocks_per_comp = kept_blocks_per_comp
      st_env$kept_blocks_per_comp_ci = built_ci$kept
      st_env$kept_blocks_per_comp_frequency = built_freq$kept
      st_env$selection_ci = built_ci$selection
      st_env$selection_frequency = built_freq$selection

      # Publish a stable variant only if it keeps at least one feature, so that
      # predict_weights never picks up all-zero weights.
      st_env$weights_stable_ci = if (nonempty_ci) built_ci$W else NULL
      st_env$loadings_stable_ci = if (nonempty_ci) rec_ci$P else NULL
      st_env$weights_stable_frequency = if (nonempty_freq) built_freq$W else NULL
      st_env$loadings_stable_frequency = if (nonempty_freq) rec_freq$P else NULL

      if (!length(kept_components)) {
        n_eff_txt = paste(sprintf(
          "%s=%d/%d", n_eff_by_component$component, n_eff_by_component$n_eff, as.integer(pv$B)
        ), collapse = ", ")
        rule_txt = if (identical(pv$selection_method, "frequency")) {
          sprintf("frequency_threshold=%s", format(pv$frequency_threshold))
        } else {
          sprintf("alpha=%s, magnitude_threshold=%s", format(pv$alpha), format(mag_thr))
        }
        empty_msg = sprintf(
          "%s: no feature passed '%s' stability selection in any component (%s, min_score_cor=%s; accepted replicates: %s).",
          self$id, pv$selection_method, rule_txt, format(pv$min_score_cor), n_eff_txt
        )
        if (!isTRUE(pv$stability_only)) {
          stop(paste(
            empty_msg,
            "The output task would contain no LV features.",
            "Set stability_only = TRUE to keep the upstream LV columns and inspect the bootstrap summaries,",
            "or relax the selection (selection_method = 'frequency', a lower frequency_threshold or",
            "magnitude_threshold, a larger alpha, a lower min_score_cor or less sparsity),",
            "or disable bootstrap selection."
          ), call. = FALSE)
        }
        warning(paste(
          empty_msg,
          "No stable weights are published; the upstream LV columns pass through unchanged."
        ), call. = FALSE)

        self$state$kept_components = integer(0)
        self$state$kept_blocks_per_comp = kept_blocks_per_comp
        self$state$weights_stable = list()
        self$state$loadings_stable = list()

        st_env$weights_stable = NULL
        st_env$loadings_stable = NULL
        T0 = matrix(0, nrow = nrow(X_blocks_train[[1]]), ncol = 0)
        st_env$T_mat_train_stable_all = T0
        st_env$T_mat_train_stable_kept = T0
        st_env$T_mat_train_kept = T0

        log_env_store_state(pv$log_env, st_env, warn_overwrite = FALSE)
        return(task)
      }

      dropped = setdiff(seq_len(K), kept_components)
      if (length(dropped)) {
        lgr$info("%s: component(s) %s keep no stable feature and contribute no LV columns.",
          self$id, paste(sprintf("LC_%02d", dropped), collapse = ", "))
      }

      rec_sel = if (pv$selection_method == "ci") rec_ci else rec_freq
      T_all_dt = rec_sel$T_mat
      P_all = rec_sel$P

      # keep only non-empty block columns for the chosen variant; the column
      # names keep the upstream component index
      keep_cols = character(0)
      for (k in seq_along(W_stable)) {
        kb = kept_blocks_per_comp[[k]]
        if (length(kb)) keep_cols = c(keep_cols, paste0("LV", k, "_", kb))
      }
      T_all = as.matrix(T_all_dt)
      keep_cols = intersect(keep_cols, colnames(T_all))
      T_keep = T_all[, keep_cols, drop = FALSE]

      if (!isTRUE(pv$stability_only)) {
        # ------- Drop upstream LVs and original block features; then append stable LVs
        old_lv = unlist(lm$map, use.names = FALSE)
        orig_feats = intersect(unlist(blocks_map, use.names = FALSE), task$feature_names)
        keep_features = setdiff(task$feature_names, c(old_lv, orig_feats))
        task$select(keep_features)
        task$cbind(data.table::as.data.table(T_keep))
        lgr$info("Bootstrap-select (train): dropped %d original block features and %d upstream LV columns; kept %d stable LV columns.",
          length(orig_feats), length(old_lv), ncol(T_keep))
      } else {
        lgr$info("Stability-only: computed stability selection outputs; leaving task unchanged (raw upstream LVs/features pass through).")
      }

      # ------- Persist to state + env
      self$state$kept_components = kept_components
      self$state$kept_blocks_per_comp = kept_blocks_per_comp
      self$state$weights_stable = W_stable
      self$state$loadings_stable = P_all

      # chosen variant; po("mbspls") uses it for its evaluation payload only
      st_env$weights_stable = W_stable
      st_env$loadings_stable = P_all

      T_all_m = as.matrix(T_all)
      T_keep_m = as.matrix(T_keep)

      # Always store stable-score matrices under stable-specific names
      st_env$T_mat_train_stable_all = T_all_m
      st_env$T_mat_train_stable_kept = T_keep_m

      # Keep backwards compatibility if you want
      st_env$T_mat_train_kept = T_keep_m

      # Only overwrite raw T_mat_train when NOT in stability-only mode
      if (!isTRUE(pv$stability_only)) {
        st_env$T_mat_train = T_all_m
      }

      log_env_store_state(pv$log_env, st_env, warn_overwrite = FALSE)

      if (isTRUE(pv$stability_only)) {
        return(task)
      }
      return(private$.finalize_scores_only(task, blocks_map))
    },

    # ---------------- PREDICT ----------------
    .predict_task = function(task) {
      # Stability-only mode: do not alter the task at predict time
      if (isTRUE(self$state$stability_only)) {
        return(task)
      }

      st = self$state

      # Bootstrap selection disabled: mirror training, where the upstream LV
      # columns were kept. States without the flag carry no stable weights.
      if (isFALSE(st$bootstrap) || (is.null(st$bootstrap) && is.null(st$weights_stable))) {
        env = self$param_set$values$log_env
        st_env = .mbspls_state_from_env(env, run_id = st$run_id %||% NULL, require_train_blocks = FALSE, where = "log_env")
        return(private$.finalize_scores_only(task, st_env$blocks))
      }

      # Legacy states without stable weights: drop upstream LVs and block
      # features, then return
      if (is.null(st$weights_stable) || !length(st$weights_stable)) {
        dt_all = task$data()
        lm = private$.lv_column_map(names(dt_all))
        env = self$param_set$values$log_env
        st_env = .mbspls_state_from_env(env, run_id = st$run_id %||% NULL, require_train_blocks = FALSE, where = "log_env")
        blocks_map = st_env$blocks
        old_lv = if (lm$K) unlist(lm$map, use.names = FALSE) else character(0)
        orig_feats = intersect(unlist(blocks_map, use.names = FALSE), task$feature_names)
        keep_features = setdiff(task$feature_names, c(old_lv, orig_feats))
        task$select(keep_features)
        return(private$.finalize_scores_only(task, blocks_map))
      }

      dt = task$data()
      env = self$param_set$values$log_env
      st_env = .mbspls_state_from_env(env, run_id = st$run_id %||% NULL, require_train_blocks = FALSE, where = "log_env")
      blocks_map = st_env$blocks
      all_blocks = names(blocks_map)
      # States written before centring was introduced carry no means (zero).
      center = st$center %||% st_env$center

      # Build X_test by blocks (strictly require trained features) and centre
      # with the training means of the upstream fit
      X_test = lapply(names(blocks_map), function(bn) {
        cols = blocks_map[[bn]]
        mb_assert_columns_present(
          names(dt),
          cols,
          context = sprintf("PipeOpMBsPLSBootstrapSelect prediction block '%s'", bn),
          hint = "Ensure the prediction task still contains the raw block features used during training."
        )
        m = as.matrix(dt[, ..cols])
        storage.mode(m) = "double"
        mu = center[[bn]]
        if (!is.null(mu)) {
          mu = as.numeric(mb_align_named_numeric(
            mu,
            cols = cols,
            context = sprintf("center[['%s']]", bn)
          ))
          m = m - rep(mu, each = nrow(m))
        }
        m
      })
      names(X_test) = names(blocks_map)

      # Use stable weights + stable loadings to deflate
      W_stable = st$weights_stable
      P_train = st$loadings_stable
      if (is.null(P_train) || !length(P_train)) {
        stop(
          "Prediction with stable bootstrap-selected components requires stored stable loadings. Refit the bootstrap selection stage so that 'loadings_stable' is available.",
          call. = FALSE
        )
      }

      K = length(W_stable)
      X_cur = lapply(X_test, identity)
      tabs = vector("list", K)

      for (k in seq_len(K)) {
        Tk = matrix(0, nrow(X_cur[[1]]), length(all_blocks))
        for (bi in seq_along(all_blocks)) {
          b = all_blocks[bi]
          cols = colnames(X_cur[[b]])
          wv = as.numeric(mb_align_named_numeric(
            W_stable[[k]][[b]],
            cols = cols,
            context = sprintf("weights_stable[[%d]][['%s']]", k, b)
          ))
          storage.mode(wv) = "double"
          Tk[, bi] = X_cur[[b]] %*% wv
        }
        colnames(Tk) = paste0("LV", k, "_", all_blocks)
        tabs[[k]] = data.table::as.data.table(Tk)
        if (k < K) {
          for (bi in seq_along(all_blocks)) {
            b = all_blocks[bi]
            cols = colnames(X_cur[[b]])
            pb = as.numeric(mb_align_named_numeric(
              P_train[[k]][[b]],
              cols = cols,
              context = sprintf("loadings_stable[[%d]][['%s']]", k, b)
            ))
            X_cur[[b]] = X_cur[[b]] - Tk[, bi, drop = FALSE] %*% t(as.matrix(pb))
          }
        }
      }
      T_pred_all_dt = do.call(cbind, tabs)

      # keep only non-empty block columns from kept blocks
      keep_cols = character(0)
      for (newk in seq_along(st$kept_blocks_per_comp)) {
        kb = st$kept_blocks_per_comp[[newk]]
        if (length(kb) > 0L) {
          keep_cols = c(keep_cols, paste0("LV", newk, "_", kb))
        }
      }

      T_pred_all = as.matrix(T_pred_all_dt)
      T_pred_keep = if (length(keep_cols)) {
        keep_cols = intersect(keep_cols, colnames(T_pred_all))
        T_pred_all[, keep_cols, drop = FALSE]
      } else {
        matrix(0, nrow = nrow(T_pred_all), ncol = 0)
      }

      # ------- Drop upstream LVs and original block features; then append stable LVs
      dt_all = task$data()
      lm = private$.lv_column_map(names(dt_all))
      old_lv = if (lm$K) unlist(lm$map, use.names = FALSE) else character(0)
      orig_feats = intersect(unlist(blocks_map, use.names = FALSE), task$feature_names)
      keep_features = setdiff(task$feature_names, c(old_lv, orig_feats))
      task$select(keep_features)
      if (ncol(T_pred_keep)) task$cbind(data.table::as.data.table(T_pred_keep))
      lgr$info("Bootstrap-select (predict): dropped %d original block features and %d upstream LV columns; kept %d stable LV columns.",
        length(orig_feats), length(old_lv), ncol(T_pred_keep))

      return(private$.finalize_scores_only(task, blocks_map))
    }
  )
)

# Count the exchangeability units of a bootstrap design (rows when `design` is
# NULL) per stratum. Stops when no stratum holds two or more units, because
# every replicate would then equal the training data. Flags designs with fewer
# than `min_units` units or fewer distinct bootstrap samples than replicates;
# G units give choose(2G - 1, G) distinct samples (multisets), multiplied over
# strata.
.mb_bootstrap_unit_check = function(
  design, n_rows, B,
  group_source = c("rows", "explicit", "task_group_role"),
  group_column = NULL, stratify_block = NULL, min_units = 10L
) {
  group_source = match.arg(group_source)
  stratified = !is.null(design) && !is.null(design$group_strata)
  units_by_stratum = if (is.null(design)) {
    n_rows
  } else if (!stratified) {
    length(design$groups)
  } else {
    c(table(factor(design$group_strata, levels = unique(design$group_strata))))
  }
  n_units = as.integer(sum(units_by_stratum))
  unit_label = switch(group_source,
    rows = "training rows",
    explicit = "groups of the explicit `bootstrap_groups`",
    task_group_role = sprintf("groups of the task's group role '%s'", group_column %||% "group")
  )

  if (all(units_by_stratum < 2L)) {
    what = if (stratified) {
      sprintf(
        "every stratum of `stratify_by_block` = '%s' holds a single exchangeability unit (%s)",
        stratify_block, unit_label
      )
    } else {
      sprintf("there is only %d exchangeability unit (%s)", n_units, unit_label)
    }
    remedy = switch(group_source,
      rows = "At least two training rows are required.",
      explicit = "Supply at least two groups in `bootstrap_groups`.",
      task_group_role = paste(
        "Remove the group role from the task before this operator (rows, or an explicit",
        "`bootstrap_groups`, are then resampled) or disable bootstrap selection."
      )
    )
    stop(sprintf(
      "Bootstrap selection cannot resample: %s, so every replicate equals the training data and every interval has zero width. %s%s",
      what, remedy, if (stratified) " Alternatively drop `stratify_by_block`." else ""
    ), call. = FALSE)
  }

  n_distinct = exp(sum(lchoose(2 * units_by_stratum - 1, units_by_stratum)))
  few = n_units < min_units || n_distinct < B
  if (few) {
    warning(sprintf(
      paste(
        "bootstrap_select: only %d exchangeability units (%s)%s can be resampled, i.e. at most %s distinct bootstrap samples for B = %d replicates.",
        "Percentile intervals and selection frequencies from so few units are unreliable and can have zero width."
      ),
      n_units, unit_label,
      if (stratified) {
        sprintf(" (per stratum: %s)", paste(names(units_by_stratum), units_by_stratum, sep = " = ", collapse = ", "))
      } else {
        ""
      },
      format(round(n_distinct), big.mark = ",", scientific = FALSE, trim = TRUE), B
    ), call. = FALSE)
  }
  list(
    n_units = n_units,
    units_by_stratum = if (stratified) units_by_stratum else NULL,
    few = few
  )
}

# Exact maximum-similarity assignment of replicate components (rows) to
# reference components (columns) of a square matrix `S`, via the Hungarian
# algorithm on `max(S) - S`. Returns `p` with row `i` assigned to column
# `p[i]`. Pure R and deterministic, so matching does not depend on optional
# packages or the platform.
.mb_assignment_max = function(S) {
  S = as.matrix(S)
  n = nrow(S)
  if (!n) {
    return(integer(0))
  }
  if (ncol(S) != n) {
    stop("Assignment requires a square similarity matrix.", call. = FALSE)
  }
  S[!is.finite(S)] = 0
  cost = max(S) - S

  # Potentials u (rows) and v (columns); `match_col[j + 1]` is the row matched
  # to column j, with column 0 a virtual start column.
  u = numeric(n + 1L)
  v = numeric(n + 1L)
  match_col = integer(n + 1L)
  way = integer(n + 1L)
  for (i in seq_len(n)) {
    match_col[1L] = i
    j0 = 0L
    minv = rep(Inf, n + 1L)
    used = rep(FALSE, n + 1L)
    repeat {
      used[j0 + 1L] = TRUE
      i0 = match_col[j0 + 1L]
      delta = Inf
      j1 = 0L
      for (j in seq_len(n)) {
        if (!used[j + 1L]) {
          cur = cost[i0, j] - u[i0 + 1L] - v[j + 1L]
          if (cur < minv[j + 1L]) {
            minv[j + 1L] = cur
            way[j + 1L] = j0
          }
          if (minv[j + 1L] < delta) {
            delta = minv[j + 1L]
            j1 = j
          }
        }
      }
      for (j in 0:n) {
        if (used[j + 1L]) {
          u[match_col[j + 1L] + 1L] = u[match_col[j + 1L] + 1L] + delta
          v[j + 1L] = v[j + 1L] - delta
        } else {
          minv[j + 1L] = minv[j + 1L] - delta
        }
      }
      j0 = j1
      if (match_col[j0 + 1L] == 0L) break
    }
    repeat {
      j1 = way[j0 + 1L]
      match_col[j0 + 1L] = match_col[j1 + 1L]
      j0 = j1
      if (j0 == 0L) break
    }
  }

  p = integer(n)
  p[match_col[-1L]] = seq_len(n)
  p
}

# Recover the levels of a single dummy-coded factor from a (possibly centred
# or scaled) block. Every column must take exactly two values; the larger one
# marks an active indicator. Full one-hot coding has one active indicator per
# row; with treatment coding, rows without an active indicator form the
# reference stratum.
.mb_strata_from_dummy_block = function(X, block) {
  X = as.matrix(X)
  if (!ncol(X) || !nrow(X)) {
    stop(sprintf("Stratification block '%s' is empty.", block), call. = FALSE)
  }
  labels = colnames(X) %||% paste0(block, "_", seq_len(ncol(X)))
  active = matrix(FALSE, nrow(X), ncol(X))
  for (j in seq_len(ncol(X))) {
    x = X[, j]
    if (anyNA(x) || any(!is.finite(x))) {
      stop(sprintf(
        "Stratification block '%s' contains missing or non-finite values in column '%s'.",
        block, labels[j]
      ), call. = FALSE)
    }
    lo = min(x)
    hi = max(x)
    tol = sqrt(.Machine$double.eps) * max(1, abs(lo), abs(hi))
    is_hi = abs(x - hi) <= tol
    if (hi - lo <= tol || !all(is_hi | abs(x - lo) <= tol)) {
      stop(sprintf(
        paste(
          "`stratify_by_block` block '%s' is not a dummy-coded factor: column '%s' does not take exactly two distinct values.",
          "Use a block that holds the one-hot or treatment dummies of one factor."
        ),
        block, labels[j]
      ), call. = FALSE)
    }
    active[, j] = is_hi
  }
  n_active = rowSums(active)
  if (any(n_active > 1L)) {
    stop(sprintf(
      paste(
        "`stratify_by_block` block '%s' has rows with more than one active indicator,",
        "so it does not encode a single factor. Use a block that holds the one-hot or treatment dummies of one factor."
      ),
      block
    ), call. = FALSE)
  }
  reference = make.unique(c(labels, "(reference)"))[length(labels) + 1L]
  strata = rep(reference, nrow(X))
  has_level = n_active == 1L
  strata[has_level] = labels[max.col(active[has_level, , drop = FALSE], ties.method = "first")]
  strata = factor(strata, levels = unique(strata))
  if (nlevels(strata) < 2L) {
    stop(sprintf(
      "`stratify_by_block` block '%s' defines fewer than two strata; stratification needs at least two levels.",
      block
    ), call. = FALSE)
  }
  strata
}
