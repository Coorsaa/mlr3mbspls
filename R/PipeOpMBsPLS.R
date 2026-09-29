#' Multi-Block Sparse Partial Least Squares (MB-sPLS) Transformer
#'
#' @title Extract up to \code{ncomp} orthogonal latent components from multiple data blocks
#'
#' @description
#' \strong{PipeOpMBsPLS} fits \emph{sequential} MB-sPLS models and appends one
#' latent variable (LV) per block and component to the task's backend.
#' After each component the corresponding rank-1 structure is removed
#' (block-wise score deflation in the sense of Westerhuis et al., 2001),
#' ensuring that successive LVs are orthogonal within every block.
#'
#' The association criterion reported per component is controlled by
#' \code{performance_metric}:
#' \itemize{
#'   \item \code{"mac"}: mean absolute correlation (average of \eqn{|r|}
#'         across all block-score pairs) per component;
#'   \item \code{"frobenius"}: Euclidean norm of the distinct off-diagonal
#'         block-score correlations \eqn{\sqrt{\sum_{i<j} r_{ij}^2}}.
#' }
#' We refer to this scalar as the \emph{latent correlation}. Weights are updated
#' block by block (Gauss-Seidel sweeps) by constrained PMD steps against
#' standardized scores from the other blocks, starting from a deterministic
#' cross-covariance start, so fits do not depend on the random seed. The
#' updates do not directly optimize the reported correlation criterion, and the
#' solution is a local optimum of a non-convex problem: with strong sparsity,
#' pure noise or many more features than rows, other local optima can reach a
#' higher objective. \code{correlation_method} and \code{performance_metric}
#' change evaluation, tuning and testing, not the weight update or the stopping
#' rule. A component is converged when the largest change of a block weight
#' vector between two sweeps is below \code{1e-4}; after at most 600 sweeps the
#' last iterate is returned and a warning names the components that did not
#' converge.
#'
#' \strong{Centring.} Every retained block column is centred by its training
#' mean before fitting; the means are stored in the state and subtracted from
#' prediction data. The fitted weights do not depend on centring, but the
#' deflation loadings, explained variances and scores do.
#'
#' During prediction, the same criterion (MAC/Frobenius) and explained variances
#' are computed on test data and, if \code{log_env} is provided, a compact
#' payload is written to \code{log_env$last}. Optionally, prediction-side
#' diagnostics can be requested (permutation or descriptive bootstrap; see
#' Parameters).
#'
#' The operator is a pure transformer: it performs no internal resampling,
#' tuning or preprocessing beyond centring. Hyper-parameters such as the L1
#' sparsity levels \eqn{c_\mathrm{block}} are tuned externally (e.g., with
#' \pkg{mlr3tuning}).
#'
#' @section State (after training):
#' \describe{
#'   \item{\code{blocks}}{Named list mapping block names to the resolved feature
#'         columns used for fitting.}
#'   \item{\code{center}}{Named list (by block) of training column means,
#'         each a numeric vector named by column; subtracted from prediction
#'         data.}
#'   \item{\code{weights}}{List of length \code{ncomp}; block-specific weight vectors \eqn{w_b^{(k)}}.}
#'   \item{\code{loadings}}{List of block loadings \eqn{p_b^{(k)}} used for deflation.}
#'   \item{\code{ncomp}}{Number of components retained.}
#'   \item{\code{obj_vec}}{Objective values (MAC/Frobenius) per component (training).}
#'   \item{\code{latent_cor_train}}{Objective value of the last retained component (training).}
#'   \item{\code{ev_block}}{Training explained variance per block, a matrix
#'         with rows \code{LC_xx} and one column per block.}
#'   \item{\code{ev_comp}}{Training explained variance per component, a named
#'         numeric vector (\code{LC_xx}). It is SS-weighted across blocks,
#'         \eqn{\sum_b SS_{exp,b} / \sum_b SS_{tot,b}}, not the row sum of
#'         \code{ev_block}.}
#'   \item{\code{converged}, \code{iterations}}{Per component: whether the
#'         solver converged and the number of sweeps it used.}
#'   \item{\code{p_values}}{Conditional component-wise permutation p-values if
#'         enabled during training (\code{NA} otherwise). They assume the
#'         supplied preprocessed data and hyperparameters are fixed and are not
#'         full-pipeline inference.}
#'   \item{\code{p_value_scope}}{Description of that scope when the diagnostic
#'         ran, otherwise \code{NULL}.}
#'   \item{\code{performance_metric}, \code{correlation_method}}{Settings used
#'         for the objective.}
#'   \item{\code{c_matrix}}{If provided, the block-by-component sparsity matrix
#'         actually used (retained blocks; entries capped at
#'         \eqn{\sqrt{p_b}} of the retained columns).}
#'   \item{\code{T_mat}}{Training score matrix (per-component deflation applied);
#'         columns ordered \code{LV1_<block1>, ..., LV1_<blockB>, LV2_<block1>, ...}.}
#'   \item{\code{run_id}}{Identifier of the training snapshot in \code{log_env},
#'         if a \code{log_env} is used.}
#' }
#'
#' @section Emitted features:
#' The \code{LVk_<block>} columns are always computed from the training fit:
#' raw weights, training centring and training deflation loadings, both at
#' train and at predict time, so a downstream learner sees one feature
#' definition. Stability-selected weights published by a
#' [PipeOpMBsPLSBootstrapSelect] only affect the prediction-side payload
#' (see \code{predict_weights}); stable LV features reach a learner only through
#' a PipeOpMBsPLSBootstrapSelect that is not in stability-only mode, which
#' replaces the upstream LV columns consistently at train and predict.
#'
#' @section Prediction-side logging (\code{log_env$last}):
#' A list containing:
#' \itemize{
#'   \item \code{mac_comp}: numeric vector (length \code{ncomp}) with test MAC/Frobenius per component,
#'   \item \code{ev_block}: matrix \code{(ncomp x n_blocks)} with test per-block explained variances,
#'   \item \code{ev_comp}: numeric vector \code{(ncomp)} with test per-component
#'         EV, SS-weighted across blocks (see [compute_test_ev()]),
#'   \item \code{ev_block_cum}, \code{ev_comp_cum}: cumulative counterparts,
#'   \item \code{T_mat}: test scores \code{(n_test x (ncomp * n_blocks))} of the
#'         evaluated weights, with the same column order as training,
#'   \item \code{weights}, \code{loadings}: the evaluated weights and loadings,
#'         aligned to the trained features,
#'   \item \code{weights_source}: \code{"raw"}, \code{"stable_ci"} or
#'         \code{"stable_frequency"}, the source of \code{weights},
#'   \item \code{emitted_weights_source}: always \code{"raw"}; the emitted LV
#'         features use the training weights,
#'   \item \code{blocks}: character vector with block names,
#'   \item \code{perf_metric}, \code{corr_method}: evaluation settings,
#'   \item \code{run_id}: identifier of the matching training snapshot,
#'   \item \code{time}: POSIXct timestamp,
#'   \item \code{val_test_p}: (if \code{val_test = "permutation"}) per-component
#'         conditional permutation p-values,
#'   \item \code{val_test_stat}: (if available) observed test statistic per component,
#'   \item \code{val_test_status}: (if \code{val_test != "none"}) per component,
#'         \code{"computed"} or the reason the diagnostic was not computed,
#'   \item \code{val_bootstrap}: (if \code{val_test = "bootstrap"}) data.table with
#'         the observed statistic, bias, standard error, interval, confidence
#'         level, and effective replicate count. Its p-value fields are `NA`
#'         because an ordinary bootstrap distribution is not a null distribution.
#' }
#'
#' @section Parameters:
#' Hyperparameters are defined in the object's \code{param_set} and can be set
#' via \code{param_vals}. Block membership (\code{blocks}) is a constructor
#' argument and stored in the object.
#' * `blocks` (`uty`, default: the constructor `blocks`; tag `"train"`): named
#'   list mapping block names to declared feature names. Declared names absent
#'   from the data expand to encoded columns `<name>.<suffix>` as described in
#'   [mb_resolve_block_columns()]; the resolved blocks must be disjoint.
#'   Non-numeric columns and columns without finite positive variance in the
#'   training data are dropped, and blocks without usable columns are dropped.
#' * `ncomp` (`int`, default `1`; tag `"train"`): number of latent components to
#'   extract. Replaced by `ncol(c_matrix)` when `c_matrix` is set.
#' * `c_<block>` (one `dbl` per block, lower `1`, upper `sqrt(p_b)` of the
#'   declared names, default `max(1, sqrt(p_b) / 3)`; tags `c("train", "tune")`):
#'   L1 budget of the unit-L2 weight vector of that block. Budgets above
#'   `sqrt(p)` of the retained columns are nonbinding.
#' * `c_matrix` (`uty`, default `NULL`; tags `c("train", "tune")`): matrix of L1
#'   budgets (rows = blocks, columns = components) that overrides `c_<block>`
#'   and `ncomp`. Named rows must be unique and non-empty and match either all
#'   declared or all retained blocks; an unnamed matrix is matched by position
#'   to the declared blocks. Every entry must lie in `[1, sqrt(p_b)]`, where
#'   `p_b` is the structural width of the block (its resolved numeric columns
#'   before constant columns are removed). Entries above `sqrt(p)` of the
#'   retained columns are nonbinding; they are capped there and logged.
#' * `efficient` (`lgl`, default `FALSE`; tag `"train"`): reserved; no effect.
#' * `correlation_method` (`fct`, `"pearson"` (default) or `"spearman"`; tags
#'   `c("train", "predict")`): correlation of block scores used for the
#'   objective, the prediction-side payload and the diagnostics.
#' * `performance_metric` (`fct`, `"mac"` (default) or `"frobenius"`; tags
#'   `c("train", "predict")`): latent-correlation summary reported per component
#'   and used by tuning, measures and diagnostics.
#' * `permutation_test` (`lgl`, default `FALSE`; tag `"train"`): run a
#'   conditional component-wise permutation diagnostic during training and stop
#'   extraction when its empirical p-value exceeds `perm_alpha` (LV1 is always
#'   retained). It conditions on the supplied preprocessed data and
#'   hyperparameters and does not replace a design-valid full-pipeline
#'   permutation analysis (see [mbspls_permutation_test()]).
#' * `n_perm` (`int`, default `100`; tag `"train"`): permutations of the
#'   training diagnostic.
#' * `perm_alpha` (`dbl` in `[0, 1]`, default `0.05`; tag `"train"`): cutoff for
#'   the conditional training diagnostic.
#' * `store_train_blocks` (`lgl`, default `FALSE`; tag `"train"`): if `TRUE` and
#'   `log_env` is set, store the centred training block matrices
#'   (`X_train_blocks`) and the sparsity settings in the training snapshot.
#'   Required by [PipeOpMBsPLSBootstrapSelect].
#' * `predict_weights` (`fct`, one of `"auto"` (default), `"raw"`,
#'   `"stable_ci"`, `"stable_frequency"`; tag `"predict"`): weights evaluated in
#'   the prediction-side payload (`log_env$last`) and by `val_test`. The emitted
#'   LV features always use the training weights (see Emitted features).
#'   `"auto"` uses the stable weights published by a
#'   [PipeOpMBsPLSBootstrapSelect] for the same run on the same `log_env`, and
#'   raw weights if there are none or if that stage ran with
#'   `stability_only = TRUE`. Explicit `"stable_ci"`/`"stable_frequency"`
#'   requests error if the requested weights are unavailable, or if the
#'   bootstrap stage ran in stability-only mode, where stable weights define no
#'   feature a downstream learner sees.
#' * `val_test` (`fct`, `"none"` (default), `"permutation"` or `"bootstrap"`;
#'   tag `"predict"`): prediction-side diagnostic with the trained weights held
#'   fixed: a conditional held-out permutation diagnostic of the latent
#'   correlation, or descriptive bootstrap uncertainty (no p-value). A component
#'   with fewer than two blocks that have non-zero weights and non-degenerate
#'   held-out scores is not tested: its results are `NA` and `val_test_status`
#'   records the reason. A design-level hypothesis test must rerun the complete
#'   preprocessing, tuning and fitting pipeline within each valid permutation.
#' * `val_test_alpha` (`dbl` in `[0, 1]`, default `0.05`; tag `"predict"`): alpha
#'   of the descriptive bootstrap confidence interval. Retained for permutation
#'   calls for API compatibility; sampled permutation p-values always use all
#'   replicates.
#' * `val_test_n` (`int`, default `1000`; tag `"predict"`): permutations or
#'   bootstrap replicates of the prediction-side diagnostic.
#' * `val_test_permute_all` (`lgl`, default `TRUE`; tag `"predict"`): permute all
#'   blocks; for two blocks, `FALSE` permutes block 2 only.
#' * `seed_validation` (`int` or `NULL`, default `NULL`; tag `"predict"`):
#'   optional seed for the prediction-side diagnostics. One L'Ecuyer-CMRG stream
#'   is assigned per component and the caller's RNG state is restored.
#' * `log_env` (`environment` or `NULL`, default `NULL`; tags
#'   `c("train", "predict")`): if set, training stores a snapshot via
#'   `log_env$mbspls_state` / `log_env$mbspls_states` and prediction writes the
#'   payload to `log_env$last` and `log_env$mbspls_last[[run_id]]`.
#' * `append` (`lgl`, default `FALSE`; tags `c("train", "predict")`): if `TRUE`,
#'   keep the original features and append the LV columns; otherwise output only
#'   the LV columns.
#' * `seed_train` (`int` or `NULL`, default `NULL`; tag `"train"`): optional
#'   seed for random draws during training, i.e. the permutations of
#'   `permutation_test`. The fit itself is deterministic and does not depend on
#'   the seed.
#'
#' @section Construction:
#' `PipeOpMBsPLS$new(id = "mbspls", blocks, param_vals = list())`
#'
#' @return
#' A \code{PipeOpMBsPLS} that outputs either only \code{LVk_<block>} columns
#' (default) or the original features plus appended LV columns (if \code{append=TRUE}).
#'
#' @family PipeOps
#' @keywords internal
#' @importFrom R6 R6Class
#' @import data.table lgr
#' @importFrom checkmate assert_list
#' @importFrom paradox ps p_int p_lgl p_uty p_dbl p_fct
#' @importFrom mlr3pipelines PipeOpTaskPreproc
#' @export
PipeOpMBsPLS = R6::R6Class(
  "PipeOpMBsPLS",
  inherit = mlr3pipelines::PipeOpTaskPreproc,

  public = list(
    #' @field blocks Named list mapping block names to character vectors of feature names.
    blocks = NULL,

    #' @description Initialize the PipeOpMBsPLS.
    #' @param id character(1). Identifier of the resulting object.
    #' @param blocks named list. Map of block names to feature column names.
    #' @param param_vals named list. List of hyperparameter settings.
    initialize = function(id = "mbspls", blocks, param_vals = list()) {

      blocks = mb_normalize_blocks(blocks, .var.name = "blocks")

      base_params = list(
        blocks = p_uty(tags = "train", default = blocks),
        ncomp = p_int(lower = 1L, default = 1L, tags = "train"),
        c_matrix = p_uty(tags = c("train", "tune"), default = NULL),
        efficient = p_lgl(default = FALSE, tags = "train"),
        correlation_method = p_fct(c("pearson", "spearman"), default = "pearson", tags = c("train", "predict")),
        performance_metric = p_fct(c("mac", "frobenius"), default = "mac", tags = c("train", "predict")),
        permutation_test = p_lgl(default = FALSE, tags = "train"),
        n_perm = p_int(lower = 1L, default = 100L, tags = "train"),
        perm_alpha = p_dbl(lower = 0, upper = 1, default = 0.05, tags = "train"),
        store_train_blocks = p_lgl(default = FALSE, tags = "train"),
        predict_weights = p_fct(c("auto", "raw", "stable_ci", "stable_frequency"), default = "auto", tags = "predict"),
        val_test = p_fct(c("none", "permutation", "bootstrap"), default = "none", tags = "predict"),
        val_test_alpha = p_dbl(lower = 0, upper = 1, default = 0.05, tags = "predict"),
        val_test_n = p_int(lower = 1L, default = 1000L, tags = "predict"),
        val_test_permute_all = p_lgl(default = TRUE, tags = "predict"),
        seed_validation = p_uty(tags = "predict", default = NULL),
        log_env = p_uty(tags = c("train", "predict"), default = NULL),
        append = p_lgl(default = FALSE, tags = c("train", "predict")),
        seed_train = p_uty(tags = "train", default = NULL)
      )

      for (bn in names(blocks)) {
        p = length(blocks[[bn]])
        base_params[[paste0("c_", bn)]] = p_dbl(
          lower   = 1,
          upper   = sqrt(p),
          default = max(1, sqrt(p) / 3),
          tags    = c("train", "tune")
        )
      }

      if (!is.null(param_vals$c_matrix)) {
        cm = param_vals$c_matrix
        if (!is.matrix(cm) || !is.numeric(cm) || !ncol(cm) ||
          anyNA(cm) || any(!is.finite(cm))) {
          stop(
            "`c_matrix` must be a finite numeric matrix with at least one column.",
            call. = FALSE
          )
        }
        if (nrow(cm) != length(blocks)) {
          stop(sprintf(
            "c_matrix must have %d rows (blocks); got %d",
            length(blocks),
            nrow(cm)
          ), call. = FALSE)
        }
        if (!is.null(rownames(cm))) {
          if (anyNA(rownames(cm)) || any(!nzchar(rownames(cm))) ||
            anyDuplicated(rownames(cm))) {
            stop("c_matrix row names must be unique and non-empty.",
              call. = FALSE)
          }
          missing_rows = setdiff(names(blocks), rownames(cm))
          extra_rows = setdiff(rownames(cm), names(blocks))
          if (length(missing_rows) || length(extra_rows)) {
            stop(sprintf(
              "c_matrix rows must match all blocks exactly. Missing: %s; unexpected: %s",
              paste(missing_rows, collapse = ", "),
              paste(extra_rows, collapse = ", ")
            ), call. = FALSE)
          }
          cm = cm[names(blocks), , drop = FALSE]
        }
        if (!is.null(colnames(cm)) &&
          (anyNA(colnames(cm)) || any(!nzchar(colnames(cm))) ||
            anyDuplicated(colnames(cm)))) {
          stop("c_matrix column names must be unique and non-empty.",
            call. = FALSE)
        }
        param_vals$c_matrix = cm
        param_vals$ncomp = ncol(cm)
      }

      super$initialize(id = id, param_set = do.call(ps, base_params), param_vals = param_vals)
      self$packages = "mlr3mbspls"
      self$blocks = blocks
    }
  ),

  private = list(
    # ------------------------------- train -----------------------------------
    .train_dt = function(dt, levels, target = NULL) {
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "train"),
        keep.null = TRUE)

      use_frob = (pv$performance_metric == "frobenius")
      max_iter = 600L

      resolved = .mb_training_blocks(dt, pv$blocks)
      blocks = resolved$blocks
      if (!length(blocks)) stop("No block contains at least one numeric, non-constant feature.")
      n_block = length(blocks)

      X_raw = lapply(names(blocks), function(name) {
        cols = blocks[[name]]
        m = .mb_numeric_matrix(
          as.matrix(dt[, ..cols]),
          sprintf("training block '%s'", name)
        )
        storage.mode(m) = "double"
        m
      }) |>
        stats::setNames(names(blocks))
      # The solver assumes column-centred blocks for deflation, EV and scores.
      center = .mb_block_means(X_raw)
      X_list = .mb_center_blocks(X_raw, center)

      if (!is.null(pv$c_matrix)) {
        cm_input = .mb_align_c_matrix(
          pv$c_matrix,
          declared = names(pv$blocks),
          retained = names(blocks)
        )
        cm = .mb_prepare_c_matrix(
          blocks = X_list,
          c_matrix = cm_input,
          ncomp = ncol(cm_input),
          ncomp_missing = FALSE,
          upper_p = resolved$p_struct
        )
        cm = .mb_finalize_c_matrix(cm, X_list, self$id)
        pv$ncomp = ncol(cm)
        c_matrix = cm
        c_vec = NULL
        lgr$info("Fitting MB-sPLS: %d blocks, %d components (c-matrix).", n_block, pv$ncomp)
      } else {
        c_vec = vapply(names(blocks), \(bn) pv[[paste0("c_", bn)]], numeric(1))
        c_matrix = NULL
        lgr$info("Fitting MB-sPLS: %d blocks, %d components; c = %s",
          n_block, pv$ncomp, paste(c_vec, collapse = ", "))
      }
      .mb_assert_component_rank(X_list, pv$ncomp, "PipeOpMBsPLS")

      # The one-component solver is deterministic (declared rng = false) and
      # canonicalises its global sign, so the *fit* needs no seed. Permutation
      # testing does draw from the RNG, and its p-values therefore vary between
      # runs unless a seed is set.
      if (is.null(pv$seed_train) && isTRUE(pv$permutation_test)) {
        lgr::lgr$warn(paste(
          "[%s] permutation_test = TRUE with seed_train = NULL: the permutation",
          "p-values (and hence which components are retained) are not",
          "reproducible across runs. Set seed_train for reproducible results."
        ), self$id)
      }
      fit = with_seed_local(pv$seed_train, function() {
        if (is.null(c_matrix)) {
          cpp_mbspls_multi_lv(
            X_blocks      = X_list,
            c_constraints = c_vec,
            K             = pv$ncomp,
            max_iter      = max_iter,
            tol           = 1e-4,
            spearman      = (pv$correlation_method == "spearman"),
            do_perm       = isTRUE(pv$permutation_test),
            n_perm        = pv$n_perm,
            alpha         = pv$perm_alpha,
            frobenius     = use_frob
          )
        } else {
          cpp_mbspls_multi_lv_cmatrix(
            X_blocks  = X_list,
            c_matrix  = c_matrix,
            max_iter  = max_iter,
            tol       = 1e-4,
            spearman  = (pv$correlation_method == "spearman"),
            do_perm   = isTRUE(pv$permutation_test),
            n_perm    = pv$n_perm,
            alpha     = pv$perm_alpha,
            frobenius = use_frob
          )
        }
      })

      self$state$c_matrix = c_matrix

      if (length(fit$W) == 0) stop("No components extracted - check sparsity settings.")
      lgr$info("C++ returned %d component(s)", length(fit$W))
      lgr$info("Objectives per component: %s", paste(round(fit$objective, 4), collapse = ", "))
      if (isTRUE(pv$permutation_test) && !is.null(fit$p_values)) {
        lgr$info("Permutation p-values: %s", paste(signif(fit$p_values, 3), collapse = ", "))
      }

      W_all = fit$W
      P_all = fit$P

      n_kept = length(W_all)
      B = length(blocks)
      block_names = names(blocks)
      comp_names = sprintf("LC_%02d", seq_len(n_kept))

      obj = stats::setNames(as.numeric(fit$objective), comp_names)
      pvals = stats::setNames(as.numeric(fit$p_values %||% rep(NA_real_, n_kept)), comp_names)
      ev_blk = matrix(
        as.numeric(fit$ev_block),
        nrow = n_kept,
        ncol = B,
        dimnames = list(comp_names, block_names)
      )
      ev_cmp = stats::setNames(as.numeric(fit$ev_comp), comp_names)
      converged = stats::setNames(
        as.logical(fit$converged %||% rep(NA, n_kept)),
        comp_names
      )
      iterations = stats::setNames(
        as.integer(fit$iterations %||% rep(NA_integer_, n_kept)),
        comp_names
      )
      .mb_warn_nonconverged(converged, sprintf("[%s] MB-sPLS", self$id), max_iter)

      pad_and_name = function(x, feat_names) {
        if (length(x) == 0L) {
          x = numeric(length(feat_names))
        }
        if (length(x) != length(feat_names)) {
          stop(sprintf("Internal size mismatch: expected %d, got %d", length(feat_names), length(x)))
        }
        stats::setNames(as.numeric(x), feat_names)
      }
      for (k in seq_len(n_kept)) {
        for (bn in block_names) {
          feats = blocks[[bn]]
          idx_b = match(bn, names(blocks))
          W_all[[k]][[idx_b]] = pad_and_name(W_all[[k]][[idx_b]], feats)
          P_all[[k]][[idx_b]] = pad_and_name(P_all[[k]][[idx_b]], feats)
        }
        names(W_all[[k]]) = names(P_all[[k]]) = block_names
      }
      names(W_all) = names(P_all) = comp_names

      # Training scores with the same deflation path as prediction
      T_mat_train = .mb_deflated_scores(X_list, W_all, P_all, block_names)$T
      dt_lat = data.table::as.data.table(T_mat_train)

      self$state$blocks = blocks
      self$state$center = center
      self$state$weights = W_all
      self$state$loadings = P_all
      self$state$ncomp = n_kept
      self$state$T_mat = T_mat_train
      self$state$obj_vec = obj
      self$state$p_values = pvals
      self$state$p_value_scope = if (isTRUE(pv$permutation_test)) {
        .mb_train_p_value_scope("mbspls")
      } else {
        NULL
      }
      self$state$ev_block = ev_blk
      self$state$ev_comp = ev_cmp
      self$state$converged = converged
      self$state$iterations = iterations
      self$state$latent_cor_train = unname(utils::tail(obj, 1))
      self$state$performance_metric = pv$performance_metric
      self$state$correlation_method = pv$correlation_method
      self$state$pkg_version = as.character(utils::packageVersion("mlr3mbspls"))

      if (!is.null(pv$log_env) && inherits(pv$log_env, "environment")) {
        sparsity = if (is.null(c_matrix)) {
          list(type = "c_vec", c_vec = stats::setNames(as.numeric(c_vec), names(blocks)))
        } else {
          list(type = "c_matrix", c_matrix = c_matrix)
        }

        payload = list(
          blocks       = blocks,
          center       = center,
          ncomp        = n_kept,
          weights      = W_all,
          loadings     = P_all,
          T_mat_train  = T_mat_train,
          comp_names   = comp_names,
          block_names  = names(blocks),
          sparsity     = sparsity,
          corr_method  = pv$correlation_method,
          perf_metric  = pv$performance_metric,
          converged    = converged,
          iterations   = iterations,
          time         = Sys.time()
        )
        if (isTRUE(pv$store_train_blocks)) {
          payload$X_train_blocks = X_list
        }
        payload$pipeop_id = self$id
        run_id = log_env_store_state(pv$log_env, payload, warn_overwrite = TRUE)
        self$state$run_id = run_id
      }

      lgr$info("Training done; last latent correlation = %.4f", utils::tail(obj, 1))

      # ---- output: append or replace
      if (isTRUE(pv$append)) {
        # Append LV columns to the original features
        dt_out = cbind(data.table::as.data.table(dt), dt_lat)
        data.table::setDT(dt_out)
        return(dt_out)
      } else {
        return(dt_lat)
      }
    },

    # ------------------------------- predict ---------------------------------
    .predict_dt = function(dt, levels, target = NULL) {
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "predict"),
        keep.null = TRUE)

      st = self$state
      block_names = names(st$blocks)
      B = length(block_names)

      # Ensure trained columns exist
      mb_assert_columns_present(
        colnames_dt = names(dt),
        required = unlist(st$blocks),
        context = sprintf("[%s] Prediction task", self$id),
        hint = "Apply the same preprocessing used during training and retain all trained feature columns before PipeOpMBsPLS."
      )

      # Build X_test and apply the training centre (absent in older states)
      X_test = lapply(st$blocks, function(cols) {
        m = as.matrix(dt[, ..cols])
        storage.mode(m) = "double"
        m
      })
      names(X_test) = block_names
      X_test = .mb_center_blocks(X_test, st$center,
        context = sprintf("[%s] Training centre", self$id))

      align_components = function(W, label) {
        for (k in seq_along(W)) {
          if (!is.list(W[[k]])) {
            stop(sprintf("Component %d of the selected prediction %s is not a block-wise list.", k, label), call. = FALSE)
          }
          for (bnm in block_names) {
            W[[k]][[bnm]] = mb_align_named_numeric(
              W[[k]][[bnm]],
              cols = colnames(X_test[[bnm]]),
              context = sprintf("Prediction %s for component %d, block '%s'", label, k, bnm)
            )
          }
          # Native routines match blocks by position.
          W[[k]] = W[[k]][block_names]
        }
        W
      }

      # ----------------- choose which weights to evaluate -----------------
      # The emitted LV features always use the raw training weights; this
      # choice only affects the payload and the prediction-side diagnostics.
      used_source = "raw"
      W_active = st$weights
      P_active = st$loadings
      K_active = length(W_active)

      st_env = NULL
      run_id_for_lookup = st$run_id %||% NULL
      if (!is.null(run_id_for_lookup) && nzchar(as.character(run_id_for_lookup)) &&
        !is.null(pv$log_env) && inherits(pv$log_env, "environment")) {
        st_env = tryCatch(
          .mbspls_state_from_env(
            pv$log_env,
            run_id = as.character(run_id_for_lookup),
            require_train_blocks = FALSE,
            where = "log_env"
          ),
          error = function(e) NULL
        )
      }
      stability_only = !is.null(st_env) && isTRUE(st_env$stability_only)

      get_env_weights = function(ci = FALSE, freq = FALSE) {
        if (is.null(st_env)) {
          return(NULL)
        }
        if (ci) {
          if (length(st_env$weights_stable_ci)) {
            return(list(
              weights = st_env$weights_stable_ci,
              loadings = st_env$loadings_stable_ci %||% NULL,
              source = "stable_ci"
            ))
          }
          if (identical(st_env$selection_method, "ci") && length(st_env$weights_stable)) {
            return(list(
              weights = st_env$weights_stable,
              loadings = st_env$loadings_stable %||% NULL,
              source = "stable_ci"
            ))
          }
        }
        if (freq) {
          if (length(st_env$weights_stable_frequency)) {
            return(list(
              weights = st_env$weights_stable_frequency,
              loadings = st_env$loadings_stable_frequency %||% NULL,
              source = "stable_frequency"
            ))
          }
          if (identical(st_env$selection_method, "frequency") && length(st_env$weights_stable)) {
            return(list(
              weights = st_env$weights_stable,
              loadings = st_env$loadings_stable %||% NULL,
              source = "stable_frequency"
            ))
          }
        }
        NULL
      }

      stable_request_error = function(requested) {
        requested = as.character(requested)[1L]
        run_label = if (!is.null(run_id_for_lookup) && nzchar(as.character(run_id_for_lookup))) {
          as.character(run_id_for_lookup)
        } else {
          "<latest>"
        }
        selection = switch(requested,
          stable_ci = "ci",
          stable_frequency = "frequency",
          requested
        )
        stop(sprintf(
          "predict_weights='%s' was requested, but matching stability-selected weights/loadings are not available in log_env for run_id='%s'. Run PipeOpMBsPLSBootstrapSelect with selection_method='%s' on the same log_env, or use predict_weights='raw'.",
          requested,
          run_label,
          selection
        ), call. = FALSE)
      }

      pick = pv$predict_weights %||% "auto"
      if (pick %in% c("stable_ci", "stable_frequency") && stability_only) {
        stop(sprintf(
          paste0(
            "predict_weights='%s' cannot be used because PipeOpMBsPLSBootstrapSelect ran with ",
            "stability_only = TRUE for run_id='%s'. In stability-only mode the LV features that ",
            "reach downstream learners are computed from the raw training weights, so a payload ",
            "evaluated with stable weights would describe a model no feature is derived from. ",
            "Use predict_weights = 'raw' (or 'auto'), or set stability_only = FALSE."
          ),
          pick,
          as.character(run_id_for_lookup)
        ), call. = FALSE)
      }
      if (identical(pick, "auto")) {
        if (stability_only) {
          lgr$info("[%s] predict_weights='auto': the bootstrap-selection stage ran in stability-only mode; using 'raw'.", self$id)
        } else if (!is.null(st_env) && length(st_env$weights_stable)) {
          W_active = st_env$weights_stable
          P_active = st_env$loadings_stable %||% NULL
          K_active = length(W_active)
          used_source = paste0("stable_", st_env$selection_method %||% "ci")
          # Guard: if all components have empty block weights, fall back to raw
          all_populated = K_active > 0L && all(vapply(seq_len(K_active), function(ki) {
            is.list(W_active[[ki]]) && length(W_active[[ki]]) > 0L &&
              all(vapply(W_active[[ki]], function(w) length(w) > 0L, logical(1L)))
          }, logical(1L)))
          if (all_populated) {
            lgr$info("[%s] predict_weights='auto': using '%s' from log_env.", self$id, used_source)
          } else {
            lgr$warn("[%s] predict_weights='auto': stable weights exist but have 0 components or empty block weights; falling back to 'raw'.", self$id)
            W_active = st$weights
            P_active = st$loadings
            K_active = length(W_active)
            used_source = "raw"
          }
        } else {
          lgr$info("[%s] predict_weights='auto': no stable weights found in log_env; falling back to 'raw'.", self$id)
        }
      } else if (identical(pick, "stable_ci")) {
        selected = get_env_weights(ci = TRUE, freq = FALSE)
        if (is.null(selected)) {
          stable_request_error("stable_ci")
        }
        W_active = selected$weights
        P_active = selected$loadings
        K_active = length(W_active)
        used_source = selected$source
      } else if (identical(pick, "stable_frequency")) {
        selected = get_env_weights(ci = FALSE, freq = TRUE)
        if (is.null(selected)) {
          stable_request_error("stable_frequency")
        }
        W_active = selected$weights
        P_active = selected$loadings
        K_active = length(W_active)
        used_source = selected$source
      } else {
        used_source = "raw"
      }

      if (!is.list(W_active) || length(W_active) < K_active) {
        stop(sprintf(
          "Selected prediction weights are inconsistent: expected %d component(s), got %d.",
          K_active,
          length(W_active %||% list())
        ), call. = FALSE)
      }
      if (!is.list(P_active) || length(P_active) < K_active) {
        stop(sprintf(
          "Selected prediction loadings are unavailable or incomplete for weights_source='%s'. Prediction requires matching loadings to preserve the trained deflation path.",
          used_source
        ), call. = FALSE)
      }

      # ---- align the evaluated and the emitted weights to trained features ----
      W_active = align_components(W_active[seq_len(K_active)], "weights")
      P_active = align_components(P_active[seq_len(K_active)], "loadings")
      W_raw = align_components(st$weights, "weights")
      P_raw = align_components(st$loadings, "loadings")

      # Then compute EV/MAC safely
      test_ev_results = compute_test_ev(
        X_blocks_test      = X_test,
        W_all              = W_active,
        P_all              = P_active,
        deflate            = TRUE,
        performance_metric = pv$performance_metric,
        correlation_method = pv$correlation_method,
        loading_source     = "train"
      )

      use_frob = identical(pv$performance_metric, "frobenius")
      use_spear = identical(pv$correlation_method, "spearman")

      val_test = pv$val_test
      val_test_n = pv$val_test_n
      val_test_permute_all = pv$val_test_permute_all

      if (val_test != "none" && B < 2L) {
        stop("Prediction-side validation requires at least two blocks.", call. = FALSE)
      }
      if (val_test == "bootstrap" && nrow(dt) < 10L) {
        stop(sprintf(
          "Prediction-side bootstrap validation requires at least 10 test rows; got %d.",
          nrow(dt)
        ), call. = FALSE)
      }
      if (val_test == "bootstrap" && val_test_n < 2L) {
        stop("Prediction-side bootstrap validation requires at least two replicates.",
          call. = FALSE)
      }
      if (val_test == "bootstrap" &&
        (!is.finite(pv$val_test_alpha) || pv$val_test_alpha <= 0 ||
          pv$val_test_alpha >= 1)) {
        stop("Prediction-side bootstrap `val_test_alpha` must be strictly between 0 and 1.",
          call. = FALSE)
      }

      validation_streams = if (val_test == "none" ||
        is.null(pv$seed_validation)) {
        NULL
      } else {
        mb_rng_streams(K_active, pv$seed_validation)
      }
      run_validation = function(component, fn) {
        if (is.null(validation_streams)) {
          fn()
        } else {
          with_rng_stream_local(validation_streams[[component]], fn)
        }
      }

      # -------- optional prediction-side validation, one component at a time --------
      component_diagnostic = function(k, Tk, X_cur) {
        Wk = W_active[[k]]
        # The native diagnostics need two blocks with non-zero weights and
        # non-degenerate scores (the criterion of compute_scores_core).
        informative = vapply(seq_len(B), function(bi) {
          t_b = Tk[, bi]
          any(Wk[[block_names[[bi]]]] != 0) && all(is.finite(t_b)) &&
            isTRUE(stats::var(t_b) > 1e-12)
        }, logical(1L))
        residual_ok = all(vapply(X_cur, function(x) sum(abs(x)) >= 1e-12, logical(1L)))
        status = if (sum(informative) < 2L) {
          "not computed: fewer than two blocks with non-zero weights and non-degenerate scores"
        } else if (!residual_ok) {
          "not computed: a deflated test block is numerically zero"
        } else {
          "computed"
        }
        if (!identical(status, "computed")) {
          lgr$warn("[%s] Component %d: prediction-side %s diagnostic %s.", self$id, k, val_test, status)
          return(list(status = status))
        }
        if (val_test == "permutation") {
          res = run_validation(k, function() {
            cpp_perm_test_oos(
              X_test = X_cur,
              W_trained = Wk,
              n_perm = val_test_n,
              spearman = use_spear,
              frobenius = use_frob,
              permute_all_blocks = isTRUE(val_test_permute_all),
              early_stop_threshold = 1.0
            )
          })
          out = if (is.list(res)) {
            list(
              p = if (is.null(res$p_value)) NA_real_ else as.numeric(res$p_value),
              stat = if (is.null(res$stat_obs)) NA_real_ else as.numeric(res$stat_obs)
            )
          } else {
            list(p = as.numeric(res), stat = NA_real_)
          }
          lgr$info("Component %d: prediction-side permutation test p = %s",
            k, if (is.na(out$p)) "NA" else formatC(out$p, digits = 3, format = "f"))
          return(c(list(status = status), out))
        }
        bres = run_validation(k, function() {
          cpp_bootstrap_test_oos(
            X_test = X_cur,
            W_trained = Wk,
            n_boot = val_test_n,
            spearman = use_spear,
            frobenius = use_frob,
            alpha = pv$val_test_alpha
          )
        })
        list(status = status, boot = bres)
      }

      diag = .mb_deflated_scores(
        X_test, W_active, P_active, block_names,
        fn = if (val_test == "none") NULL else component_diagnostic
      )
      T_mat_test = diag$T
      # Emitted features: raw training weights, centring and deflation
      T_emit = if (identical(used_source, "raw")) {
        T_mat_test
      } else {
        .mb_deflated_scores(X_test, W_raw, P_raw, block_names)$T
      }
      dt_lat = data.table::as.data.table(T_emit)

      comp_names = sprintf("LC_%02d", seq_len(K_active))
      val_test_p = rep(NA_real_, K_active)
      val_test_stat = rep(NA_real_, K_active)
      val_test_status = rep(NA_character_, K_active)
      val_bootstrap_results = NULL
      val_bootstrap_vectors = NULL
      if (val_test != "none") {
        val_bootstrap_rows = vector("list", K_active)
        val_bootstrap_vectors = if (val_test == "bootstrap") vector("list", K_active) else NULL
        for (k in seq_len(K_active)) {
          d = diag$diagnostics[[k]]
          val_test_status[k] = d$status
          if (val_test == "permutation") {
            val_test_p[k] = d$p %||% NA_real_
            val_test_stat[k] = d$stat %||% NA_real_
            next
          }
          if (identical(d$status, "computed")) {
            bres = d$boot
            observed_correlation = as.numeric(bres$stat_obs %||% NA_real_)
            boot_mean = as.numeric(bres$boot_mean %||% NA_real_)
            boot_bias = as.numeric(bres$bias %||% (boot_mean - observed_correlation))
            boot_se = as.numeric(bres$boot_se %||% NA_real_)
            ci_lower = as.numeric(bres$ci_lower %||% NA_real_)
            ci_upper = as.numeric(bres$ci_upper %||% NA_real_)
            n_boot_done = as.integer(bres$n_boot %||% val_test_n)
            n_boot_requested = as.integer(bres$n_boot_requested %||% val_test_n)
            n_boot_failed = as.integer(bres$n_boot_failed %||%
              (n_boot_requested - n_boot_done))
            conf = as.numeric(bres$confidence_level %||%
              (1 - pv$val_test_alpha))
            interval_type = as.character(bres$interval_type %||% "percentile")
            p_value_note = as.character(bres$p_value_note %||% paste(
              "Not computed: an ordinary bootstrap distribution estimates",
              "uncertainty and is not a null distribution for hypothesis testing."
            ))
            val_bootstrap_vectors[[k]] = as.numeric(bres$replicates %||% numeric())
          } else {
            observed_correlation = boot_mean = boot_bias = boot_se = NA_real_
            ci_lower = ci_upper = NA_real_
            n_boot_done = n_boot_failed = 0L
            n_boot_requested = as.integer(val_test_n)
            conf = 1 - pv$val_test_alpha
            interval_type = "percentile"
            p_value_note = d$status
            val_bootstrap_vectors[[k]] = numeric()
          }
          val_test_stat[k] = observed_correlation
          val_bootstrap_rows[[k]] = data.table::data.table(
            component = k,
            estimate = observed_correlation,
            bootstrap_mean = boot_mean,
            bias = boot_bias,
            standard_error = boot_se,
            conf_low = ci_lower,
            conf_high = ci_upper,
            confidence_level = conf,
            interval_type = interval_type,
            replicates_requested = n_boot_requested,
            replicates_effective = n_boot_done,
            replicates_failed = n_boot_failed,
            p_value = NA_real_,
            p_value_note = p_value_note,
            # Backward-compatible descriptive aliases. The former p-value
            # alias remains present but deliberately has no numeric value.
            observed_correlation = observed_correlation,
            boot_mean = boot_mean,
            boot_se = boot_se,
            boot_p_value = NA_real_,
            boot_ci_lower = ci_lower,
            boot_ci_upper = ci_upper,
            n_boot = n_boot_done
          )
        }
        if (val_test == "bootstrap") {
          val_bootstrap_results = data.table::rbindlist(val_bootstrap_rows, fill = TRUE)
        }
      }

      ev_block_test = as.matrix(test_ev_results$ev_block)
      ev_comp_test = as.numeric(test_ev_results$ev_comp)
      mac_comp_test = as.numeric(test_ev_results$mac_comp)
      colnames(ev_block_test) = block_names
      rownames(ev_block_test) = comp_names
      names(ev_comp_test) = comp_names
      names(mac_comp_test) = comp_names

      log_env = self$param_set$values$log_env
      if (!is.null(log_env) && inherits(log_env, "environment")) {
        payload = list(
          mac_comp = mac_comp_test,
          ev_block = ev_block_test,
          ev_comp = ev_comp_test,
          ev_block_cum = as.matrix(test_ev_results$ev_block_cum),
          ev_comp_cum = as.numeric(test_ev_results$ev_comp_cum),
          corr_method = pv$correlation_method,
          T_mat = T_mat_test,
          blocks = block_names,
          perf_metric = pv$performance_metric,
          weights = stats::setNames(W_active, comp_names),
          loadings = stats::setNames(P_active, comp_names),
          weights_source = used_source,
          emitted_weights_source = "raw",
          time = Sys.time()
        )
        if (pv$val_test != "none") {
          names(val_test_stat) = comp_names
          names(val_test_status) = comp_names
          payload$val_test_stat = val_test_stat
          payload$val_test_status = val_test_status
          if (pv$val_test == "permutation") {
            names(val_test_p) = comp_names
            payload$val_test_p = val_test_p
            payload$val_test_params = list(
              n_perm = pv$val_test_n,
              alternative = "greater",
              correction = "(b + 1) / (B + 1)",
              permute_all_blocks = isTRUE(val_test_permute_all),
              seed = pv$seed_validation,
              rng_streams = validation_streams,
              scope = paste(
                "Conditional association diagnostic with fixed trained",
                "weights; not a full-pipeline permutation test."
              )
            )
          } else if (pv$val_test == "bootstrap") {
            payload$val_bootstrap = val_bootstrap_results
            payload$val_boot_vectors = val_bootstrap_vectors
            payload$val_test_params = list(
              n_boot = pv$val_test_n,
              alpha = pv$val_test_alpha,
              interval_type = "percentile",
              seed = pv$seed_validation,
              rng_streams = validation_streams,
              scope = "Descriptive uncertainty; no null-hypothesis p-value."
            )
          }
        }
        payload$run_id = st$run_id %||% self$state$run_id %||% NULL
        log_env_store_last(log_env, payload, run_id = payload$run_id)
      }

      # Output: append the latent variables to the input or return them alone
      if (isTRUE(pv$append)) {
        dt_out = cbind(data.table::as.data.table(dt), dt_lat)
        data.table::setDT(dt_out)
        return(dt_out)
      } else {
        return(dt_lat)
      }
    },

    .additional_phash_input = function() {
      list(blocks = self$blocks)
    }
  )
)

# ------------------------------------------------------------------------------
# Internal helpers shared by PipeOpMBsPLS, PipeOpMBsPLSXY and PipeOpMBsPCA
# ------------------------------------------------------------------------------

# Resolve the declared blocks against the training data (see
# mb_resolve_block_columns()) and keep the numeric columns with finite,
# positive variance; blocks without such columns are dropped. `p_struct` is the
# structural width of each retained block: its resolved numeric columns before
# the data-dependent variance filter.
.mb_training_blocks = function(dt, blocks) {
  resolved = mb_resolve_block_columns(names(dt), blocks)
  numeric_cols = lapply(resolved, function(cols) {
    cols[vapply(cols, function(cl) is.numeric(dt[[cl]]), logical(1L))]
  })
  retained = lapply(numeric_cols, function(cols) {
    cols[vapply(cols, function(cl) mb_has_finite_variance(dt[[cl]]), logical(1L))]
  })
  retained = Filter(length, retained)
  list(
    blocks = retained,
    p_struct = lengths(numeric_cols)[names(retained)]
  )
}

# Column means of each block, named by column.
.mb_block_means = function(X_list) {
  lapply(X_list, function(m) stats::setNames(colMeans(m), colnames(m)))
}

# Subtract stored column means from each block. `center = NULL` (states fitted
# before centring was introduced) leaves the blocks unchanged.
.mb_center_blocks = function(X_list, center, context = "Training centre") {
  if (is.null(center)) {
    return(X_list)
  }
  out = lapply(names(X_list), function(bn) {
    m = X_list[[bn]]
    mu = if (is.null(colnames(m))) {
      # Unnamed matrices can only be matched by position.
      if (length(center[[bn]]) != ncol(m)) {
        stop(sprintf(
          "%s for block '%s' has %d values but the block has %d unnamed columns.",
          context, bn, length(center[[bn]]), ncol(m)
        ), call. = FALSE)
      }
      as.numeric(center[[bn]])
    } else {
      mb_align_named_numeric(
        center[[bn]],
        cols = colnames(m),
        context = sprintf("%s for block '%s'", context, bn)
      )
    }
    m - rep(mu, each = nrow(m))
  })
  names(out) = names(X_list)
  out
}

# Align a user-supplied c_matrix with the blocks retained for fitting. Named
# rows must be unique and non-empty and cover either all declared or all
# retained blocks (plus optional `extra` rows such as ".target"). An unnamed
# matrix is matched by position to the declared layout: the declared blocks,
# optionally followed by the `extra` rows. Returns the rows of the retained
# blocks followed by the `extra` rows present.
.mb_align_c_matrix = function(c_matrix, declared, retained, extra = character(0)) {
  if (!is.matrix(c_matrix) || !is.numeric(c_matrix) || !ncol(c_matrix) ||
    anyNA(c_matrix) || any(!is.finite(c_matrix))) {
    stop("`c_matrix` must be a finite numeric matrix with at least one column.",
      call. = FALSE)
  }
  if (is.null(rownames(c_matrix))) {
    if (nrow(c_matrix) == length(declared)) {
      rownames(c_matrix) = declared
    } else if (length(extra) && nrow(c_matrix) == length(declared) + length(extra)) {
      rownames(c_matrix) = c(declared, extra)
    } else {
      layout = if (length(extra)) {
        sprintf(
          "%d rows (declared blocks) or %d rows (declared blocks + %s)",
          length(declared), length(declared) + length(extra),
          paste(sprintf("'%s'", extra), collapse = ", ")
        )
      } else {
        sprintf("%d rows (declared blocks)", length(declared))
      }
      stop(sprintf(
        paste0(
          "An unnamed c_matrix is matched by position to the declared blocks ",
          "and must have %s; got %d. Supply row names to address retained ",
          "blocks explicitly."
        ),
        layout, nrow(c_matrix)
      ), call. = FALSE)
    }
  }
  rn = rownames(c_matrix)
  if (anyNA(rn) || any(!nzchar(rn)) || anyDuplicated(rn)) {
    stop("c_matrix row names must be unique and non-empty.", call. = FALSE)
  }
  block_rows = setdiff(rn, extra)
  if (!setequal(block_rows, declared) && !setequal(block_rows, retained)) {
    stop(
      paste0(
        "Named c_matrix rows must match either all declared or all ",
        "retained blocks exactly."
      ),
      call. = FALSE
    )
  }
  cn = colnames(c_matrix)
  if (!is.null(cn) && (anyNA(cn) || any(!nzchar(cn)) || anyDuplicated(cn))) {
    stop("c_matrix column names must be unique and non-empty.", call. = FALSE)
  }
  c_matrix[c(retained, intersect(extra, rn)), , drop = FALSE]
}

# Log budgets of a prepared c_matrix that were capped at sqrt(p) of the
# retained columns (see .mb_prepare_c_matrix()) and drop the marker attribute.
.mb_finalize_c_matrix = function(c_matrix, X_list, id) {
  capped = attr(c_matrix, "capped")
  attr(c_matrix, "capped") = NULL
  if (is.matrix(capped) && any(capped)) {
    rows = rownames(capped)[rowSums(capped) > 0]
    lgr$info(
      "[%s] c_matrix budgets above sqrt(p) of the retained columns are nonbinding and were capped: %s.",
      id,
      paste(sprintf("%s (p = %d)", rows, vapply(X_list[rows], ncol, integer(1L))), collapse = ", ")
    )
  }
  c_matrix
}

# Warn once about kept components whose solver run did not converge.
.mb_warn_nonconverged = function(converged, context, max_iter) {
  bad = names(converged)[!is.na(converged) & !converged]
  if (length(bad)) {
    warning(sprintf(
      paste0(
        "%s: the solver did not converge within %d iterations for component(s) %s; ",
        "the last iterate is returned."
      ),
      context, as.integer(max_iter), paste(bad, collapse = ", ")
    ), call. = FALSE)
  }
  invisible(bad)
}

# Scope statement stored with train-time permutation p-values.
.mb_train_p_value_scope = function(model = c("mbspls", "mbspca")) {
  switch(match.arg(model),
    mbspls = paste(
      "Conditional component-wise diagnostic with fixed preprocessing and",
      "hyperparameters; not a full-pipeline permutation test."
    ),
    mbspca = paste(
      "Conditional component-wise cross-block diagnostic (rows permuted",
      "independently per block) with fixed preprocessing and hyperparameters;",
      "not a full-pipeline permutation test."
    )
  )
}

# Block scores of sequential components with block-wise deflation (X_b is
# replaced by X_b - t_b p_b^T). `W` and `P` are component lists of block-wise vectors
# aligned to the block columns. If given, `fn(k, Tk, X_cur)` is called for
# every component with its score matrix and the residual blocks it was
# computed from; its results are returned in `diagnostics`.
.mb_deflated_scores = function(X_list, W, P, block_names, fn = NULL) {
  K = length(W)
  X_cur = X_list
  T_list = vector("list", K)
  diagnostics = vector("list", K)
  for (k in seq_len(K)) {
    Tk = matrix(0, nrow(X_list[[1L]]), length(block_names),
      dimnames = list(NULL, paste0("LV", k, "_", block_names)))
    for (bi in seq_along(block_names)) {
      bn = block_names[[bi]]
      Tk[, bi] = X_cur[[bn]] %*% as.numeric(W[[k]][[bn]])
    }
    T_list[[k]] = Tk
    if (!is.null(fn)) {
      diagnostics[[k]] = fn(k, Tk, X_cur)
    }
    if (k < K) {
      for (bi in seq_along(block_names)) {
        bn = block_names[[bi]]
        pb = P[[k]][[bn]]
        if (is.null(pb)) {
          stop(sprintf("Loadings are missing for component %d, block '%s'.", k, bn), call. = FALSE)
        }
        X_cur[[bn]] = X_cur[[bn]] - Tk[, bi] %*% t(as.numeric(pb))
      }
    }
  }
  list(T = do.call(cbind, T_list), diagnostics = diagnostics)
}
