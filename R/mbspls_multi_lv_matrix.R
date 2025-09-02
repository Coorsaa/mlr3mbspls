#' Sequential MB‑sPLS extraction with a full sparsity matrix
#'
#' @description
#' High‑level R wrapper that extracts up to \code{K} latent components
#' \emph{sequentially}, allowing each block and each component to have its
#' own sparsity level.  Per component it calls the existing C++ one‑LV solver
#' \code{cpp_mbspls_one_lv()}, then deflates all blocks on the R side.
#' The return structure is identical to \code{cpp_mbspls_multi_lv()} so that
#' downstream code (e.g. \code{PipeOpMBsPLS}) can treat both interchangeably.
#'
#' @param X_blocks \code{list} of numeric matrices; one element per block.
#' @param c_matrix Numeric matrix of dimension \eqn{B \times K} containing the
#'   sparsity value for every block (row) and component (column).
#' @param max_iter,max_tol,method,spearman,do_perm,n_perm,alpha
#'   See \code{cpp_mbspls_multi_lv}.
#'
#' @return
#' List with components \code{W}, \code{P}, \code{T_mat}, \code{objective},
#' \code{p_values}, \code{ev_block}, \code{ev_comp}; identical in structure
#' to the C++ multi‑LV solver.
#'
#' @keywords internal
#' @export
mbspls_multi_lv_matrix = function(
  X_blocks,
  c_matrix,
  max_iter  = 500L,
  max_tol   = 1e-4,
  spearman  = FALSE,
  do_perm   = FALSE,
  n_perm    = 100L,
  alpha     = 0.05,
  frobenius = FALSE) {

  B = length(X_blocks)
  K = ncol(c_matrix)
  stopifnot(
    is.list(X_blocks), B >= 2L,
    is.matrix(c_matrix), nrow(c_matrix) == B
  )

  lgr::lgr$info("Sequential MB‑sPLS (%d blocks, %d comps, custom C‑matrix)",
                B, K)

  # pre‑compute total SSqs for EVs
  ss_tot = vapply(X_blocks, function(x) sum(x * x), numeric(1))
  n      = nrow(X_blocks[[1]])

  W_all = P_all = vector("list", K)
  T_all <- matrix(0.0, n, B * K)
  obj_vec  = p_vec  = numeric(0)
  ev_block = matrix(NA_real_, K, B)
  ev_comp  = numeric(0)

  for (k in seq_len(K)) {

    lgr::lgr$debug("↪  component %d / %d", k, K)

    fit = cpp_mbspls_one_lv(
      X_blocks,
      c_matrix[, k],
      max_iter,
      max_tol,
      frobenius,
      spearman
    )

    Wk = lapply(fit$W, as.numeric)           # weight vectors
    Tk = fit$T_mat                           # n × B score matrix
    obj_k = fit$objective

    # optional permutation test
    early_stop <- FALSE
    p_k = NA_real_
    if (do_perm) {
      p_k = perm_test_component(
        X_orig              = X_blocks,
        W_orig              = Wk,
        c_vec               = c_matrix[, k],
        n_perm              = n_perm,
        spearman            = spearman,
        max_iter            = max_iter,
        tol                 = max_tol,
        early_stop_threshold= alpha,       # <- use perm_alpha here
        frobenius           = frobenius
      )
      if (p_k > alpha) {
        if (k == 1L) {              # first component → keep it, but stop afterwards
          lgr$info("    LV‑1 not significant (p = %.3f) – keep but stop", p_k)
          early_stop <- TRUE
        } else {                    # later component → discard and stop
          lgr$info("    LV‑%d not significant (p = %.3f) – discard and stop", k, p_k)
          break                     # nothing has been stored yet → nothing to undo
        }
      }
    }

    # loadings + EVs + deflation
    Pk = vector("list", B)
    ss_exp_total = 0
    for (b in seq_len(B)) {
      tb = Tk[, b, drop = FALSE]
      pb = crossprod(X_blocks[[b]], tb) / drop(crossprod(tb))
      Pk[[b]] = as.numeric(pb)
      ss_exp_block = drop(crossprod(tb)) * drop(crossprod(pb))
      ev_block[k, b] = ss_exp_block / ss_tot[b]
      ss_exp_total  = ss_exp_total + ss_exp_block

      # deflate
      X_blocks[[b]] = X_blocks[[b]] - tcrossprod(tb, pb)
    }
    ev_comp[k] = ss_exp_total / sum(ss_tot)

    # store
    W_all[[k]] = Wk
    P_all[[k]] = Pk
    obj_vec    = c(obj_vec, obj_k)
    p_vec      = c(p_vec,  p_k)
    T_all[, ((k - 1L) * B + 1L):(k * B)] <- Tk

    if (early_stop)
      break
  }

  # trim matrices in case of early stop
  keep = length(obj_vec)
  ev_block = ev_block[seq_len(keep), , drop = FALSE]
  ev_comp  = ev_comp[seq_len(keep)]

  list(
    W         = W_all[seq_len(keep)],
    P         = P_all[seq_len(keep)],
    T_mat     = T_all,
    objective = obj_vec,
    p_values  = p_vec,
    ev_block  = ev_block,
    ev_comp   = ev_comp
  )
}

#' Compute test metrics for MB-sPLS model evaluation
#'
#' @description
#' Computes both explained variance (EV) and mean absolute correlation for test data
#' using pre-trained MB-sPLS weights and loadings. This function is intended
#' to be called during the predict phase to assess how well the model
#' generalizes to new data.
#'
#' The mean absolute correlation metric matches the training objective used in
#' the C++ implementation, which optimizes the average of |r| between all pairs
#' of block scores for each component.
#'
#' @param X_blocks_test \code{list} of numeric matrices; test data blocks.
#' @param W_all \code{list} of weight vectors from training (one per component).
#' @param P_all \code{list} of loading vectors from training (one per component).
#' @param deflate \code{logical(1)} Whether to perform deflation between components.
#'   Default \code{TRUE} to match training behavior.
#'
#' @return
#' List with components:
#' \describe{
#'   \item{\code{ev_block}}{Matrix of dimension \code{K × B} with block-wise explained variances.}
#'   \item{\code{ev_comp}}{Vector of length \code{K} with component-wise total explained variances.}
#'   \item{\code{mac_comp}}{Vector of length \code{K} with component-wise mean absolute correlations (matches training objective).}
#'   \item{\code{T_mat}}{Matrix of test scores (n_test × (K*B)).}
#' }
#'
#' @keywords internal
#' @export
compute_test_ev = function(X_blocks_test, W_all, P_all, deflate = TRUE, performance_metric, correlation_method) {

  B = length(X_blocks_test)
  K = length(W_all)
  n_test = nrow(X_blocks_test[[1]])

  method <- match.arg(correlation_method, c("pearson","spearman"))

  stopifnot(
    is.list(X_blocks_test), B >= 1L,
    is.list(W_all), length(W_all) == K,
    is.list(P_all), length(P_all) == K,
    is.logical(deflate), length(deflate) == 1L
  )

  # pre-compute total SSqs for test data
  ss_tot_test = vapply(X_blocks_test, function(x) sum(x * x), numeric(1))

  # initialize output structures
  ev_block_test = matrix(NA_real_, K, B)
  ev_comp_test = numeric(K)
  mac_comp_test = numeric(K)  # mean absolute correlation per component
  T_all_test = matrix(numeric(0), n_test, 0)

  # make a working copy for deflation
  X_work = if (deflate) {
    lapply(X_blocks_test, function(x) x)  # deep copy
  } else {
    X_blocks_test  # use original data for all components
  }

  for (k in seq_len(K)) {

    Wk = W_all[[k]]
    Pk = P_all[[k]]

    # compute test scores using training weights
    Tk_test = matrix(0, n_test, B)
    for (b in seq_len(B)) {
      Tk_test[, b] = X_work[[b]] %*% Wk[[b]]
    }

    # compute explained variance for this component
    ss_exp_total = 0
    for (b in seq_len(B)) {
      tb_test = Tk_test[, b, drop = FALSE]

      # EV = variance explained by this component's score
      # using the training loadings
      ss_exp_block = drop(crossprod(tb_test)) * drop(crossprod(Pk[[b]]))
      ev_block_test[k, b] = ss_exp_block / ss_tot_test[b]
      ss_exp_total = ss_exp_total + ss_exp_block

      # deflate test data if requested (for sequential components)
      if (deflate && k < K) {
        X_work[[b]] = X_work[[b]] - tcrossprod(tb_test, Pk[[b]])
      }
    }

    # total EV for this component across all blocks
    ev_comp_test[k] = ss_exp_total / sum(ss_tot_test)

    # compute mean absolute correlation between block scores (matches training objective)
    use_frob <- (match.arg(performance_metric, c("mac","frobenius")) == "frobenius")
    mac_k = 0.0 # will store either ⟨|r|⟩ or Frobenius
    valid_pairs = 0L
    if (B > 1) {
      for (i in seq_len(B - 1)) {
        for (j in (i + 1):B) {
          # compute correlation between block scores
          r <- stats::cor(Tk_test[, i], Tk_test[, j], method = method)
          if (is.finite(r)) {
            if (use_frob) {
              mac_k <- mac_k + r*r          # Σ r²
            } else {
              mac_k <- mac_k + abs(r)       # Σ |r|
            }
            valid_pairs <- valid_pairs + 1L
          }
        }
      }
    }
    if (valid_pairs) {
      mac_comp_test[k] <- if (use_frob)
        sqrt(mac_k)                   # mac_k already holds Σ r²
      else
        mac_k / valid_pairs           # mean |r|
    }

    # store test scores
    T_all_test = cbind(T_all_test, Tk_test)
  }

  list(
    ev_block = ev_block_test,
    ev_comp  = ev_comp_test,
    mac_comp = mac_comp_test,  # mean absolute correlation (matches training objective)
    T_mat    = T_all_test
  )
}

#' Compute test metrics during prediction (convenience wrapper)
#'
#' @description
#' Convenience wrapper for computing test metrics that can be easily integrated
#' into PipeOpMBsPLS predict method. This function extracts the necessary
#' components from the PipeOp state and calls \code{compute_test_ev}.
#'
#' Returns both explained variance and mean absolute correlation metrics for
#' test data, where the mean absolute correlation matches the training objective.
#'
#' @param X_blocks_test \code{list} of test data matrices.
#' @param state \code{list} PipeOp state containing \code{weights}, \code{loadings}, etc.
#'
#' @return
#' List with test metrics (see \code{compute_test_ev}).
#'
#' @keywords internal
#' @export
compute_pipeop_test_ev = function(X_blocks_test, state) {
  compute_test_ev(
    X_blocks_test = X_blocks_test,
    W_all = state$weights,
    P_all = state$loadings,
    deflate = TRUE,
    performance_metric = state$performance_metric,
    correlation_method = state$correlation_method %||% "pearson"
  )
}

