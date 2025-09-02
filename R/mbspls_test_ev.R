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

