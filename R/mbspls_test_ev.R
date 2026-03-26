#' Compute out-of-sample EV and latent correlation for MB-sPLS components
#'
#' @description
#' Computes prediction-side diagnostics for a fitted MB-sPLS model on a set of
#' test blocks. For each component \eqn{k} and block \eqn{b}, the function:
#' \itemize{
#'   \item projects the (optionally deflated) test residual block
#'         \eqn{X^{(k-1)}_{b,\mathrm{test}}} onto the trained weight vector
#'         \eqn{w^{(k)}_b} to obtain a test score \eqn{t^{(k)}_b};
#'   \item computes a prediction-side latent correlation (MAC or Frobenius)
#'         between block score vectors \eqn{t^{(k)}_b};
#'   \item computes explained variance (EV) on test data as a \emph{sum-of-squares
#'         (SS) reduction} induced by applying the rank-1 deflation step
#'         \eqn{X \leftarrow X - t p^\top}.
#' }
#'
#' @details
#' \strong{Explained variance definition (out-of-sample SS reduction).}
#' For each block \eqn{b} and component \eqn{k}, EV is computed on the current
#' residual matrix as:
#' \deqn{\mathrm{SS}_{\text{exp}}^{(k,b)} =
#'       \|X_{b,\mathrm{test}}^{(k-1)}\|_F^2 -
#'       \|X_{b,\mathrm{test}}^{(k)}\|_F^2,}
#' where
#' \deqn{X_{b,\mathrm{test}}^{(k)} =
#'       X_{b,\mathrm{test}}^{(k-1)} - t_{b}^{(k)} (p_{b}^{(k)})^\top.}
#' The reported \code{ev_block} values are \emph{incremental} EVs:
#' \deqn{\mathrm{EV}_{k,b} = \mathrm{SS}_{\text{exp}}^{(k,b)} / \|X_{b,\mathrm{test}}^{(0)}\|_F^2.}
#' Cumulative EVs (\code{ev_block_cum}, \code{ev_comp_cum}) are obtained by
#' accumulating SS reductions across components and dividing by the \emph{original}
#' test SS.
#'
#' Because this is a true out-of-sample measure, incremental EV may be
#' \emph{negative} (the deflation step increases SS on new data), unless you
#' clamp values via \code{clamp_ev}.
#'
#' \strong{Deflation behavior.}
#' If \code{deflate = TRUE} (recommended; matches training), components are applied
#' sequentially and the test residuals are updated after each component.
#' If \code{deflate = FALSE}, each component is evaluated on the original test
#' blocks; in that case, \code{ev_block} and \code{ev_comp} are marginal SS
#' reductions for each component applied in isolation, and cumulative EVs may
#' not represent a true cumulative reconstruction.
#'
#' \strong{Loadings source.}
#' The deflation/loading vector \eqn{p_b^{(k)}} is chosen by \code{loading_source}:
#' \itemize{
#'   \item \code{"train"}: use the supplied training loadings \code{P_all} (aligned
#'         to test columns by name when available);
#'   \item \code{"test_ls"}: recompute loadings on the current test residual via
#'         least squares,
#'         \eqn{p = X^\top t / (t^\top t)},
#'         yielding a "weights-only" EV diagnostic (note this uses test data to
#'         estimate \eqn{p}).
#' }
#' If \code{P_all} is \code{NULL} or empty, the function automatically falls back
#' to \code{"test_ls"}.
#'
#' \strong{Latent correlation (MAC/Frobenius).}
#' For each component, the function computes pairwise correlations between block
#' score vectors \eqn{t_b^{(k)}} using \code{stats::cor} with
#' \code{correlation_method}. Blocks whose score variance is \eqn{\le} \code{eps_var}
#' are treated as invalid for the objective: their scores are set to zero and they
#' are excluded from the MAC/Frobenius aggregation.
#' \itemize{
#'   \item \code{performance_metric = "mac"}: mean absolute correlation
#'         \eqn{\langle |r| \rangle};
#'   \item \code{performance_metric = "frobenius"}: Frobenius-style norm
#'         \eqn{\sqrt{\sum r^2}} over valid block pairs.
#' }
#'
#' \strong{Global EV across blocks.}
#' \code{ev_comp} and \code{ev_comp_cum} are computed by summing SS reductions across
#' blocks and dividing by the summed original test SS. This is SS-weighted; large /
#' high-dimensional blocks can dominate the global EV.
#'
#' @param X_blocks_test \code{list} of numeric matrices. Test data blocks, one matrix
#'   per block, each of dimension \code{n_test × p_b}. All blocks must have the same
#'   number of rows (samples). Column names are used to align named weights/loadings.
#' @param W_all \code{list}. Component-wise weight vectors learned in training.
#'   Must be a list of length \code{K}; each element is a block-wise list of length
#'   \code{B} containing weight vectors. If a weight vector is named, it is aligned
#'   to the corresponding block's column names; missing entries are set to zero.
#' @param P_all \code{list} or \code{NULL}. Optional component-wise block loadings.
#'   Same nesting convention as \code{W_all}. If \code{NULL} or empty, loadings are
#'   computed from test data when \code{loading_source = "test_ls"}.
#' @param deflate \code{logical(1)}. If \code{TRUE} (default), apply sequential
#'   deflation on test blocks between components. If \code{FALSE}, evaluate each
#'   component on the original test blocks without updating residuals.
#' @param performance_metric \code{character(1)}. Latent correlation metric:
#'   \code{"mac"} (mean absolute correlation) or \code{"frobenius"} (sqrt-sum-squared).
#' @param correlation_method \code{character(1)}. Correlation estimator for score
#'   correlations: \code{"pearson"} or \code{"spearman"}.
#' @param eps_var \code{numeric(1)}. Variance threshold for score validity. Block
#'   scores with variance \eqn{\le} \code{eps_var} are considered invalid for the
#'   objective (and are set to zero).
#' @param loading_source \code{character(1)} Source of loadings for the rank-1
#'   deflation step used in out-of-sample EV:
#'   \itemize{
#'     \item \code{"auto"} (default): use \code{"train"} if \code{P_all} is available,
#'           otherwise fall back to \code{"test_ls"};
#'     \item \code{"train"}: use training loadings \code{P_all} (strict application of the
#'           trained model; recommended when available);
#'     \item \code{"test_ls"}: compute least-squares loadings on the current test residual
#'           (\eqn{p = X^\top t / (t^\top t)}), yielding a "weights-only" EV diagnostic.
#'   }
#' @param clamp_ev \code{character(1)} How to clamp explained variance ratios:
#'   \code{"none"} (default; preserves negative out-of-sample EV as a diagnostic),
#'   \code{"zero"} (set negative EV to 0), or \code{"zero_one"} (clamp to \[0,1\]).
#'
#' @return
#' A \code{list} with:
#' \describe{
#'   \item{\code{ev_block}}{Incremental explained variance per component and block
#'     (\code{K × B}), computed as SS reduction divided by original test SS.}
#'   \item{\code{ev_comp}}{Incremental explained variance per component across all
#'     blocks (length \code{K}); SS-weighted across blocks.}
#'   \item{\code{ev_block_cum}}{Cumulative explained variance per component and block
#'     (\code{K × B}), accumulated across components (most meaningful when
#'     \code{deflate=TRUE}).}
#'   \item{\code{ev_comp_cum}}{Cumulative explained variance across all blocks
#'     (length \code{K}); SS-weighted.}
#'   \item{\code{mac_comp}}{Latent correlation per component (length \code{K}):
#'     MAC or Frobenius, matching the training objective convention.}
#'   \item{\code{valid_block}}{Logical matrix (\code{K × B}) indicating which blocks
#'     had valid (non-degenerate) scores for the objective at each component.}
#'   \item{\code{T_mat}}{Test score matrix of dimension \code{n_test × (K·B)} with
#'     columns ordered \code{LV1_<block1>, …, LV1_<blockB>, LV2_<block1>, …}.}
#' }
#'
#' @keywords internal
#' @export
compute_test_ev = function(
  X_blocks_test,
  W_all,
  P_all = NULL,
  deflate = TRUE,
  performance_metric = c("mac", "frobenius"),
  correlation_method = c("pearson", "spearman"),
  eps_var = 1e-12,
  loading_source = c("auto", "train", "test_ls"),
  clamp_ev = c("none", "zero", "zero_one")
) {
  performance_metric = match.arg(performance_metric)
  correlation_method = match.arg(correlation_method)
  loading_source = match.arg(loading_source)
  clamp_ev = match.arg(clamp_ev)

  # Resolve loading source automatically if requested
  if (identical(loading_source, "auto")) {
    loading_source = if (!is.null(P_all) && length(P_all) > 0L) "train" else "test_ls"
  }
  if (identical(loading_source, "train") && (is.null(P_all) || length(P_all) == 0L)) {
    stop("loading_source='train' requested but P_all is NULL/empty.")
  }

  stopifnot(is.list(X_blocks_test), length(X_blocks_test) >= 1L)
  stopifnot(is.list(W_all), length(W_all) >= 1L)
  stopifnot(is.logical(deflate), length(deflate) == 1L)

  B = length(X_blocks_test)
  K = length(W_all)
  n_test = nrow(X_blocks_test[[1]])
  stopifnot(all(vapply(X_blocks_test, nrow, integer(1)) == n_test))

  block_names = names(X_blocks_test)
  if (is.null(block_names) || any(!nzchar(block_names))) {
    block_names = paste0("block", seq_len(B))
  }

  comp_names = names(W_all)
  if (is.null(comp_names) || any(!nzchar(comp_names))) {
    comp_names = sprintf("LC_%02d", seq_len(K))
  }

  # If no train loadings, default to test LS loadings (weights-only EV)
  if (is.null(P_all) || length(P_all) == 0L) {
    loading_source = "test_ls"
  } else {
    loading_source = match.arg(loading_source)
  }
  if (identical(loading_source, "train") && (is.null(P_all) || length(P_all) == 0L)) {
    stop("loading_source='train' requested but P_all is NULL/empty.")
  }

  align_vec = function(v, cols, p) {
    if (is.null(v)) {
      return(rep(0, p))
    }
    if (!is.null(names(v)) && !is.null(cols)) {
      out = as.numeric(v[cols])
      out[is.na(out)] = 0
      return(out)
    }
    out = as.numeric(v)
    if (length(out) != p) {
      return(rep(0, p))
    }
    out
  }

  # Align weights/loadings once in R (handles names), compute heavy loops in C++
  W_aligned = vector("list", K)
  P_aligned = vector("list", K)
  for (k in seq_len(K)) {
    Wk = W_all[[k]]
    Pk = if (!is.null(P_all) && length(P_all) >= k) P_all[[k]] else NULL
    W_aligned[[k]] = lapply(seq_len(B), function(b) {
      Xb = X_blocks_test[[b]]
      align_vec(if (is.list(Wk)) Wk[[b]] else NULL, colnames(Xb), ncol(Xb))
    })
    P_aligned[[k]] = lapply(seq_len(B), function(b) {
      Xb = X_blocks_test[[b]]
      align_vec(if (is.list(Pk)) Pk[[b]] else NULL, colnames(Xb), ncol(Xb))
    })
  }

  clamp_mode = switch(clamp_ev,
    none = 0L,
    zero = 1L,
    zero_one = 2L
  )

  res = cpp_compute_test_ev_core(
    X_blocks_test = X_blocks_test,
    W_all = W_aligned,
    P_all = P_aligned,
    deflate = isTRUE(deflate),
    spearman = identical(correlation_method, "spearman"),
    frobenius = identical(performance_metric, "frobenius"),
    eps_var = eps_var,
    use_train_loadings = identical(loading_source, "train"),
    clamp_mode = clamp_mode
  )

  ev_block_inc = as.matrix(res$ev_block)
  ev_block_cum = as.matrix(res$ev_block_cum)
  ev_comp_inc = as.numeric(res$ev_comp)
  ev_comp_cum = as.numeric(res$ev_comp_cum)
  mac_comp = as.numeric(res$mac_comp)
  valid_block = as.matrix(res$valid_block)
  T_mat = as.matrix(res$T_mat)

  dimnames(ev_block_inc) = list(comp_names, block_names)
  dimnames(ev_block_cum) = list(comp_names, block_names)
  dimnames(valid_block) = list(comp_names, block_names)
  names(ev_comp_inc) = comp_names
  names(ev_comp_cum) = comp_names
  names(mac_comp) = comp_names
  colnames(T_mat) = as.vector(vapply(
    seq_len(K),
    function(k) paste0("LV", k, "_", block_names),
    character(B)
  ))

  list(
    ev_block      = ev_block_inc,
    ev_comp       = ev_comp_inc,
    ev_block_cum  = ev_block_cum,
    ev_comp_cum   = ev_comp_cum,
    mac_comp      = mac_comp,
    valid_block   = valid_block,
    T_mat         = T_mat
  )
}

#' Compute prediction-side EV and latent correlation from a PipeOpMBsPLS state
#'
#' @description
#' Convenience wrapper around \code{\link{compute_test_ev}} used during prediction
#' in \code{PipeOpMBsPLS}. It extracts trained weights/loadings and configuration
#' from a PipeOp state list and computes prediction-side explained variance and
#' latent correlation on a list of test blocks.
#'
#' @details
#' The wrapper always uses \code{deflate = TRUE} to mirror the sequential
#' (component-wise) behavior used during training. The correlation method for
#' MAC/Frobenius is taken from \code{state$correlation_method} when available;
#' otherwise it defaults to \code{"pearson"}.
#'
#' If the state does not contain loadings (\code{state$loadings} is \code{NULL} or
#' empty), \code{\link{compute_test_ev}} will automatically fall back to
#' \code{loading_source = "test_ls"} (least-squares loadings on test residuals).
#'
#' @param X_blocks_test \code{list} of numeric matrices; test data blocks
#'   (\code{n_test × p_b}). All blocks must have the same number of rows.
#' @param state \code{list}. Trained PipeOp state. Must contain at least:
#'   \itemize{
#'     \item \code{weights}: component-wise block weights (as in training),
#'     \item \code{loadings}: (optional) component-wise block loadings,
#'     \item \code{performance_metric}: \code{"mac"} or \code{"frobenius"},
#'     \item \code{correlation_method}: (optional) \code{"pearson"} or \code{"spearman"}.
#'   }
#'
#' @return
#' A list of prediction-side metrics as returned by \code{\link{compute_test_ev}},
#' including incremental and cumulative EV, latent correlation per component, a
#' validity mask, and the test score matrix.
#'
#' @keywords internal
#' @export
compute_pipeop_test_ev = function(X_blocks_test, state) {
  compute_test_ev(
    X_blocks_test      = X_blocks_test,
    W_all              = state$weights,
    P_all              = state$loadings,
    deflate            = TRUE,
    performance_metric = state$performance_metric %||% "mac",
    correlation_method = state$correlation_method %||% "pearson",
    loading_source     = "auto",
    clamp_ev           = "none",
    eps_var            = 1e-12
  )
}
