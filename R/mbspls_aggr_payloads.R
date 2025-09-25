#' @title Aggregate MB-sPLS test metrics across outer CV folds
#'
#' @description
#' Collapses the per-fold payloads written by `PipeOpMBsPLS` (via `log_env$last`)
#' into weighted cross-fold summaries. Supports component-wise aggregation of:
#' - latent correlation (MAC or Frobenius), with mean/SD,
#' - explained variance (per-block and total),
#' - optional validation p-values (combined via Stouffer or Fisher).
#'
#' @param payloads list of payloads (one per outer fold), each as produced in
#'   `PipeOpMBsPLS$predict()` with fields `mac_comp`, `ev_block`, `ev_comp`,
#'   `T_mat`, `blocks`, and optionally `val_test_p`, `val_test_stat`.
#' @param weight_by character(1). Weights for cross-fold means:
#'   `"sqrt_n"` (default; weight = √n_test), `"n"` (weight = n_test), or `"equal"`.
#' @param p_method character(1). How to combine validation p-values across folds:
#'   `"stouffer"` (default), `"fisher"`, or `"none"`.
#' @param enforce_monotone logical(1). If TRUE, makes combined p-values
#'   nondecreasing across components (useful for sequential stopping). Default FALSE.
#'
#' @return list with:
#' \itemize{
#'   \item \code{summary}: list of matrices/vectors
#'     (\code{mac_mean}, \code{mac_sd}, \code{ev_block_mean}, \code{ev_comp_mean},
#'      \code{p_combined}, \code{perf_metric}, \code{blocks})
#'   \item \code{fold_table}: long data.table (fold × component × block)
#'     with per-fold metrics and n_test
#' }
#' @examples
#' agg = aggregate_mbspls_payloads(payloads)
#'
#' @export
aggregate_mbspls_payloads = function(
  payloads,
  weight_by = c("sqrt_n", "n", "equal"),
  p_method = c("stouffer", "fisher", "none"),
  enforce_monotone = FALSE
) {
  stopifnot(is.list(payloads), length(payloads) > 0L)
  weight_by = match.arg(weight_by)
  p_method = match.arg(p_method)

  # --- helpers ---------------------------------------------------------------
  n_from_payload = function(pl) {
    if (!is.null(pl$T_mat)) nrow(pl$T_mat) else NA_integer_
  }
  w_from_n = function(n) {
    if (is.na(n)) {
      return(1)
    }
    switch(weight_by,
      "equal"  = 1,
      "n"      = n,
      "sqrt_n" = sqrt(n)
    )
  }
  stouffer_combine = function(p, w = NULL) {
    p = p[is.finite(p) & p >= 0 & p <= 1]
    if (!length(p)) {
      return(NA_real_)
    }
    z = stats::qnorm(1 - p) # one-sided mapping
    if (is.null(w)) w <- rep(1, length(z))
    zc = sum(w * z) / sqrt(sum(w^2))
    1 - stats::pnorm(zc)
  }
  fisher_combine = function(p) {
    p = p[is.finite(p) & p > 0 & p <= 1]
    if (!length(p)) {
      return(NA_real_)
    }
    X = -2 * sum(log(p))
    stats::pchisq(X, df = 2 * length(p), lower.tail = FALSE)
  }
  pad_len = function(x, L) {
    c(x, rep(NA_real_, max(0L, L - length(x))))
  }

  # --- discover dimensions/labels -------------------------------------------
  K_max = max(vapply(payloads, function(pl) length(pl$mac_comp %||% NA), 1L), na.rm = TRUE)
  block_union = unique(unlist(lapply(payloads, function(pl) pl$blocks %||% character())))
  B_all = length(block_union)
  comp_names = sprintf("LC_%02d", seq_len(K_max))

  # --- per-fold tidy table ---------------------------------------------------
  rows = list()
  for (i in seq_along(payloads)) {
    pl = payloads[[i]]
    if (is.null(pl)) next
    n_test = n_from_payload(pl)
    w_i = w_from_n(n_test)
    perf = pl$perf_metric %||% "mac"

    # pad component-wise vectors
    mac_i = pad_len(as.numeric(pl$mac_comp %||% numeric()), K_max)
    evc_i = pad_len(as.numeric(pl$ev_comp %||% numeric()), K_max)
    pv_i = pad_len(as.numeric(pl$val_test_p %||% rep(NA_real_, length(pl$mac_comp %||% 0))), K_max)

    # map ev_block to full block union
    evb_i = matrix(NA_real_, K_max, B_all,
      dimnames = list(comp_names, block_union))
    if (!is.null(pl$ev_block)) {
      Eb = as.matrix(pl$ev_block)
      # try assign by column name; fallback to order if unnamed
      if (!is.null(colnames(Eb))) {
        inter = intersect(colnames(Eb), block_union)
        evb_i[seq_len(nrow(Eb)), inter] = Eb[, inter, drop = FALSE]
      } else {
        takeB = min(ncol(Eb), B_all)
        evb_i[seq_len(nrow(Eb)), seq_len(takeB)] = Eb[, seq_len(takeB), drop = FALSE]
      }
    }

    # build long rows
    for (k in seq_len(K_max)) {
      rows[[length(rows) + 1L]] = data.table::data.table(
        fold = i,
        component = comp_names[k],
        perf_metric = perf,
        n_test = n_test,
        weight = w_i,
        mac = mac_i[k],
        ev_total = evc_i[k],
        p_val = pv_i[k],
        block = NA_character_,
        ev_block = NA_real_
      )
      # blockwise
      for (b in block_union) {
        rows[[length(rows) + 1L]] = data.table::data.table(
          fold = i,
          component = comp_names[k],
          perf_metric = perf,
          n_test = n_test,
          weight = w_i,
          mac = mac_i[k],
          ev_total = evc_i[k],
          p_val = pv_i[k],
          block = b,
          ev_block = evb_i[k, b]
        )
      }
    }
  }
  fold_table = data.table::rbindlist(rows, use.names = TRUE, fill = TRUE)

  # infer performance metric (assume constant)
  perf_metric = fold_table$perf_metric[which(!is.na(fold_table$perf_metric))[1]] %||% "mac"

  # --- aggregate (weighted) --------------------------------------------------
  # component-wise MAC
  mac_dt = fold_table[!is.na(mac),
    .(w = weight, mac = mac),
    by = .(component, fold)]
  mac_mean = mac_dt[, .(
    mean = sum(w * mac, na.rm = TRUE) / sum(w, na.rm = TRUE),
    sd   = stats::sd(mac, na.rm = TRUE)
  ), by = component]
  mac_mean = mac_mean[match(comp_names, component)]
  mac_mean_vec = setNames(mac_mean$mean, comp_names)
  mac_sd_vec = setNames(mac_mean$sd, comp_names)

  # component-wise total EV
  evc_dt = fold_table[!is.na(ev_total),
    .(w = weight, ev = ev_total),
    by = .(component, fold)]
  evc_mean = evc_dt[, .(
    mean = sum(w * ev, na.rm = TRUE) / sum(w, na.rm = TRUE)
  ), by = component]
  evc_mean = evc_mean[match(comp_names, component)]
  evc_mean_vec = setNames(evc_mean$mean, comp_names)

  # block-wise EV (component × block)
  evb_dt = fold_table[!is.na(block) & !is.na(ev_block),
    .(w = weight, evb = ev_block),
    by = .(component, block, fold)]
  evb_mean = evb_dt[, .(
    mean = sum(w * evb, na.rm = TRUE) / sum(w, na.rm = TRUE)
  ), by = .(component, block)]
  # reassemble matrix
  ev_block_mean = matrix(NA_real_, nrow = K_max, ncol = B_all,
    dimnames = list(comp_names, block_union))
  if (nrow(evb_mean)) {
    for (r in seq_len(nrow(evb_mean))) {
      ev_block_mean[evb_mean$component[r], evb_mean$block[r]] = evb_mean$mean[r]
    }
  }

  # combine p-values per component (optional)
  p_combined = rep(NA_real_, K_max)
  names(p_combined) = comp_names
  if (p_method != "none" && "p_val" %in% names(fold_table)) {
    for (k in seq_len(K_max)) {
      rows_k = fold_table[component == comp_names[k] & is.finite(p_val)]
      if (nrow(rows_k)) {
        if (p_method == "stouffer") {
          ww = rows_k$weight
          pp = rows_k$p_val
          p_combined[k] = stouffer_combine(pp, w = ww)
        } else if (p_method == "fisher") {
          p_combined[k] = fisher_combine(rows_k$p_val)
        }
      }
    }
    if (isTRUE(enforce_monotone)) {
      # make p-values nondecreasing across components (sequential testing)
      for (k in 2:K_max) {
        if (is.finite(p_combined[k - 1]) && is.finite(p_combined[k])) {
          p_combined[k] = max(p_combined[k], p_combined[k - 1])
        }
      }
    }
  }

  list(
    summary = list(
      mac_mean      = mac_mean_vec,
      mac_sd        = mac_sd_vec,
      ev_comp_mean  = evc_mean_vec,
      ev_block_mean = ev_block_mean,
      p_combined    = p_combined,
      perf_metric   = perf_metric,
      blocks        = block_union
    ),
    fold_table = fold_table[]
  )
}
