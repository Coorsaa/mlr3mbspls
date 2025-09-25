#' Flip signs of MB-sPLS weights/loadings/scores
#'
#' @param x Object containing MB-sPLS state/model
#' @param signs Controls what to flip. One of:
#'   - scalar +1/-1 (applied to ALL components & blocks; -1 means "flip all"),
#'   - numeric vector length K (per component; replicated across blocks),
#'   - numeric matrix K x B with rownames = components (e.g., "LC_01") and
#'     colnames = block names; entries must be +1/-1.
#' @param ... Additional arguments passed to methods
#' @return The modified object (invisibly)
#' @export
mbspls_flip_weights = function(x, signs = -1L, ...) {
  UseMethod("mbspls_flip_weights")
}

#' @rdname mbspls_flip_weights
#' @param flip_boot Also flip bootstrap summaries/draws if present (default TRUE)
#' @param flip_T Also flip training T_mat columns (default TRUE)
#' @export
mbspls_flip_weights.list = function(x, signs = -1L, flip_boot = TRUE, flip_T = TRUE, ...) {
  # This is the original mbspls_flip_state logic
  stopifnot(is.list(x), length(x$weights) == length(x$loadings))
  K = as.integer(x$ncomp %||% length(x$weights))
  bn = names(x$blocks)
  B = length(bn)
  comp_names = sprintf("LC_%02d", seq_len(K))

  # -- resolve signs -> K x B matrix with dimnames (components x blocks)
  S = (function() {
    # default: flip all
    if (is.null(signs)) signs <- -1L
    if (length(signs) == 1L && !is.matrix(signs)) {
      matrix(as.integer(signs), nrow = K, ncol = B,
        dimnames = list(comp_names, bn))
    } else if (is.vector(signs) && !is.matrix(signs) && length(signs) == K) {
      matrix(as.integer(signs), nrow = K, ncol = B,
        byrow = FALSE, dimnames = list(comp_names, bn))
    } else if (is.matrix(signs)) {
      S0 = signs
      # Try to align by dimnames if present
      if (!is.null(rownames(S0)) && !is.null(colnames(S0))) {
        S0 = S0[comp_names, bn, drop = FALSE]
      } else {
        # accept KxB or BxK; guess orientation by matching one dimension
        if (nrow(S0) == K && ncol(S0) == B) {
          # as is
        } else if (nrow(S0) == B && ncol(S0) == K) {
          S0 = t(S0)
        } else {
          stop("signs matrix must be KxB or BxK")
        }
        dimnames(S0) = list(comp_names, bn)
      }
      storage.mode(S0) = "integer"
      S0
    } else {
      stop("Unsupported 'signs' specification")
    }
  })()

  # helper: swap lower/upper if needed after a sign flip
  .fix_bounds = function(lo, hi) {
    lo2 = pmin(lo, hi, na.rm = TRUE)
    hi2 = pmax(lo, hi, na.rm = TRUE)
    list(lo2, hi2)
  }

  # --- flip weights & loadings (and optional training scores) ---
  for (k in seq_len(K)) {
    for (b in seq_len(B)) {
      s = S[k, b]
      if (is.na(s) || s == 1L) next
      blk = bn[b]
      # weights / loadings are named numeric vectors
      x$weights[[k]][[blk]] = s * x$weights[[k]][[blk]]
      x$loadings[[k]][[blk]] = s * x$loadings[[k]][[blk]]

      if (isTRUE(flip_T) && !is.null(x$T_mat)) {
        col = paste0("LV", k, "_", blk)
        if (col %in% colnames(x$T_mat)) {
          x$T_mat[, col] = s * x$T_mat[, col]
        }
      }
    }
  }

  # --- optional: flip bootstrap artifacts coherently ----------------
  if (isTRUE(flip_boot)) {
    # 1) stability-filtered weights (if present)
    if (!is.null(x$weights_stability_filtered)) {
      for (k in seq_len(K)) {
        for (b in seq_len(B)) {
          s = S[k, b]
          if (is.na(s) || s == 1L) next
          blk = bn[b]
          if (!is.null(x$weights_stability_filtered[[k]][[blk]])) {
            x$weights_stability_filtered[[k]][[blk]] =
              s * x$weights_stability_filtered[[k]][[blk]]
          }
        }
      }
    }

    # 2) bootstrap summary table (if present)
    if (!is.null(x$weights_ci)) {
      df = x$weights_ci
      # accept both data.frame and data.table
      comp_col = "component"
      block_col = "block"
      need_cols = intersect(c("boot_mean", "ci_lower", "ci_upper", "ci_lower_nz", "ci_upper_nz"),
        names(df))
      for (k in seq_len(K)) {
        for (b in seq_len(B)) {
          s = S[k, b]
          if (is.na(s) || s == 1L) next
          sel = (as.character(df[[comp_col]]) == comp_names[k]) & (as.character(df[[block_col]]) == bn[b])
          if (any(sel)) {
            for (cc in need_cols) df[[cc]][sel] = s * df[[cc]][sel]
            if (all(c("ci_lower", "ci_upper") %in% names(df))) {
              ii = which(sel)
              if (length(ii)) {
                z = .fix_bounds(df$ci_lower[ii], df$ci_upper[ii])
                df$ci_lower[ii] = z[[1]]
                df$ci_upper[ii] = z[[2]]
              }
            }
            if (all(c("ci_lower_nz", "ci_upper_nz") %in% names(df))) {
              ii = which(sel & is.finite(df$ci_lower_nz) & is.finite(df$ci_upper_nz))
              if (length(ii)) {
                z = .fix_bounds(df$ci_lower_nz[ii], df$ci_upper_nz[ii])
                df$ci_lower_nz[ii] = z[[1]]
                df$ci_upper_nz[ii] = z[[2]]
              }
            }
          }
        }
      }
      x$weights_ci = df
    }

    # 3) bootstrap draws (if present)
    if (!is.null(x$weights_boot_draws)) {
      dr = x$weights_boot_draws
      # component can be like "LC_01"; align safely
      comp_is_factor = is.factor(dr$component)
      comp_val = as.character(dr$component)
      for (k in seq_len(K)) {
        for (b in seq_len(B)) {
          s = S[k, b]
          if (is.na(s) || s == 1L) next
          sel = (comp_val == comp_names[k]) & (as.character(dr$block) == bn[b])
          if (any(sel)) dr$weight[sel] <- s * dr$weight[sel]
        }
      }
      # keep original classing
      if (comp_is_factor) dr$component <- factor(dr$component, levels = levels(dr$component))
      x$weights_boot_draws = dr
    }

    # 4) stored bootstrap vectors (if present)
    if (!is.null(x$weights_boot_vectors)) {
      for (k in seq_len(K)) {
        for (b in seq_len(B)) {
          s = S[k, b]
          if (is.na(s) || s == 1L) next
          blk_map = x$weights_boot_vectors[[k]][[bn[b]]]
          if (is.list(blk_map) && length(blk_map)) {
            x$weights_boot_vectors[[k]][[bn[b]]] =
              lapply(blk_map, function(v) if (is.numeric(v)) s * v else v)
          }
        }
      }
    }
  }

  invisible(x)
}

#' @rdname mbspls_flip_weights
#' @param inplace Modify object in-place (TRUE) or return flipped state (FALSE)
#' @export
mbspls_flip_weights.PipeOpMBsPLS = function(x, signs = -1L, inplace = TRUE, ...) {
  if (is.null(x$state)) stop("PipeOp has no $state (not trained?)")
  mbspls_flip_weights(x$state, signs = signs, ...)
  if (isTRUE(inplace)) invisible(x) else return(x$state)
}

#' @rdname mbspls_flip_weights
#' @export
mbspls_flip_weights.GraphLearner = function(x, signs = -1L, inplace = TRUE, ...) {
  po = .mbspls_find_po(x)
  # prefer trained state if available; otherwise flip the pipeop's state
  if (!is.null(po$state)) {
    mbspls_flip_weights(po$state, signs = signs, ...)
  } else {
    warning("PipeOpMBsPLS has no state yet; nothing flipped.")
  }
  if (isTRUE(inplace)) invisible(x) else return(po$state)
}

#' Find first MB-sPLS pipeop in a GraphLearner
.mbspls_find_po = function(gl) {
  pops = gl$graph$pipeops
  idx = vapply(pops, function(p) inherits(p, "PipeOpMBsPLS"), logical(1))
  if (!any(idx)) stop("No PipeOpMBsPLS found in gl$graph")
  pops[[which(idx)[1L]]]
}

# Keep the old functions as wrappers for backward compatibility
mbspls_flip_state = function(state, signs = -1L, flip_boot = TRUE, flip_T = TRUE) {
  mbspls_flip_weights(state, signs = signs, flip_boot = flip_boot, flip_T = flip_T)
}

mbspls_flip_pipeop = function(po, signs = -1L, inplace = TRUE, ...) {
  mbspls_flip_weights(po, signs = signs, inplace = inplace, ...)
}

mbspls_flip_graphlearner = function(gl, signs = -1L, inplace = TRUE, ...) {
  mbspls_flip_weights(gl, signs = signs, inplace = inplace, ...)
}

mbspls_flip_model_path = function(obj, signs = -1L, ...) {
  # If obj is the 'mbspls' node under gl$model, try $state first, then $model.
  st = NULL
  if (!is.null(obj$state)) st <- obj$state else if (!is.null(obj$model)) st <- obj$model
  if (is.null(st)) stop("No $state or $model found on supplied object.")
  mbspls_flip_weights(st, signs = signs, ...)
  invisible(obj)
}
