# Statistical-validity utilities shared by the package's inference,
# preprocessing, and resampling paths. These deliberately use base R only.

.mb_assert_scalar_integer = function(x, name, lower = 1L) {
  if (length(x) != 1L || is.na(x) || !is.numeric(x) || !is.finite(x) ||
    x != floor(x) || x < lower || x > .Machine$integer.max) {
    stop(sprintf(
      "`%s` must be one finite integer between %d and %d.",
      name, lower, .Machine$integer.max
    ), call. = FALSE)
  }
  as.integer(x)
}

.mb_assert_probability = function(x, name = "conf") {
  if (length(x) != 1L || is.na(x) || !is.numeric(x) || !is.finite(x) ||
    x <= 0 || x >= 1) {
    stop(sprintf(
      "`%s` must be one finite number strictly between 0 and 1.", name
    ), call. = FALSE)
  }
  as.numeric(x)
}

.mb_numeric_matrix = function(x, name = deparse(substitute(x))) {
  if (is.data.frame(x)) {
    bad = !vapply(x, is.numeric, logical(1L))
    if (any(bad)) {
      stop(sprintf(
        "`%s` contains non-numeric columns: %s.",
        name, paste(names(x)[bad], collapse = ", ")
      ), call. = FALSE)
    }
    x = as.matrix(x)
  }
  if (!is.matrix(x) || !is.numeric(x)) {
    stop(sprintf(
      "`%s` must be a numeric matrix or all-numeric data frame.", name
    ), call. = FALSE)
  }
  if (!length(x) || nrow(x) < 1L || ncol(x) < 1L) {
    stop(sprintf(
      "`%s` must have at least one row and one column.", name
    ), call. = FALSE)
  }
  if (anyNA(x) || any(!is.finite(x))) {
    stop(sprintf(
      "`%s` must contain only finite, non-missing values.", name
    ), call. = FALSE)
  }
  x
}

.mb_validate_column_names = function(x, context) {
  nms = colnames(x)
  if (is.null(nms) || anyNA(nms) || any(!nzchar(nms)) || anyDuplicated(nms)) {
    stop(sprintf(
      "%s predictors must have unique, non-empty column names.", context
    ), call. = FALSE)
  }
  nms
}

.mb_assert_component_rank = function(blocks, ncomp, context) {
  ncomp = .mb_assert_scalar_integer(ncomp, "ncomp")
  ranks = vapply(blocks, function(block) {
    qr(block)$rank
  }, integer(1L))
  if (any(ranks < ncomp)) {
    limiting = names(ranks)[ranks < ncomp]
    stop(sprintf(
      paste0(
        "%s requested %d components, but effective block rank is lower ",
        "for: %s."
      ),
      context,
      ncomp,
      paste(sprintf("%s (%d)", limiting, ranks[limiting]), collapse = ", ")
    ), call. = FALSE)
  }
  invisible(ranks)
}

#' Fit and apply a frozen predictor scaler
#'
#' `mb_fit_scaler()` estimates column means and sample standard deviations from
#' training data only. `mb_apply_scaler()` reuses that state after checking the
#' complete predictor schema and restoring the training column order. Missing,
#' non-finite, duplicate, and numerically constant predictors are rejected.
#'
#' These helpers are useful outside an `mlr3` graph. Within a graph,
#' [PipeOpBlockScaling] provides the corresponding train/predict separation.
#'
#' @param x A numeric matrix or all-numeric data frame.
#' @param scaler An object returned by `mb_fit_scaler()`.
#' @param tolerance Positive numeric threshold below which a training standard
#'   deviation is treated as numerically zero.
#'
#' @return `mb_fit_scaler()` returns an `mbspls_scaler` object.
#'   `mb_apply_scaler()` returns a numeric matrix in training-column order.
#' @export
mb_fit_scaler = function(x, tolerance = sqrt(.Machine$double.eps)) {
  x = .mb_numeric_matrix(x, "x")
  columns = .mb_validate_column_names(x, "Training")
  if (nrow(x) < 2L) {
    stop(
      "At least two training observations are required to estimate scale.",
      call. = FALSE
    )
  }
  if (length(tolerance) != 1L || !is.numeric(tolerance) ||
    !is.finite(tolerance) || tolerance <= 0) {
    stop("`tolerance` must be one finite positive number.", call. = FALSE)
  }

  center = colMeans(x)
  scale = apply(x, 2L, stats::sd)
  bad = !is.finite(scale) | scale <= tolerance
  if (any(bad)) {
    stop(sprintf(
      "Zero-variance or numerically constant training predictors: %s.",
      paste(columns[bad], collapse = ", ")
    ), call. = FALSE)
  }

  structure(
    list(
      center = stats::setNames(as.numeric(center), columns),
      scale = stats::setNames(as.numeric(scale), columns),
      columns = columns,
      tolerance = as.numeric(tolerance)
    ),
    class = "mbspls_scaler"
  )
}

#' @rdname mb_fit_scaler
#' @export
mb_apply_scaler = function(x, scaler) {
  if (!inherits(scaler, "mbspls_scaler") ||
    !is.numeric(scaler$center) || !is.numeric(scaler$scale) ||
    !is.character(scaler$columns)) {
    stop("`scaler` must be a valid object returned by `mb_fit_scaler()`.",
      call. = FALSE)
  }
  if (!length(scaler$columns) || anyNA(scaler$columns) ||
    any(!nzchar(scaler$columns)) || anyDuplicated(scaler$columns) ||
    length(scaler$center) != length(scaler$columns) ||
    length(scaler$scale) != length(scaler$columns) ||
    !identical(names(scaler$center), scaler$columns) ||
    !identical(names(scaler$scale), scaler$columns) ||
    anyNA(scaler$center) || any(!is.finite(scaler$center)) ||
    anyNA(scaler$scale) || any(!is.finite(scaler$scale)) ||
    any(scaler$scale <= 0)) {
    stop("`scaler` contains an invalid fitted predictor schema or scale state.",
      call. = FALSE)
  }

  x = .mb_numeric_matrix(x, "x")
  columns = .mb_validate_column_names(x, "Prediction")
  missing = setdiff(scaler$columns, columns)
  extra = setdiff(columns, scaler$columns)
  if (length(missing) || length(extra)) {
    pieces = c(
      if (length(missing)) {
        sprintf(
          "missing: %s", paste(missing, collapse = ", ")
        )
      },
      if (length(extra)) {
        sprintf(
          "unexpected: %s", paste(extra, collapse = ", ")
        )
      }
    )
    stop(sprintf(
      "Prediction schema differs from training schema (%s).",
      paste(pieces, collapse = "; ")
    ), call. = FALSE)
  }

  x = x[, scaler$columns, drop = FALSE]
  sweep(sweep(x, 2L, scaler$center, FUN = "-"),
    2L, scaler$scale, FUN = "/")
}

#' Create deterministic independent RNG streams
#'
#' Creates one L'Ecuyer-CMRG stream per resample while restoring the caller's
#' RNG kind and state. Assign stream `i` to resample `i` to make results
#' independent of worker scheduling. Streams use the Inversion normal
#' generator and Rejection sampler, independently of the caller's RNG kinds.
#' As with other seed-based restoration in R, a cached Box-Muller normal
#' variate in the calling session cannot be restored from `.Random.seed`.
#'
#' @param n Number of streams.
#' @param seed Non-negative integer seed.
#'
#' @return A list of `.Random.seed` vectors, one per resample.
#' @export
mb_rng_streams = function(n, seed) {
  n = .mb_assert_scalar_integer(n, "n")
  seed = .mb_assert_scalar_integer(seed, "seed", lower = 0L)
  old_kind = RNGkind()
  old_seed = if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit(.mb_restore_rng_state(old_kind, old_seed), add = TRUE)

  RNGkind("L'Ecuyer-CMRG", "Inversion", "Rejection")
  set.seed(seed)
  streams = vector("list", n)
  streams[[1L]] = get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (n > 1L) {
    for (i in 2L:n) {
      streams[[i]] = parallel::nextRNGStream(streams[[i - 1L]])
    }
  }
  streams
}

with_rng_stream_local = function(stream, fn) {
  if (!is.function(fn)) {
    stop("`fn` must be a function.", call. = FALSE)
  }
  # The last two digits of `.Random.seed[1]` encode the generator and the next
  # two the normal generator. Box-Muller (2) and user-supplied (3) generators
  # keep state outside the seed. Higher digits encode further kinds (sample,
  # and in newer R versions binomial) and are deliberately not constrained.
  if (!is.integer(stream) || length(stream) != 7L || is.na(stream[[1L]]) ||
    stream[[1L]] < 0L || stream[[1L]] %% 100L != 7L ||
    stream[[1L]] %/% 100L %% 100L %in% c(2L, 3L)) {
    stop(paste(
      "`stream` must be a valid L'Ecuyer-CMRG `.Random.seed` vector",
      "with a normal generator whose state is stored in the seed."
    ),
    call. = FALSE)
  }
  # R stores the six unsigned 32-bit state words as signed integers. In
  # particular NA_integer_ represents the valid unsigned state word 2^31.
  state = as.double(stream[-1L])
  state[is.na(state)] = 2^31
  state[state < 0] = state[state < 0] + 2^32
  if (all(state[1:3] == 0) || all(state[4:6] == 0) ||
    any(state[1:3] >= 4294967087) || any(state[4:6] >= 4294944443)) {
    stop("`stream` contains an invalid L'Ecuyer-CMRG generator state.",
      call. = FALSE)
  }

  old_kind = RNGkind()
  old_seed = if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit(.mb_restore_rng_state(old_kind, old_seed), add = TRUE)

  RNGkind("L'Ecuyer-CMRG", "Inversion", "Rejection")
  assign(".Random.seed", stream, envir = .GlobalEnv)
  fn()
}

.mb_permutation_extreme = function(
  observed,
  null_distribution,
  alternative,
  null_center = NULL
) {
  if (identical(alternative, "two.sided")) {
    observed = abs(observed - null_center)
    null_distribution = abs(null_distribution - null_center)
    if (!is.finite(observed) || any(!is.finite(null_distribution))) {
      stop("Two-sided distances from `null_center` must remain finite.",
        call. = FALSE)
    }
  }
  extreme = if (identical(alternative, "less")) {
    null_distribution <= observed
  } else {
    null_distribution >= observed
  }
  # A theoretically tied statistic may differ after a permutation solely due
  # to floating-point summation order. Count such near ties conservatively.
  tolerance = 100 * .Machine$double.eps * abs(observed)
  extreme | abs(null_distribution - observed) <= tolerance
}

#' Calculate a sampled permutation p-value
#'
#' Uses inclusive ties and the finite Monte Carlo correction
#' `(b + 1) / (B + 1)`, so a sampled permutation p-value cannot be zero.
#' Numerical ties within `100 * .Machine$double.eps * abs(observed)` are
#' counted conservatively as equally extreme; for a two-sided test this
#' tolerance applies to the distance from `null_center`.
#' For a two-sided test, `null_center` is mandatory and extremeness is measured
#' by distance from that explicitly stated centre.
#'
#' This function only summarizes a supplied null distribution. Valid inference
#' additionally requires a design-valid exchangeability scheme and rerunning
#' the complete data-dependent analysis pipeline for every permutation.
#'
#' @param observed Observed scalar statistic.
#' @param null_distribution Finite statistics from `B` sampled permutations.
#' @param alternative One of `"greater"`, `"less"`, or `"two.sided"`.
#' @param null_center Explicit null centre required for a two-sided test.
#'
#' @return A scalar p-value in `(0, 1]`.
#' @export
mb_permutation_pvalue = function(
  observed,
  null_distribution,
  alternative = c("greater", "less", "two.sided"),
  null_center = NULL
) {
  alternative = match.arg(alternative)
  if (length(observed) != 1L || is.na(observed) || !is.numeric(observed) ||
    !is.finite(observed)) {
    stop("`observed` must be one finite numeric value.", call. = FALSE)
  }
  if (!is.numeric(null_distribution) || !is.null(dim(null_distribution)) ||
    !length(null_distribution) ||
    anyNA(null_distribution) || any(!is.finite(null_distribution))) {
    stop("`null_distribution` must be a non-empty finite numeric vector.",
      call. = FALSE)
  }
  if (identical(alternative, "two.sided")) {
    if (length(null_center) != 1L || is.na(null_center) ||
      !is.numeric(null_center) || !is.finite(null_center)) {
      stop(
        "`null_center` must be supplied as one finite number for a two-sided test.",
        call. = FALSE
      )
    }
  }

  extreme = .mb_permutation_extreme(
    observed, null_distribution, alternative, null_center
  )
  (sum(extreme) + 1) / (length(null_distribution) + 1)
}

#' Permute labels at the exchangeability-group level
#'
#' Each group must have one internally consistent label. Labels are shuffled
#' across groups and then expanded back to rows; inconsistent groups are
#' rejected instead of being silently permuted row by row.
#'
#' @param y Atomic outcome or treatment-label vector.
#' @param group Exchangeability-group identifier of the same length as `y`.
#' @param seed Optional non-negative integer seed. When supplied, the caller's
#'   RNG kind and state are restored.
#'
#' @return A vector with the same type and attributes as `y`.
#' @export
mb_permute_group_labels = function(y, group, seed = NULL) {
  if (!is.atomic(y) || is.matrix(y) || length(y) != length(group) || !length(y)) {
    stop("`y` and `group` must be non-empty vectors of equal length.",
      call. = FALSE)
  }
  if (anyNA(y) || anyNA(group)) {
    stop("`y` and `group` must not contain missing values.", call. = FALSE)
  }

  group_key = as.character(group)
  groups = unique(group_key)
  first_row = match(groups, group_key)
  inconsistent = .mb_varies_within_groups(y, group_key, groups, first_row)
  if (any(inconsistent)) {
    stop(sprintf(
      "Outcome labels vary within exchangeability groups: %s.",
      paste(groups[inconsistent], collapse = ", ")
    ), call. = FALSE)
  }
  group_labels = y[first_row]

  permute = function() {
    shuffled = group_labels[sample.int(length(groups), replace = FALSE)]
    out = y
    out[] = shuffled[match(group_key, groups)]
    attributes(out) = attributes(y)
    out
  }
  if (is.null(seed)) {
    permute()
  } else {
    seed = .mb_assert_scalar_integer(seed, "seed", lower = 0L)
    with_seed_local(seed, permute)
  }
}

#' Draw a cluster bootstrap sample
#'
#' Samples whole exchangeability groups with replacement. Optional strata must
#' be constant within groups and cause groups to be sampled within stratum.
#' Duplicate sampled groups receive distinct `bootstrap_cluster` identifiers.
#'
#' @param group Non-missing exchangeability-group identifier.
#' @param strata Optional non-missing stratum vector of the same length.
#' @param seed Optional non-negative integer seed.
#'
#' @return A list containing row `indices`, `sampled_groups`, and a
#'   `bootstrap_cluster` identifier for every selected row.
#' @export
mb_cluster_bootstrap = function(group, strata = NULL, seed = NULL) {
  design = .mb_cluster_design(group, strata)
  if (is.null(seed)) {
    .mb_draw_cluster(design)
  } else {
    seed = .mb_assert_scalar_integer(seed, "seed", lower = 0L)
    with_seed_local(seed, function() .mb_draw_cluster(design))
  }
}

# For each group (in `groups` order, with `first_row` its first row), whether
# `x` takes more than one value within the group. Linear in the number of rows.
.mb_varies_within_groups = function(x, group_key, groups, first_row) {
  differs = x != x[first_row[match(group_key, groups)]]
  groups %in% group_key[differs]
}

# Validate a cluster-bootstrap design once and precompute its row lists, so
# repeated draws cost O(n) instead of O(n x groups).
.mb_cluster_design = function(group, strata = NULL) {
  if (!length(group) || anyNA(group)) {
    stop("`group` must be non-empty and contain no missing values.",
      call. = FALSE)
  }
  if (!is.null(strata) &&
    (length(strata) != length(group) || anyNA(strata))) {
    stop("`strata` must be NULL or a non-missing vector as long as `group`.",
      call. = FALSE)
  }

  group_key = as.character(group)
  groups = unique(group_key)
  first_row = match(groups, group_key)
  group_strata = NULL
  if (!is.null(strata)) {
    inconsistent = .mb_varies_within_groups(strata, group_key, groups, first_row)
    if (any(inconsistent)) {
      stop(sprintf(
        "Strata vary within exchangeability groups: %s.",
        paste(groups[inconsistent], collapse = ", ")
      ), call. = FALSE)
    }
    group_strata = as.character(strata[first_row])
  }

  list(
    groups = groups,
    rows = unname(split(seq_along(group_key), factor(group_key, levels = groups))),
    group_strata = group_strata
  )
}

# Draw one cluster-bootstrap sample from a design built by .mb_cluster_design().
.mb_draw_cluster = function(design) {
  groups = design$groups
  group_strata = design$group_strata
  if (is.null(group_strata)) {
    sampled = sample(groups, length(groups), replace = TRUE)
  } else {
    strata_levels = unique(group_strata)
    sampled = unlist(lapply(strata_levels, function(level) {
      eligible = groups[group_strata == level]
      sample(eligible, length(eligible), replace = TRUE)
    }), use.names = FALSE)
    sampled = sampled[sample.int(length(sampled), replace = FALSE)]
  }

  selected = design$rows[match(sampled, groups)]
  list(
    indices = unlist(selected, use.names = FALSE),
    sampled_groups = sampled,
    bootstrap_cluster = rep.int(seq_along(selected), lengths(selected))
  )
}

#' Align PLS component signs to a reference
#'
#' PLS loadings and weights are sign-indeterminate. This helper flips each
#' estimate column according to the sign of its cosine similarity with the
#' matching reference column before element-wise aggregation. A column whose
#' absolute cosine does not exceed `tolerance`, or which has zero norm in
#' either matrix, is marked as ambiguous and left unchanged: its sign cannot be
#' determined reliably, and near-orthogonality usually means that the columns
#' do not describe the same component. Because the cosine is scale-free, the
#' flag does not depend on how the columns are normalised.
#'
#' The helper does not match components. Columns must already correspond, for
#' example after solving an assignment problem (e.g. the Hungarian algorithm) on
#' `1 - abs(cosine)` or on absolute score correlations; a pair of swapped
#' components typically shows up as ambiguous. For MB-sPLS, the reported
#' criterion depends only on absolute (or squared) cross-block correlations and
#' refits can differ in their relative block orientation, so signs must be
#' aligned block by block. Call the helper once per block on that block's rows,
#' never on weights stacked across blocks.
#'
#' @param estimate Numeric matrix with variables in rows and components in
#'   columns.
#' @param reference Numeric matrix with the identical schema.
#' @param tolerance Ambiguity threshold in `[0, 1)` for the absolute cosine
#'   similarity between matching columns.
#'
#' @return The aligned estimate with attributes `signs` (applied signs),
#'   `ambiguous` (logical flags), and `cosines` (cosine similarity of each
#'   estimate column with its reference column before alignment; `NA` for
#'   zero-norm columns).
#' @export
mb_align_component_signs = function(
  estimate,
  reference,
  tolerance = 0.2
) {
  estimate = .mb_numeric_matrix(estimate, "estimate")
  reference = .mb_numeric_matrix(reference, "reference")
  if (!identical(dim(estimate), dim(reference))) {
    stop("`estimate` and `reference` must have identical dimensions.",
      call. = FALSE)
  }
  if (!identical(dimnames(estimate), dimnames(reference))) {
    stop("`estimate` and `reference` must have identical dimnames.",
      call. = FALSE)
  }
  if (length(tolerance) != 1L || !is.numeric(tolerance) ||
    !is.finite(tolerance) || tolerance < 0 || tolerance >= 1) {
    stop("`tolerance` must be one finite number in [0, 1).", call. = FALSE)
  }

  # Divide each column by its largest absolute entry before squaring: the
  # cosine is unchanged, and extreme finite values cannot overflow.
  scale_columns = function(x) {
    peak = apply(abs(x), 2L, max)
    sweep(x, 2L, ifelse(peak > 0, peak, 1), FUN = "/")
  }
  unit_estimate = scale_columns(estimate)
  unit_reference = scale_columns(reference)
  norm_estimate = sqrt(colSums(unit_estimate^2))
  norm_reference = sqrt(colSums(unit_reference^2))
  cosines = colSums(unit_estimate * unit_reference) /
    (norm_estimate * norm_reference)
  cosines[!(norm_estimate > 0 & norm_reference > 0)] = NA_real_
  ambiguous = is.na(cosines) | abs(cosines) <= tolerance
  signs = ifelse(!ambiguous & cosines < 0, -1, 1)
  names(signs) = names(ambiguous) = names(cosines) = colnames(estimate)
  aligned = sweep(estimate, 2L, signs, FUN = "*")
  attr(aligned, "signs") = signs
  attr(aligned, "ambiguous") = ambiguous
  attr(aligned, "cosines") = cosines
  aligned
}

#' Summarize descriptive bootstrap uncertainty
#'
#' Reports an observed estimate, bootstrap mean and bias, standard error, and a
#' percentile, basic, or normal interval. Ordinary bootstrap replicates are not
#' a null distribution, so the returned `p_value` is deliberately `NA` and the
#' reason is recorded in `p_value_note`.
#'
#' Percentile and basic endpoints use Hyndman-Fan type-8 sample quantiles
#' (`stats::quantile(type = 8)`), as does the prediction-side bootstrap of
#' `PipeOpMBsPLS` (`val_test = "bootstrap"`). The normal interval is
#' `(estimate - bias) +/- z * standard_error`. The field names match the rows of
#' that prediction-side `val_bootstrap` payload.
#'
#' Non-finite replicates are counted as failed rather than silently treated as
#' valid. At least two finite replicates are required. Summaries then describe
#' the successful replicates only; selective fitting failures can invalidate
#' nominal interval coverage and must be investigated separately.
#'
#' @param replicates Numeric vector of bootstrap statistics.
#' @param observed Observed scalar estimate.
#' @param conf Confidence level strictly between zero and one.
#' @param type Interval type: `"percentile"`, `"basic"`, or `"normal"`.
#'
#' @return An `mbspls_bootstrap_summary` list with elements `estimate`,
#'   `bootstrap_mean`, `bias`, `standard_error`, `conf_low`, `conf_high`,
#'   `confidence_level`, `interval_type`, `replicates_requested`,
#'   `replicates_effective`, `replicates_failed`, `p_value` (always `NA`), and
#'   `p_value_note`.
#' @export
mb_bootstrap_summary = function(
  replicates,
  observed,
  conf = 0.95,
  type = c("percentile", "basic", "normal")
) {
  type = match.arg(type)
  conf = .mb_assert_probability(conf)
  if (!is.numeric(replicates) || !length(replicates)) {
    stop("`replicates` must be a non-empty numeric vector.", call. = FALSE)
  }
  requested = length(replicates)
  replicates = replicates[is.finite(replicates)]
  if (length(replicates) < 2L) {
    stop("`replicates` must contain at least two finite values.",
      call. = FALSE)
  }
  if (length(observed) != 1L || is.na(observed) || !is.numeric(observed) ||
    !is.finite(observed)) {
    stop("`observed` must be one finite numeric value.", call. = FALSE)
  }

  alpha = 1 - conf
  quantiles = stats::quantile(
    replicates,
    probs = c(alpha / 2, 1 - alpha / 2),
    names = FALSE,
    type = 8
  )
  bootstrap_mean = mean(replicates)
  bias = bootstrap_mean - observed
  standard_error = stats::sd(replicates)
  interval = switch(type,
    percentile = quantiles,
    basic = c(2 * observed - quantiles[[2L]],
      2 * observed - quantiles[[1L]]),
    normal = (observed - bias) +
      stats::qnorm(c(alpha / 2, 1 - alpha / 2)) * standard_error
  )

  structure(list(
    estimate = as.numeric(observed),
    bootstrap_mean = as.numeric(bootstrap_mean),
    bias = as.numeric(bias),
    standard_error = as.numeric(standard_error),
    conf_low = as.numeric(interval[[1L]]),
    conf_high = as.numeric(interval[[2L]]),
    confidence_level = conf,
    interval_type = type,
    replicates_requested = as.integer(requested),
    replicates_effective = as.integer(length(replicates)),
    replicates_failed = as.integer(requested - length(replicates)),
    p_value = NA_real_,
    p_value_note = paste(
      "Not computed: an ordinary bootstrap distribution estimates uncertainty",
      "and is not a null distribution for hypothesis testing."
    )
  ), class = "mbspls_bootstrap_summary")
}

#' Assert that exchangeability groups do not cross a split
#'
#' @param train_group Group identifiers in the analysis partition.
#' @param test_group Group identifiers in the assessment partition.
#'
#' @return `TRUE`, invisibly, or an error identifying leaked groups.
#' @export
mb_assert_disjoint_groups = function(train_group, test_group) {
  if (!length(train_group) || !length(test_group)) {
    stop("Training and assessment group identifiers must be non-empty.",
      call. = FALSE)
  }
  if (anyNA(train_group) || anyNA(test_group)) {
    stop("Training and assessment group identifiers must not be missing.",
      call. = FALSE)
  }
  overlap = intersect(
    unique(as.character(train_group)),
    unique(as.character(test_group))
  )
  if (length(overlap)) {
    stop(sprintf(
      paste0(
        "Resampling leakage: groups occur in both analysis and assessment ",
        "sets: %s."
      ),
      paste(overlap, collapse = ", ")
    ), call. = FALSE)
  }
  invisible(TRUE)
}

mb_task_group_vector = function(task, rows = task$row_ids) {
  group_columns = task$col_roles$group %||% character(0)
  if (!length(group_columns)) {
    return(NULL)
  }
  if (length(group_columns) != 1L) {
    stop("Exactly one task column may have the `group` role.", call. = FALSE)
  }
  values = task$data(rows = rows, cols = group_columns)[[group_columns]]
  if (length(values) != length(rows) || anyNA(values)) {
    stop("The task's group role must provide one non-missing value per row.",
      call. = FALSE)
  }
  values
}
