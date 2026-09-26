# Design-valid permutation inference for complete multiblock analyses.

.mb_validate_permutation_blocks = function(blocks) {
  if (!is.list(blocks) || length(blocks) < 2L) {
    stop("`blocks` must be a list containing at least two aligned data blocks.",
      call. = FALSE)
  }
  if (is.null(names(blocks))) {
    names(blocks) = paste0("block", seq_along(blocks))
  }
  if (anyNA(names(blocks)) || any(!nzchar(names(blocks))) ||
    anyDuplicated(names(blocks))) {
    stop("`blocks` must have unique, non-empty names.", call. = FALSE)
  }
  valid_block = vapply(blocks, function(x) {
    is.matrix(x) || is.data.frame(x)
  }, logical(1L))
  if (any(!valid_block)) {
    stop(sprintf(
      "Every block must be a matrix or data frame; invalid blocks: %s.",
      paste(names(blocks)[!valid_block], collapse = ", ")
    ), call. = FALSE)
  }

  nr = vapply(blocks, nrow, integer(1L))
  if (any(nr < 3L) || length(unique(nr)) != 1L) {
    stop(
      "Every block must contain the same aligned rows, with at least three rows.",
      call. = FALSE
    )
  }

  nc = vapply(blocks, ncol, integer(1L))
  if (any(nc < 1L)) {
    stop("Every block must contain at least one column.", call. = FALSE)
  }

  row_names = lapply(blocks, rownames)
  has_row_names = lengths(row_names) > 0L
  if (any(has_row_names) && !all(has_row_names)) {
    stop(
      "Either every block or no block must provide row names.",
      call. = FALSE
    )
  }
  if (all(has_row_names)) {
    invalid_row_names = vapply(row_names, function(value) {
      anyNA(value) || any(!nzchar(value)) || anyDuplicated(value) > 0L
    }, logical(1L))
    if (any(invalid_row_names)) {
      stop(sprintf(
        "Block row names must be unique and non-empty: %s.",
        paste(names(blocks)[invalid_row_names], collapse = ", ")
      ), call. = FALSE)
    }
    reference = row_names[[1L]]
    mismatch = !vapply(row_names, identical, logical(1L), reference)
    if (any(mismatch)) {
      stop(sprintf(
        "Block row names are not identically aligned: %s.",
        paste(names(blocks)[mismatch], collapse = ", ")
      ), call. = FALSE)
    }
  }
  blocks
}

.mb_resolve_block_selection = function(selection, block_names) {
  if (is.null(selection)) {
    selection = block_names[-1L]
  }
  if (is.numeric(selection)) {
    if (anyNA(selection) || any(!is.finite(selection)) ||
      any(selection != floor(selection)) ||
      any(selection < 1L | selection > length(block_names))) {
      stop("Numeric `permute_blocks` entries must be valid block indices.",
        call. = FALSE)
    }
    selection = block_names[as.integer(selection)]
  }
  if (!is.character(selection) || !length(selection) || anyNA(selection) ||
    any(!nzchar(selection))) {
    stop("`permute_blocks` must identify at least one block.", call. = FALSE)
  }
  unknown = setdiff(selection, block_names)
  if (length(unknown)) {
    stop(sprintf(
      "Unknown `permute_blocks`: %s.", paste(unknown, collapse = ", ")
    ), call. = FALSE)
  }
  selection = unique(selection)
  if (length(selection) >= length(block_names)) {
    stop(
      "At least one block must remain fixed as the alignment anchor.",
      call. = FALSE
    )
  }
  selection
}

.mb_validate_exchangeability = function(
  n,
  exchangeability_unit = NULL,
  within_unit = NULL,
  strata = NULL
) {
  check_vector = function(x, name) {
    if (!is.atomic(x) || !is.null(dim(x)) || length(x) != n || anyNA(x)) {
      stop(sprintf(
        "`%s` must contain one non-missing value for every row.", name
      ), call. = FALSE)
    }
    key = as.character(x)
    if (anyNA(key) || any(!nzchar(key))) {
      stop(sprintf("`%s` identifiers must be non-empty.", name),
        call. = FALSE)
    }
    key
  }

  strata_key = if (is.null(strata)) {
    rep.int(".all", n)
  } else {
    check_vector(strata, "strata")
  }

  if (is.null(exchangeability_unit)) {
    if (!is.null(within_unit)) {
      stop("`within_unit` requires `exchangeability_unit`.", call. = FALSE)
    }
    strata_sizes = lengths(split(seq_len(n), strata_key))
    if (all(strata_sizes < 2L)) {
      stop("No stratum contains two exchangeable rows.", call. = FALSE)
    }
    return(list(
      n = n,
      unit = NULL,
      within = NULL,
      strata = strata_key,
      unit_level = FALSE,
      n_units = n
    ))
  }

  unit_key = check_vector(exchangeability_unit, "exchangeability_unit")
  units = unique(unit_key)
  unit_rows = lapply(units, function(id) which(unit_key == id))
  names(unit_rows) = units

  unit_strata = vapply(unit_rows, function(rows) {
    values = unique(strata_key[rows])
    if (length(values) != 1L) NA_character_ else values
  }, character(1L))
  inconsistent_strata = is.na(unit_strata)
  if (any(inconsistent_strata)) {
    stop(sprintf(
      "Strata vary within exchangeability units: %s.",
      paste(units[inconsistent_strata], collapse = ", ")
    ), call. = FALSE)
  }

  repeated = lengths(unit_rows) > 1L
  if (any(repeated) && is.null(within_unit)) {
    stop(
      paste(
        "Repeated-row exchangeability units require `within_unit` labels",
        "so complete trajectories can be matched without relying on row order."
      ),
      call. = FALSE
    )
  }
  within_key = if (is.null(within_unit)) {
    rep.int(".row", n)
  } else {
    check_vector(within_unit, "within_unit")
  }

  duplicate_positions = vapply(unit_rows, function(rows) {
    anyDuplicated(within_key[rows]) > 0L
  }, logical(1L))
  if (any(duplicate_positions)) {
    stop(sprintf(
      "`within_unit` labels must be unique inside each unit: %s.",
      paste(units[duplicate_positions], collapse = ", ")
    ), call. = FALSE)
  }

  for (level in unique(unit_strata)) {
    eligible = units[unit_strata == level]
    reference = sort(within_key[unit_rows[[eligible[[1L]]]]])
    same_pattern = vapply(eligible, function(id) {
      identical(sort(within_key[unit_rows[[id]]]), reference)
    }, logical(1L))
    if (any(!same_pattern)) {
      stop(sprintf(
        paste0(
          "Exchangeability units in stratum '%s' do not share the same ",
          "`within_unit` pattern: %s."
        ),
        level, paste(eligible[!same_pattern], collapse = ", ")
      ), call. = FALSE)
    }
  }

  units_per_stratum = table(unit_strata)
  if (all(units_per_stratum < 2L)) {
    stop("No stratum contains two exchangeable units.", call. = FALSE)
  }

  list(
    n = n,
    unit = unit_key,
    within = within_key,
    strata = strata_key,
    units = units,
    unit_rows = unit_rows,
    unit_strata = unit_strata,
    unit_level = TRUE,
    n_units = length(units)
  )
}

.mb_draw_permutation_index = function(exchangeability) {
  n = exchangeability$n
  index = seq_len(n)

  if (!isTRUE(exchangeability$unit_level)) {
    for (level in unique(exchangeability$strata)) {
      rows = which(exchangeability$strata == level)
      index[rows] = rows[sample.int(length(rows), replace = FALSE)]
    }
    return(index)
  }

  for (level in unique(exchangeability$unit_strata)) {
    destination_units = exchangeability$units[
      exchangeability$unit_strata == level
    ]
    source_units = destination_units[
      sample.int(length(destination_units), replace = FALSE)
    ]

    for (i in seq_along(destination_units)) {
      destination_rows = exchangeability$unit_rows[[destination_units[[i]]]]
      source_rows = exchangeability$unit_rows[[source_units[[i]]]]
      position = match(
        exchangeability$within[destination_rows],
        exchangeability$within[source_rows]
      )
      if (anyNA(position)) {
        stop(
          "Internal error while matching repeated-measure positions.",
          call. = FALSE
        )
      }
      index[destination_rows] = source_rows[position]
    }
  }
  index
}

.mb_permute_block_rows = function(block, index) {
  original_rows = rownames(block)
  out = block[index, , drop = FALSE]
  if (!is.null(original_rows)) {
    rownames(out) = original_rows
  }
  out
}

.mb_analysis_result = function(value, context) {
  statistic = if (is.list(value) && !is.null(value$statistic)) {
    value$statistic
  } else {
    value
  }
  if (length(statistic) != 1L || !is.numeric(statistic) ||
    is.na(statistic) || !is.finite(statistic)) {
    stop(sprintf(
      "%s analysis must return one finite numeric statistic, directly or as `$statistic`.",
      context
    ), call. = FALSE)
  }
  list(statistic = as.numeric(statistic), result = value)
}

#' Complete-analysis permutation test for aligned data blocks
#'
#' `mb_permutation_test()` is a design engine for a single pre-specified test
#' statistic. It breaks the selected block alignments and calls `analysis()` on
#' the resulting raw blocks for the observed data and every sampled
#' permutation. Preprocessing, filtering, tuning, component selection, fitting,
#' and scoring that used the observed alignments must therefore live inside
#' `analysis()`.
#' Failed or non-finite analyses abort the test; failed permutations are never
#' dropped or replaced by additional draws.
#'
#' Selected blocks are permuted independently while at least one block remains
#' fixed as an alignment anchor. With the default selection (all blocks except
#' the first), the generated null is mutual block independence. Permuting one
#' target block instead tests its independence from the joint remaining blocks.
#'
#' Repeated observations can be moved only as complete exchangeability units.
#' Repeated units require explicit `within_unit` positions and an identical
#' position pattern within every stratum. Unsupported incomplete or unequal
#' trajectories fail instead of falling back to row-wise shuffling.
#'
#' This function does not make residualised observations exchangeable. Tests
#' involving nuisance covariates require a design-specific method such as a
#' justified residual permutation or randomisation scheme; ordinary
#' residualisation followed by this row shuffle is not automatically valid.
#'
#' @param blocks Named list of at least two aligned matrices or data frames.
#'   Supply data before any alignment-dependent preprocessing or tuning.
#' @param analysis Function taking the complete named block list. It must rerun
#'   the full data-dependent analysis and return one finite numeric statistic,
#'   or a list containing that value in `statistic`.
#' @param permute_blocks Names or indices of blocks to permute independently.
#'   The default permutes every block except the first.
#' @param n_perm Number of random permutations. The minimum attainable sampled
#'   p-value is `1 / (n_perm + 1)`.
#' @param exchangeability_unit Optional vector identifying independent units.
#'   Whole units, not individual rows, are reassigned.
#' @param within_unit Required for repeated-row units; identifies positions
#'   such as visits that must be matched when whole trajectories are moved.
#' @param strata Optional vector restricting row or unit permutations within
#'   design-valid strata. It must be constant within an exchangeability unit.
#' @param alternative Direction of extremeness for the statistic.
#' @param null_center Explicit null centre required for a two-sided statistic.
#' @param seed Non-negative seed used to sample permutation maps. The caller's
#'   RNG kind and state are restored.
#' @param analysis_seed Non-negative seed reset before every call to
#'   `analysis()`. This makes a stochastic fitting/tuning algorithm a
#'   deterministic statistic. Outcome-dependent splits, including stratified
#'   folds, must be recreated inside `analysis()` for the permuted outcomes.
#' @param keep_null If `TRUE`, retain the sampled null statistics.
#' @param null_hypothesis Optional precise description of the scientific null.
#'
#' @return An object of class `mb_permutation_test` containing the observed
#'   statistic, corrected p-value, exceedance count, Monte Carlo precision,
#'   exchangeability metadata, and (optionally) the null distribution.
#' @export
mb_permutation_test = function(
  blocks,
  analysis,
  permute_blocks = NULL,
  n_perm = 999L,
  exchangeability_unit = NULL,
  within_unit = NULL,
  strata = NULL,
  alternative = c("greater", "less", "two.sided"),
  null_center = NULL,
  seed = 1L,
  analysis_seed = 1L,
  keep_null = TRUE,
  null_hypothesis = NULL
) {
  blocks = .mb_validate_permutation_blocks(blocks)
  if (!is.function(analysis)) {
    stop("`analysis` must be a function.", call. = FALSE)
  }
  permute_blocks = .mb_resolve_block_selection(
    permute_blocks, names(blocks)
  )
  n_perm = .mb_assert_scalar_integer(n_perm, "n_perm")
  seed = .mb_assert_scalar_integer(seed, "seed", lower = 0L)
  analysis_seed = .mb_assert_scalar_integer(
    analysis_seed, "analysis_seed", lower = 0L
  )
  if (length(keep_null) != 1L || is.na(keep_null) || !is.logical(keep_null)) {
    stop("`keep_null` must be one non-missing logical value.", call. = FALSE)
  }
  alternative = match.arg(alternative)
  if (identical(alternative, "two.sided")) {
    if (length(null_center) != 1L || !is.numeric(null_center) ||
      is.na(null_center) || !is.finite(null_center)) {
      stop(
        "`null_center` must be one finite number for a two-sided test.",
        call. = FALSE
      )
    }
  }
  if (!is.null(null_hypothesis) &&
    (length(null_hypothesis) != 1L || is.na(null_hypothesis) ||
      !is.character(null_hypothesis) || !nzchar(null_hypothesis))) {
    stop("`null_hypothesis` must be NULL or one non-empty string.",
      call. = FALSE)
  }

  exchangeability = .mb_validate_exchangeability(
    n = nrow(blocks[[1L]]),
    exchangeability_unit = exchangeability_unit,
    within_unit = within_unit,
    strata = strata
  )

  if (is.null(null_hypothesis)) {
    fixed = setdiff(names(blocks), permute_blocks)
    null_hypothesis = if (length(permute_blocks) == 1L) {
      sprintf(
        "Block '%s' is independent of the joint fixed block(s) %s within the supplied exchangeability design.",
        permute_blocks,
        paste(sprintf("'%s'", fixed), collapse = ", ")
      )
    } else {
      sprintf(
        paste0(
          "Permuted blocks %s are mutually independent and independent of ",
          "the joint fixed block(s) %s within the supplied exchangeability design."
        ),
        paste(sprintf("'%s'", permute_blocks), collapse = ", "),
        paste(sprintf("'%s'", fixed), collapse = ", ")
      )
    }
  }

  run_analysis = function(current_blocks, context) {
    value = tryCatch(
      with_seed_local(analysis_seed, function() analysis(current_blocks)),
      error = function(error) {
        stop(sprintf("%s analysis failed: %s", context, conditionMessage(error)),
          call. = FALSE)
      }
    )
    .mb_analysis_result(value, context)
  }

  run_test = function() {
    observed_result = run_analysis(blocks, "Observed")
    null_distribution = numeric(n_perm)

    for (b in seq_len(n_perm)) {
      permuted = blocks
      for (block_name in permute_blocks) {
        index = .mb_draw_permutation_index(exchangeability)
        permuted[[block_name]] = .mb_permute_block_rows(
          blocks[[block_name]], index
        )
      }
      null_distribution[[b]] = run_analysis(
        permuted, sprintf("Permutation %d", b)
      )$statistic
    }

    p_value = mb_permutation_pvalue(
      observed = observed_result$statistic,
      null_distribution = null_distribution,
      alternative = alternative,
      null_center = null_center
    )
    extreme = .mb_permutation_extreme(
      observed_result$statistic, null_distribution, alternative, null_center
    )
    exceedances = sum(extreme)
    tail_estimate = exceedances / n_perm
    mc_se = sqrt(tail_estimate * (1 - tail_estimate) / n_perm)
    mc_ci = stats::binom.test(
      exceedances, n_perm, conf.level = 0.95
    )$conf.int

    structure(list(
      method = "Complete-analysis Monte Carlo permutation test",
      null_hypothesis = null_hypothesis,
      alternative = alternative,
      null_center = if (identical(alternative, "two.sided")) {
        as.numeric(null_center)
      } else {
        NA_real_
      },
      statistic = observed_result$statistic,
      observed_result = observed_result$result,
      p_value = as.numeric(p_value),
      exceedances = as.integer(exceedances),
      n_perm = n_perm,
      minimum_p_value = 1 / (n_perm + 1),
      monte_carlo_tail_estimate = as.numeric(tail_estimate),
      monte_carlo_standard_error = as.numeric(mc_se),
      monte_carlo_conf_low = as.numeric(mc_ci[[1L]]),
      monte_carlo_conf_high = as.numeric(mc_ci[[2L]]),
      permute_blocks = permute_blocks,
      fixed_blocks = setdiff(names(blocks), permute_blocks),
      exchangeability_level = if (exchangeability$unit_level) {
        "whole_unit"
      } else {
        "row"
      },
      n_rows = nrow(blocks[[1L]]),
      n_exchangeability_units = exchangeability$n_units,
      n_strata = length(unique(exchangeability$strata)),
      seed = seed,
      analysis_seed = analysis_seed,
      null_distribution = if (isTRUE(keep_null)) null_distribution else NULL
    ), class = "mb_permutation_test")
  }

  with_seed_local(seed, run_test)
}

#' @export
print.mb_permutation_test = function(x, ...) {
  cat(x$method, "\n", sep = "")
  cat("Null: ", x$null_hypothesis, "\n", sep = "")
  cat(sprintf(
    "Statistic = %.6g; p = %.6g (%d/%d sampled statistics at least as extreme)\n",
    x$statistic, x$p_value, x$exceedances, x$n_perm
  ))
  cat(sprintf(
    "Exchangeability: %s; units = %d; strata = %d\n",
    x$exchangeability_level, x$n_exchangeability_units, x$n_strata
  ))
  invisible(x)
}

.mb_numeric_inference_blocks = function(blocks) {
  blocks = .mb_validate_permutation_blocks(blocks)
  lapply(names(blocks), function(name) {
    value = blocks[[name]]
    if (is.data.frame(value)) {
      bad = !vapply(value, is.numeric, logical(1L))
      if (any(bad)) {
        stop(sprintf(
          "Block '%s' contains non-numeric columns: %s.",
          name, paste(names(value)[bad], collapse = ", ")
        ), call. = FALSE)
      }
      value = as.matrix(value)
    }
    if (!is.numeric(value) || anyNA(value) || any(!is.finite(value))) {
      stop(sprintf(
        "Block '%s' must be a finite numeric matrix.", name
      ), call. = FALSE)
    }
    storage.mode(value) = "double"
    value
  }) |>
    stats::setNames(names(blocks))
}

.mb_prepare_c_matrix = function(blocks, c_matrix, ncomp, ncomp_missing) {
  p = vapply(blocks, ncol, integer(1L))
  if (!ncomp_missing) {
    ncomp = .mb_assert_scalar_integer(ncomp, "ncomp")
  }
  if (is.null(c_matrix)) {
    ncomp = .mb_assert_scalar_integer(ncomp, "ncomp")
    c_matrix = matrix(
      rep(sqrt(p), ncomp),
      nrow = length(blocks),
      ncol = ncomp,
      dimnames = list(names(blocks), paste0("LC", seq_len(ncomp)))
    )
  } else {
    if (!is.matrix(c_matrix) || !is.numeric(c_matrix) ||
      anyNA(c_matrix) || any(!is.finite(c_matrix)) || ncol(c_matrix) < 1L) {
      stop("`c_matrix` must be a finite numeric matrix with at least one column.",
        call. = FALSE)
    }
    if (nrow(c_matrix) != length(blocks)) {
      stop(sprintf(
        "`c_matrix` must have %d rows, one per block.", length(blocks)
      ), call. = FALSE)
    }
    if (!is.null(rownames(c_matrix))) {
      if (anyNA(rownames(c_matrix)) || any(!nzchar(rownames(c_matrix))) ||
        anyDuplicated(rownames(c_matrix))) {
        stop("Named `c_matrix` rows must be unique and non-empty.",
          call. = FALSE)
      }
      missing_rows = setdiff(names(blocks), rownames(c_matrix))
      extra_rows = setdiff(rownames(c_matrix), names(blocks))
      if (length(missing_rows) || length(extra_rows)) {
        stop(
          "Named `c_matrix` rows must match the block names exactly.",
          call. = FALSE
        )
      }
      c_matrix = c_matrix[names(blocks), , drop = FALSE]
    }
    if (!is.null(colnames(c_matrix)) &&
      (anyNA(colnames(c_matrix)) || any(!nzchar(colnames(c_matrix))) ||
        anyDuplicated(colnames(c_matrix)))) {
      stop("Named `c_matrix` columns must be unique and non-empty.",
        call. = FALSE)
    }
    if (!ncomp_missing && !identical(as.integer(ncomp), ncol(c_matrix))) {
      stop("Explicit `ncomp` must equal `ncol(c_matrix)`.", call. = FALSE)
    }
    ncomp = ncol(c_matrix)
  }

  lower_bad = c_matrix < 1
  upper = matrix(
    rep(sqrt(p), ncomp), nrow = length(blocks), ncol = ncomp
  )
  upper_bad = c_matrix > upper + sqrt(.Machine$double.eps)
  if (any(lower_bad) || any(upper_bad)) {
    stop(
      "Every sparsity budget in `c_matrix` must lie in [1, sqrt(p_block)].",
      call. = FALSE
    )
  }
  dimnames(c_matrix) = list(
    names(blocks), colnames(c_matrix) %||% paste0("LC", seq_len(ncomp))
  )
  c_matrix
}

.mb_standardize_inference_blocks = function(blocks) {
  lapply(names(blocks), function(name) {
    value = blocks[[name]]
    center = colMeans(value)
    scale = apply(value, 2L, stats::sd)
    bad = !is.finite(scale) | scale <= sqrt(.Machine$double.eps)
    if (any(bad)) {
      labels = colnames(value) %||% paste0("V", seq_len(ncol(value)))
      stop(sprintf(
        "Block '%s' has constant or numerically degenerate columns: %s.",
        name, paste(labels[bad], collapse = ", ")
      ), call. = FALSE)
    }
    sweep(sweep(value, 2L, center, FUN = "-"), 2L, scale, FUN = "/")
  }) |>
    stats::setNames(names(blocks))
}

.mb_target_component_statistics = function(
  scores,
  ncomp,
  block_names,
  target_block,
  correlation_method,
  performance_metric
) {
  n_blocks = length(block_names)
  target_index = match(target_block, block_names)
  other = setdiff(seq_len(n_blocks), target_index)

  vapply(seq_len(ncomp), function(component) {
    columns = ((component - 1L) * n_blocks + 1L):(component * n_blocks)
    component_scores = scores[, columns, drop = FALSE]
    correlations = vapply(other, function(index) {
      stats::cor(
        component_scores[, target_index],
        component_scores[, index],
        method = correlation_method
      )
    }, numeric(1L))
    if (any(!is.finite(correlations))) {
      stop(sprintf(
        "Target association is undefined for LC%d because a score is degenerate.",
        component
      ), call. = FALSE)
    }
    if (identical(performance_metric, "frobenius")) {
      sqrt(sum(correlations^2))
    } else {
      mean(abs(correlations))
    }
  }, numeric(1L))
}

#' Omnibus MB-sPLS permutation test
#'
#' Fits the requested sequential MB-sPLS analysis to the observed aligned
#' blocks and refits the same analysis from scratch for every valid
#' permutation. The returned p-value tests one omnibus null. It is not a
#' component-rank test and must not be reported as evidence that LC2 or a later
#' component exists after earlier non-null components have been removed.
#'
#' Four pre-specified statistics are available:
#'
#' * `"global_lc1"`: the optimized first-component cross-block objective;
#' * `"global_sum"`: the sum of objectives across a pre-specified number of
#'   sequential components;
#' * `"target_lc1"`: first-component association between a named target block
#'   and the remaining block scores;
#' * `"target_sum"`: the sum of those target associations across a
#'   pre-specified number of components.
#'
#' The two `global_*` statistics test the block-independence null generated by
#' `permute_blocks`. The `target_*` statistics require `target_block`, permute
#' that block only, and test its independence from the joint predictor blocks.
#' In all cases the p-value is global; the LC label identifies the statistic,
#' not a separately proven population component.
#'
#' Inputs should be raw finite numeric blocks. Optional z-standardisation is
#' refitted inside every analysis call. `ncomp`, `c_matrix`, the statistic, and
#' all other settings must be specified independently of the tested block
#' alignment. If any were selected from the observed association, use
#' [mb_permutation_test()] with a callback that repeats that selection inside
#' every permutation.
#'
#' @inheritParams mb_permutation_test
#' @param c_matrix Optional block-by-component matrix of fixed L1 budgets. With
#'   `NULL`, dense budgets `sqrt(p_block)` are used.
#' @param ncomp Pre-specified number of components. When `c_matrix` is supplied,
#'   it is inferred from its columns.
#' @param statistic One of `"global_lc1"`, `"global_sum"`,
#'   `"target_lc1"`, or `"target_sum"`.
#' @param target_block Name or index of the target block for a `target_*` test.
#' @param standardize If `TRUE`, z-standardise each block inside every observed
#'   and permuted fit.
#' @param correlation_method `"pearson"` or `"spearman"`.
#' @param performance_metric `"mac"` or `"frobenius"`.
#' @param max_iter Maximum iterations per component fit.
#' @param tol Positive numerical convergence tolerance.
#'
#' @return An `mbspls_permutation_test` object. It contains one corrected global
#'   p-value and the observed per-component objectives for descriptive use.
#' @export
mbspls_permutation_test = function(
  blocks,
  c_matrix = NULL,
  ncomp = 1L,
  statistic = c("global_lc1", "global_sum", "target_lc1", "target_sum"),
  target_block = NULL,
  permute_blocks = NULL,
  n_perm = 999L,
  exchangeability_unit = NULL,
  within_unit = NULL,
  strata = NULL,
  standardize = TRUE,
  correlation_method = c("pearson", "spearman"),
  performance_metric = c("mac", "frobenius"),
  max_iter = 500L,
  tol = 1e-4,
  seed = 1L,
  analysis_seed = 1L,
  keep_null = TRUE
) {
  ncomp_missing = missing(ncomp)
  blocks = .mb_numeric_inference_blocks(blocks)
  statistic = match.arg(statistic)
  correlation_method = match.arg(correlation_method)
  performance_metric = match.arg(performance_metric)
  if (length(standardize) != 1L || is.na(standardize) ||
    !is.logical(standardize)) {
    stop("`standardize` must be one non-missing logical value.",
      call. = FALSE)
  }
  max_iter = .mb_assert_scalar_integer(max_iter, "max_iter")
  if (length(tol) != 1L || !is.numeric(tol) || is.na(tol) ||
    !is.finite(tol) || tol <= 0) {
    stop("`tol` must be one finite positive number.", call. = FALSE)
  }
  c_matrix = .mb_prepare_c_matrix(
    blocks, c_matrix, ncomp, ncomp_missing
  )
  ncomp = ncol(c_matrix)
  rank_blocks = if (isTRUE(standardize)) {
    .mb_standardize_inference_blocks(blocks)
  } else {
    blocks
  }
  .mb_assert_component_rank(rank_blocks, ncomp, "mbspls_permutation_test()")

  is_target = startsWith(statistic, "target_")
  if (is_target) {
    if (is.numeric(target_block) && length(target_block) == 1L &&
      !is.na(target_block) && is.finite(target_block) &&
      target_block == floor(target_block) && target_block >= 1L &&
      target_block <= length(blocks)) {
      target_block = names(blocks)[as.integer(target_block)]
    }
    if (length(target_block) != 1L || !is.character(target_block) ||
      is.na(target_block) || !target_block %in% names(blocks)) {
      stop(
        "A `target_*` statistic requires one valid `target_block` name or index.",
        call. = FALSE
      )
    }
    if (!is.null(permute_blocks)) {
      resolved = .mb_resolve_block_selection(permute_blocks, names(blocks))
      if (!identical(resolved, target_block)) {
        stop("A `target_*` test must permute the target block only.",
          call. = FALSE)
      }
    }
    permute_blocks = target_block
    null_hypothesis = sprintf(
      paste0(
        "Target block '%s' is independent of the joint predictor blocks %s ",
        "within the supplied exchangeability design."
      ),
      target_block,
      paste(sprintf("'%s'", setdiff(names(blocks), target_block)),
        collapse = ", ")
    )
  } else {
    if (!is.null(target_block)) {
      stop("`target_block` is only used by a `target_*` statistic.",
        call. = FALSE)
    }
    permute_blocks = .mb_resolve_block_selection(
      permute_blocks, names(blocks)
    )
    null_hypothesis = NULL
  }

  analysis = function(current_blocks) {
    fit_blocks = if (isTRUE(standardize)) {
      .mb_standardize_inference_blocks(current_blocks)
    } else {
      current_blocks
    }
    fit = cpp_mbspls_multi_lv_cmatrix(
      X_blocks = fit_blocks,
      c_matrix = c_matrix,
      max_iter = max_iter,
      tol = tol,
      spearman = identical(correlation_method, "spearman"),
      do_perm = FALSE,
      n_perm = 1L,
      alpha = 0.05,
      frobenius = identical(performance_metric, "frobenius")
    )
    component_objectives = as.numeric(fit$objective)
    if (length(component_objectives) != ncomp ||
      any(!is.finite(component_objectives))) {
      stop(
        "MB-sPLS did not return every requested finite component objective.",
        call. = FALSE
      )
    }
    if (!is.matrix(fit$T_mat) ||
      !identical(dim(fit$T_mat), c(nrow(fit_blocks[[1L]]), ncomp * length(blocks))) ||
      any(!is.finite(fit$T_mat)) ||
      any(apply(fit$T_mat, 2L, stats::sd) <= sqrt(.Machine$double.eps))) {
      stop(paste(
        "MB-sPLS must return a finite, nondegenerate score for every",
        "requested block and component."
      ), call. = FALSE)
    }

    component_statistics = if (is_target) {
      .mb_target_component_statistics(
        scores = fit$T_mat,
        ncomp = ncomp,
        block_names = names(blocks),
        target_block = target_block,
        correlation_method = correlation_method,
        performance_metric = performance_metric
      )
    } else {
      component_objectives
    }
    test_statistic = switch(statistic,
      global_lc1 = component_statistics[[1L]],
      global_sum = sum(component_statistics),
      target_lc1 = component_statistics[[1L]],
      target_sum = sum(component_statistics)
    )

    list(
      statistic = as.numeric(test_statistic),
      component_statistics = component_statistics,
      component_objectives = component_objectives
    )
  }

  result = mb_permutation_test(
    blocks = blocks,
    analysis = analysis,
    permute_blocks = permute_blocks,
    n_perm = n_perm,
    exchangeability_unit = exchangeability_unit,
    within_unit = within_unit,
    strata = strata,
    alternative = "greater",
    seed = seed,
    analysis_seed = analysis_seed,
    keep_null = keep_null,
    null_hypothesis = null_hypothesis
  )
  result$statistic_name = statistic
  result$component_statistics = result$observed_result$component_statistics
  result$component_objectives = result$observed_result$component_objectives
  result$c_matrix = c_matrix
  result$ncomp = ncomp
  result$target_block = target_block
  result$standardize = standardize
  result$correlation_method = correlation_method
  result$performance_metric = performance_metric
  result$validity_scope = paste(
    "One omnibus p-value for the stated block-independence null.",
    "Component statistics after LC1 are descriptive and are not rank-null p-values."
  )
  class(result) = c("mbspls_permutation_test", class(result))
  result
}

.mb_lc_score_statistic = function(
  score_blocks,
  permute_blocks,
  correlation_method,
  performance_metric
) {
  score_matrix = do.call(cbind, lapply(score_blocks, function(x) x[, 1L]))
  score_scale = apply(score_matrix, 2L, stats::sd)
  if (any(!is.finite(score_scale)) ||
    any(score_scale <= sqrt(.Machine$double.eps))) {
    stop(
      "An LC confirmation statistic is undefined because a score is degenerate.",
      call. = FALSE
    )
  }
  correlations = stats::cor(score_matrix, method = correlation_method)
  pairs = utils::combn(seq_along(score_blocks), 2L)
  active = apply(pairs, 2L, function(pair) {
    any(names(score_blocks)[pair] %in% permute_blocks)
  })
  pairs = pairs[, active, drop = FALSE]
  values = vapply(seq_len(ncol(pairs)), function(index) {
    correlations[pairs[1L, index], pairs[2L, index]]
  }, numeric(1L))
  if (!length(values) || any(!is.finite(values))) {
    stop(
      "An LC confirmation statistic is undefined because a score is degenerate.",
      call. = FALSE
    )
  }
  if (identical(performance_metric, "frobenius")) {
    sqrt(sum(values^2))
  } else {
    mean(abs(values))
  }
}

.mb_validate_confirmation_scores = function(scores) {
  scores = .mb_numeric_inference_blocks(scores)
  component_counts = vapply(scores, ncol, integer(1L))
  if (length(unique(component_counts)) != 1L || component_counts[[1L]] < 1L) {
    stop(
      "Every confirmation score block must contain the same positive number of LCs.",
      call. = FALSE
    )
  }

  component_names = lapply(scores, colnames)
  named = lengths(component_names) > 0L
  if (any(named) && !all(named)) {
    stop(
      "Either every confirmation score block or no score block must name its LCs.",
      call. = FALSE
    )
  }
  if (all(named)) {
    reference = component_names[[1L]]
    if (anyNA(reference) || any(!nzchar(reference)) || anyDuplicated(reference)) {
      stop("Confirmation LC names must be unique and non-empty.", call. = FALSE)
    }
    mismatch = !vapply(component_names, identical, logical(1L), reference)
    if (any(mismatch)) {
      stop(sprintf(
        "Confirmation LC names/order differ in blocks: %s.",
        paste(names(scores)[mismatch], collapse = ", ")
      ), call. = FALSE)
    }
  } else {
    reference = sprintf("LC_%02d", seq_len(component_counts[[1L]]))
    scores = lapply(scores, function(x) {
      colnames(x) = reference
      x
    }) |>
      stats::setNames(names(scores))
  }

  list(scores = scores, component_names = reference)
}

#' Permutation confirmation tests for frozen MB-sPLS LCs
#'
#' Tests whether pre-specified, frozen block-score vectors reproduce
#' cross-block association in genuinely independent confirmation data. The
#' weights, loadings, deflation path, preprocessing, sparsity, component count,
#' and any component selection must have been learned without using these
#' confirmation observations. Set `independent_confirmation = TRUE` to make
#' that design assertion explicitly; the function cannot verify it from score
#' matrices alone.
#'
#' Each LC is tested as a fixed low-dimensional score association. This avoids
#' reusing the data that optimized its directions. It does not test whether the
#' population cross-block rank is at least `k`, and it does not make an LC
#' selected after inspecting the confirmation data valid. Raw sampled
#' permutation p-values use inclusive ties and `(b + 1) / (B + 1)`. Holm
#' adjustment is always reported across the complete supplied LC family.
#'
#' Row, stratum, and complete-unit exchangeability follow
#' [mb_permutation_test()]. With a single `permute_blocks` entry, only
#' correlations involving that target score block enter the statistic. With
#' multiple entries, every score-pair relation broken by the permutation enters
#' the statistic. The null requires independence of the permuted score blocks
#' from each other and the joint fixed score blocks within the exchangeability
#' design; zero correlation alone is insufficient.
#'
#' @param scores Named list of finite numeric confirmation-score matrices. Rows
#'   are aligned observations, columns are the same frozen LCs in the same
#'   order, and blocks contain their respective scores.
#' @param independent_confirmation Must be explicitly `TRUE`, asserting that
#'   the confirmation observations were not used to learn or select any part of
#'   the score transformation or LC family.
#' @param alpha Family-wise significance level used with Holm-adjusted p-values.
#' @inheritParams mb_permutation_test
#' @inheritParams mbspls_permutation_test
#'
#' @return An `mb_lc_confirmation_test` object with one row per LC containing
#'   the fixed-score association, raw sampled permutation p-value,
#'   Holm-adjusted p-value, Monte Carlo standard error, and adjusted decision.
#' @export
mb_lc_confirmation_test = function(
  scores,
  independent_confirmation = FALSE,
  permute_blocks = NULL,
  n_perm = 999L,
  exchangeability_unit = NULL,
  within_unit = NULL,
  strata = NULL,
  correlation_method = c("pearson", "spearman"),
  performance_metric = c("mac", "frobenius"),
  alpha = 0.05,
  seed = 1L,
  keep_null = FALSE
) {
  if (!isTRUE(independent_confirmation)) {
    stop(
      paste(
        "Set `independent_confirmation = TRUE` only when these observations",
        "were not used to learn preprocessing, weights, deflation, tuning,",
        "component count, or the reported LC family."
      ),
      call. = FALSE
    )
  }
  validated = .mb_validate_confirmation_scores(scores)
  scores = validated$scores
  component_names = validated$component_names
  permute_blocks = .mb_resolve_block_selection(
    permute_blocks, names(scores)
  )
  n_perm = .mb_assert_scalar_integer(n_perm, "n_perm")
  seed = .mb_assert_scalar_integer(seed, "seed", lower = 0L)
  alpha = .mb_assert_probability(alpha, "alpha")
  correlation_method = match.arg(correlation_method)
  performance_metric = match.arg(performance_metric)
  if (length(keep_null) != 1L || is.na(keep_null) || !is.logical(keep_null)) {
    stop("`keep_null` must be one non-missing logical value.", call. = FALSE)
  }

  component_seeds = with_seed_local(seed, function() {
    sample.int(.Machine$integer.max, length(component_names), replace = FALSE)
  })
  tests = lapply(seq_along(component_names), function(component) {
    component_blocks = lapply(scores, function(x) {
      x[, component, drop = FALSE]
    }) |>
      stats::setNames(names(scores))
    analysis = function(current_scores) {
      .mb_lc_score_statistic(
        score_blocks = current_scores,
        permute_blocks = permute_blocks,
        correlation_method = correlation_method,
        performance_metric = performance_metric
      )
    }
    mb_permutation_test(
      blocks = component_blocks,
      analysis = analysis,
      permute_blocks = permute_blocks,
      n_perm = n_perm,
      exchangeability_unit = exchangeability_unit,
      within_unit = within_unit,
      strata = strata,
      alternative = "greater",
      seed = component_seeds[[component]],
      analysis_seed = 0L,
      keep_null = keep_null,
      null_hypothesis = sprintf(
        paste0(
          "In independent confirmation data, frozen score blocks for %s ",
          "satisfy the block-independence null under the supplied ",
          "exchangeability design."
        ),
        component_names[[component]]
      )
    )
  })

  raw_p = vapply(tests, `[[`, numeric(1L), "p_value")
  holm_p = stats::p.adjust(raw_p, method = "holm")
  results = data.frame(
    component = component_names,
    statistic = vapply(tests, `[[`, numeric(1L), "statistic"),
    p_value_raw = raw_p,
    p_value_holm = as.numeric(holm_p),
    monte_carlo_standard_error = vapply(
      tests, `[[`, numeric(1L), "monte_carlo_standard_error"
    ),
    significant_holm = holm_p <= alpha,
    stringsAsFactors = FALSE
  )

  structure(list(
    method = paste(
      "Independent-confirmation permutation tests for frozen MB-sPLS LC",
      "score associations"
    ),
    results = results,
    tests = tests,
    alpha = alpha,
    adjustment = "holm",
    minimum_raw_p_value = 1 / (n_perm + 1),
    n_perm = n_perm,
    permute_blocks = permute_blocks,
    fixed_blocks = setdiff(names(scores), permute_blocks),
    correlation_method = correlation_method,
    performance_metric = performance_metric,
    seed = seed,
    component_seeds = component_seeds,
    validity_scope = paste(
      "Conditional confirmation of fixed learned score associations.",
      "Not a population-rank test and not valid after reusing confirmation data for fitting or selection."
    )
  ), class = "mb_lc_confirmation_test")
}

#' @export
print.mb_lc_confirmation_test = function(x, ...) {
  cat(x$method, "\n", sep = "")
  print(x$results, row.names = FALSE)
  cat("Scope: ", x$validity_scope, "\n", sep = "")
  invisible(x)
}
