# Design-valid permutation inference for complete multiblock analyses.

# Row identifiers of a block. Automatic data-frame row names ("1", "2", ...)
# are positions, not identifiers, and are therefore treated as absent.
.mb_block_row_ids = function(x) {
  if (is.data.frame(x) && .row_names_info(x) < 0L) NULL else rownames(x)
}

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

  row_names = lapply(blocks, .mb_block_row_ids)
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
  strata = NULL,
  row_ids = NULL
) {
  check_vector = function(x, name) {
    if (!is.atomic(x) || !is.null(dim(x)) || length(x) != n || anyNA(x)) {
      stop(sprintf(
        "`%s` must contain one non-missing value for every row.", name
      ), call. = FALSE)
    }
    # Design vectors are positional. Names are only checked, never used to
    # reorder, so that misaligned metadata fails instead of being permuted.
    if (!is.null(row_ids) && !is.null(names(x)) &&
      !identical(names(x), row_ids)) {
      stop(sprintf(
        paste0(
          "Names of `%s` do not match the block row names in order. ",
          "Reorder it by the block row names or remove its names with ",
          "`unname()`."
        ),
        name
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
    # Integer row groups in order of first appearance; drawing iterates them
    # in this order so a given seed always yields the same maps.
    row_groups = unname(split(
      seq_len(n), factor(strata_key, levels = unique(strata_key))
    ))
    group_sizes = lengths(row_groups)
    if (all(group_sizes < 2L)) {
      stop("No stratum contains two exchangeable rows.", call. = FALSE)
    }
    return(list(
      n = n,
      unit = NULL,
      within = NULL,
      strata = strata_key,
      unit_level = FALSE,
      n_units = n,
      row_groups = row_groups,
      group_sizes = group_sizes
    ))
  }

  unit_key = check_vector(exchangeability_unit, "exchangeability_unit")
  units = unique(unit_key)
  n_units = length(units)
  unit_id = match(unit_key, units)

  unit_strata = strata_key[match(seq_len(n_units), unit_id)]
  inconsistent_strata = logical(n_units)
  inconsistent_strata[unit_id[strata_key != unit_strata[unit_id]]] = TRUE
  if (any(inconsistent_strata)) {
    stop(sprintf(
      "Strata vary within exchangeability units: %s.",
      paste(units[inconsistent_strata], collapse = ", ")
    ), call. = FALSE)
  }

  repeated = tabulate(unit_id, nbins = n_units) > 1L
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
  within_id = match(within_key, unique(within_key))

  duplicate_positions = logical(n_units)
  duplicate_positions[unit_id[duplicated(cbind(unit_id, within_id))]] = TRUE
  if (any(duplicate_positions)) {
    stop(sprintf(
      "`within_unit` labels must be unique inside each unit: %s.",
      paste(units[duplicate_positions], collapse = ", ")
    ), call. = FALSE)
  }

  # Order the rows of every unit by position label. Units in one stratum share
  # the same label set, so column j of a stratum matrix is the same position
  # for every unit and whole trajectories can be moved by permuting rows.
  row_order = order(unit_id, within_id)
  unit_factor = factor(unit_id[row_order], levels = seq_len(n_units))
  rows_by_unit = split(row_order, unit_factor)
  unit_pattern = vapply(
    split(within_id[row_order], unit_factor), paste, character(1L),
    collapse = ","
  )
  stratum_levels = unique(unit_strata)
  units_by_stratum = split(
    seq_len(n_units), factor(unit_strata, levels = stratum_levels)
  )

  unit_matrices = lapply(seq_along(stratum_levels), function(index) {
    members = units_by_stratum[[index]]
    same_pattern = unit_pattern[members] == unit_pattern[[members[[1L]]]]
    if (any(!same_pattern)) {
      stop(sprintf(
        paste0(
          "Exchangeability units in stratum '%s' do not share the same ",
          "`within_unit` pattern: %s."
        ),
        stratum_levels[[index]],
        paste(units[members][!same_pattern], collapse = ", ")
      ), call. = FALSE)
    }
    matrix(
      unlist(rows_by_unit[members], use.names = FALSE),
      nrow = length(members),
      byrow = TRUE
    )
  })

  group_sizes = lengths(units_by_stratum, use.names = FALSE)
  if (all(group_sizes < 2L)) {
    stop("No stratum contains two exchangeable units.", call. = FALSE)
  }

  list(
    n = n,
    unit = unit_key,
    within = within_key,
    strata = strata_key,
    units = units,
    unit_strata = unit_strata,
    unit_level = TRUE,
    n_units = n_units,
    unit_matrices = unit_matrices,
    group_sizes = group_sizes
  )
}

.mb_draw_permutation_index = function(exchangeability) {
  index = seq_len(exchangeability$n)

  if (!isTRUE(exchangeability$unit_level)) {
    for (rows in exchangeability$row_groups) {
      index[rows] = rows[sample.int(length(rows), replace = FALSE)]
    }
    return(index)
  }

  # One unit-by-position matrix per stratum: permuting its rows reassigns
  # complete trajectories while keeping every position label in place.
  for (rows in exchangeability$unit_matrices) {
    source = rows[sample.int(nrow(rows), replace = FALSE), , drop = FALSE]
    index[as.vector(rows)] = as.vector(source)
  }
  index
}

# Log number of distinct permutation maps admitted by the design: the product
# of factorials of the exchangeable row or unit counts per stratum, for each
# independently permuted block. It bounds the number of distinct null values.
.mb_log_permutation_group_size = function(exchangeability, n_permuted_blocks) {
  n_permuted_blocks * sum(lfactorial(exchangeability$group_sizes))
}

.mb_permute_block_rows = function(block, index) {
  original_rows = .mb_block_row_ids(block)
  out = block[index, , drop = FALSE]
  if (!is.null(original_rows)) {
    rownames(out) = original_rows
  } else if (is.data.frame(out) && !inherits(out, c("data.table", "tbl_df"))) {
    # Restore automatic row names instead of carrying the shuffled positions;
    # data.tables and tibbles already return automatic row names.
    rownames(out) = NULL
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
#'   Supply data before any alignment-dependent preprocessing or tuning. Row
#'   names, when present, must be identical across blocks; automatic data-frame
#'   row names are treated as absent.
#' @param analysis Function taking the complete named block list. It must rerun
#'   the full data-dependent analysis and return one finite numeric statistic,
#'   or a list containing that value in `statistic`.
#' @param permute_blocks Names or indices of blocks to permute independently.
#'   The default permutes every block except the first.
#' @param n_perm Number of random permutations. The smallest possible sampled
#'   p-value is `1 / (n_perm + 1)`. Permutation maps are drawn with
#'   replacement from the design's permutation group, whose size is the
#'   product over strata of the factorial of the number of exchangeable rows or
#'   units, raised to the number of permuted blocks. With few exchangeable
#'   units or small strata this group, not `n_perm`, limits the practically
#'   attainable p-value to about `1 / |G|`; a warning of class
#'   `"mb_small_permutation_group"` is raised when `n_perm` is at least `|G|`.
#' @param exchangeability_unit Optional vector identifying independent units,
#'   in block row order. Whole units, not individual rows, are reassigned.
#' @param within_unit Required for repeated-row units; identifies positions
#'   such as visits that must be matched when whole trajectories are moved.
#'   Supplied in block row order.
#' @param strata Optional vector restricting row or unit permutations within
#'   design-valid strata, in block row order. It must be constant within an
#'   exchangeability unit.
#'
#'   The three design vectors are matched to rows by position. If a design
#'   vector is named and the blocks carry row names, the names must equal the
#'   block row names in the same order; mismatches are errors and vectors are
#'   never reordered.
#' @param alternative Direction of extremeness for the statistic.
#' @param null_center Explicit null centre required for a two-sided statistic.
#' @param seed Non-negative seed used to sample permutation maps. The caller's
#'   RNG kind and state are restored.
#' @param analysis_seed Non-negative seed reset before every call to
#'   `analysis()`. This makes a stochastic fitting/tuning algorithm a
#'   deterministic statistic. For such an analysis the seed is part of the
#'   definition of the statistic: different seeds define different statistics
#'   and can give materially different p-values, so fix and record it before
#'   looking at the data. Choosing among seeds after seeing results is an
#'   uncorrected multiple-testing procedure. The seed selects the
#'   `"L'Ecuyer-CMRG"` generator, so analyses that use forked parallelism
#'   (e.g. [parallel::mclapply()] with `mc.set.seed = TRUE`) remain
#'   reproducible. Outcome-dependent splits,
#'   including stratified folds, must be recreated inside `analysis()` for the
#'   permuted outcomes.
#' @param keep_null If `TRUE`, retain the sampled null statistics.
#' @param null_hypothesis Optional precise description of the scientific null.
#'
#' @return An object of class `mb_permutation_test` containing the observed
#'   statistic, corrected p-value, exceedance count, Monte Carlo precision,
#'   exchangeability metadata, and (optionally) the null distribution.
#'   Monte Carlo precision refers to the exceedance (tail) probability of the
#'   exhaustive permutation test, estimated without bias by
#'   `monte_carlo_tail_estimate = exceedances / n_perm`:
#'
#'   * `monte_carlo_conf_low`, `monte_carlo_conf_high`: exact Clopper-Pearson
#'     interval at level `monte_carlo_conf_level` (0.95). This is the precision
#'     summary to report; it stays informative when no sampled statistic is as
#'     extreme as the observed one.
#'   * `monte_carlo_standard_error`: binomial (Wald) standard error of the tail
#'     estimate. It degenerates at the boundary and is therefore `NA` when
#'     `exceedances` is `0` or `n_perm`.
#'
#'   `permutation_group_size` and `log_permutation_group_size` give the number
#'   of distinct permutation maps admitted by the design (see `n_perm`), an
#'   upper bound on the number of distinct null statistics; the former
#'   overflows to `Inf` for large designs.
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
    strata = strata,
    row_ids = .mb_block_row_ids(blocks[[1L]])
  )
  log_group_size = .mb_log_permutation_group_size(
    exchangeability, length(permute_blocks)
  )
  if (log_group_size <= log(n_perm) + sqrt(.Machine$double.eps)) {
    warning(warningCondition(
      sprintf(
        paste(
          "The exchangeability design admits only %s distinct permutation",
          "maps, which does not exceed n_perm = %d. Sampled maps repeat, so",
          "the p-value cannot be materially smaller than about 1/%s whatever",
          "n_perm is; more permutations only reduce Monte Carlo error."
        ),
        .mb_format_group_size(log_group_size),
        n_perm,
        .mb_format_group_size(log_group_size)
      ),
      class = "mb_small_permutation_group"
    ))
  }

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
      with_seed_local(analysis_seed, function() analysis(current_blocks),
        kind = "L'Ecuyer-CMRG"
      ),
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
    # The Wald standard error is zero at the boundary although the tail
    # probability is uncertain there; the exact interval is reported instead.
    mc_se = if (exceedances > 0L && exceedances < n_perm) {
      sqrt(tail_estimate * (1 - tail_estimate) / n_perm)
    } else {
      NA_real_
    }
    mc_conf_level = 0.95
    mc_ci = stats::binom.test(
      exceedances, n_perm, conf.level = mc_conf_level
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
      monte_carlo_conf_level = mc_conf_level,
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
      permutation_group_size = exp(log_group_size),
      log_permutation_group_size = log_group_size,
      seed = seed,
      analysis_seed = analysis_seed,
      null_distribution = if (isTRUE(keep_null)) null_distribution else NULL
    ), class = "mb_permutation_test")
  }

  with_seed_local(seed, run_test)
}

# Group sizes can overflow doubles; show them exactly when small.
.mb_format_group_size = function(log_size) {
  if (log_size < log(1e15)) {
    format(round(exp(log_size)), big.mark = ",", scientific = FALSE)
  } else {
    sprintf("exp(%.1f)", log_size)
  }
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
    "Monte Carlo %g%% CI for the exceedance probability: [%.4g, %.4g]\n",
    100 * (x$monte_carlo_conf_level %||% 0.95),
    x$monte_carlo_conf_low, x$monte_carlo_conf_high
  ))
  cat(sprintf(
    "Exchangeability: %s; units = %d; strata = %d\n",
    x$exchangeability_level, x$n_exchangeability_units, x$n_strata
  ))
  if (!is.null(x$log_permutation_group_size)) {
    cat(sprintf(
      "Permutation group size: %s distinct maps\n",
      .mb_format_group_size(x$log_permutation_group_size)
    ))
  }
  cat(sprintf("Seeds: permutation = %d; analysis = %d\n",
    x$seed, x$analysis_seed))
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

# `upper_p` optionally gives the structural width of each block (named like
# `blocks`), e.g. its numeric columns before data-dependent constant-column
# filtering. Budgets are then validated against sqrt(upper_p) and capped at
# sqrt(p) of the supplied blocks, where larger L1 budgets are nonbinding. The
# logical matrix attribute "capped" marks the capped entries.
.mb_prepare_c_matrix = function(blocks, c_matrix, ncomp, ncomp_missing, upper_p = NULL) {
  p = vapply(blocks, ncol, integer(1L))
  structural = !is.null(upper_p)
  if (!structural) {
    upper_p = p
  } else {
    if (is.numeric(upper_p) && !is.null(names(upper_p)) && !is.null(names(blocks))) {
      upper_p = upper_p[names(blocks)]
    }
    if (!is.numeric(upper_p) || length(upper_p) != length(blocks) ||
      anyNA(upper_p) || any(upper_p < p)) {
      stop("`upper_p` must give one structural width >= ncol(block) per block.",
        call. = FALSE)
    }
  }
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
    rep(sqrt(upper_p), ncomp), nrow = length(blocks), ncol = ncomp
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
  if (structural) {
    retained_upper = matrix(
      rep(sqrt(p), ncomp), nrow = length(blocks), ncol = ncomp
    )
    capped = c_matrix > retained_upper
    c_matrix[capped] = retained_upper[capped]
    dimnames(capped) = dimnames(c_matrix)
    attr(c_matrix, "capped") = capped
  }
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

# Per-component convergence flags of a native fit, or NA when the solver does
# not report convergence. Exact `[[` avoids partial matching of list names.
.mb_fit_converged = function(fit, ncomp) {
  converged = fit[["converged"]]
  if (is.null(converged)) {
    return(rep(NA, ncomp))
  }
  converged = as.logical(converged)
  if (length(converged) == 1L) {
    converged = rep(converged, ncomp)
  }
  if (length(converged) != ncomp) {
    return(rep(NA, ncomp))
  }
  converged
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
#' * `"global_lc1"`: the first-component cross-block objective evaluated at the
#'   weights returned by the deterministic MB-sPLS solver. The criterion is
#'   non-convex, so this is the objective at the solver's solution, not
#'   necessarily at the global maximum;
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
#' The `*_sum` statistics refit every component on every permutation. The
#' `*_lc1` statistics depend only on the first component, which sequential
#' extraction fits before any deflation, so permutations refit LC1 only; with
#' `ncomp > 1`, the later-component statistics are taken from one additional
#' descriptive fit to the observed data.
#'
#' Inputs should be raw finite numeric blocks. Optional z-standardisation is
#' refitted inside every analysis call. `ncomp`, `c_matrix`, the statistic,
#' `max_iter`, `tol`, and all other settings must be specified independently of
#' the tested block alignment. If any were selected from the observed
#' association, use [mb_permutation_test()] with a callback that repeats that
#' selection inside every permutation. The result is therefore labelled a
#' fixed-specification test, not a complete-analysis test.
#'
#' A warning of class `"mb_nonconverged_fit"` is raised when the native solver
#' reports that an observed component fit did not converge within `max_iter`;
#' the number of non-converged permutation refits is returned. Both remain
#' deterministic functions of the data, so the p-value stays valid for the
#' statistic as computed, but that statistic is then not a converged objective.
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
#' @param analysis_seed Non-negative seed reset before every observed and
#'   permuted fit. The native MB-sPLS solver is initialised deterministically,
#'   so the seed does not change the fitted statistic; it is recorded so the
#'   call is a complete, reproducible specification.
#'
#' @return An `mbspls_permutation_test` object (inheriting from
#'   `mb_permutation_test`). It contains one corrected global p-value, the
#'   observed per-component objectives and statistics for descriptive use,
#'   `refit_ncomp` (components refitted per permutation), `observed_converged`
#'   (per-component convergence of the observed fit, `NA` if the solver does
#'   not report it), `n_nonconverged_permutations`, and a `validity_scope`.
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
  n_perm = .mb_assert_scalar_integer(n_perm, "n_perm")
  analysis_seed = .mb_assert_scalar_integer(
    analysis_seed, "analysis_seed", lower = 0L
  )
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

  fit_components = function(current_blocks, component_budgets) {
    k = ncol(component_budgets)
    fit_blocks = if (isTRUE(standardize)) {
      .mb_standardize_inference_blocks(current_blocks)
    } else {
      current_blocks
    }
    fit = cpp_mbspls_multi_lv_cmatrix(
      X_blocks = fit_blocks,
      c_matrix = component_budgets,
      max_iter = max_iter,
      tol = tol,
      spearman = identical(correlation_method, "spearman"),
      do_perm = FALSE,
      n_perm = 1L,
      alpha = 0.05,
      frobenius = identical(performance_metric, "frobenius")
    )
    component_objectives = as.numeric(fit$objective)
    if (length(component_objectives) != k ||
      any(!is.finite(component_objectives))) {
      stop(
        "MB-sPLS did not return every requested finite component objective.",
        call. = FALSE
      )
    }
    if (!is.matrix(fit$T_mat) ||
      !identical(dim(fit$T_mat), c(nrow(fit_blocks[[1L]]), k * length(blocks))) ||
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
        ncomp = k,
        block_names = names(blocks),
        target_block = target_block,
        correlation_method = correlation_method,
        performance_metric = performance_metric
      )
    } else {
      component_objectives
    }

    list(
      component_statistics = component_statistics,
      component_objectives = component_objectives,
      converged = .mb_fit_converged(fit, k)
    )
  }

  # LC1 statistics need only the first component, which sequential
  # extraction fits before any deflation.
  uses_all_components = statistic %in% c("global_sum", "target_sum")
  test_c_matrix = if (uses_all_components) {
    c_matrix
  } else {
    c_matrix[, 1L, drop = FALSE]
  }

  descriptive = NULL
  if (ncol(test_c_matrix) < ncomp) {
    descriptive = tryCatch(
      with_seed_local(analysis_seed, function() {
        fit_components(blocks, c_matrix)
      }),
      error = function(error) {
        stop(sprintf(
          "Observed descriptive MB-sPLS fit failed: %s",
          conditionMessage(error)
        ), call. = FALSE)
      }
    )
  }

  # Convergence of every analysis call; the first call is the observed fit.
  convergence = new.env(parent = emptyenv())
  convergence$calls = 0L
  convergence$flags = rep(NA, n_perm + 1L)

  analysis = function(current_blocks) {
    fit = fit_components(current_blocks, test_c_matrix)
    convergence$calls = convergence$calls + 1L
    convergence$flags[[convergence$calls]] = all(fit$converged)
    test_statistic = if (uses_all_components) {
      sum(fit$component_statistics)
    } else {
      fit$component_statistics[[1L]]
    }

    c(list(statistic = as.numeric(test_statistic)), fit)
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

  observed = result$observed_result
  if (!is.null(descriptive)) {
    tested_value = descriptive$component_statistics[[1L]]
    if (abs(tested_value - result$statistic) >
      sqrt(.Machine$double.eps) * max(1, abs(result$statistic))) {
      stop(
        "Internal error: the descriptive LC1 fit differs from the tested statistic.",
        call. = FALSE
      )
    }
    observed = c(list(statistic = result$statistic), descriptive)
  }
  permutation_converged = convergence$flags[-1L]

  result$method = "Fixed-specification MB-sPLS omnibus Monte Carlo permutation test"
  result$observed_result = observed
  result$statistic_name = statistic
  result$component_statistics = observed$component_statistics
  result$component_objectives = observed$component_objectives
  result$c_matrix = c_matrix
  result$ncomp = ncomp
  result$refit_ncomp = ncol(test_c_matrix)
  result$target_block = target_block
  result$standardize = standardize
  result$correlation_method = correlation_method
  result$performance_metric = performance_metric
  result$max_iter = max_iter
  result$tol = tol
  result$observed_converged = stats::setNames(
    observed$converged, colnames(c_matrix)
  )
  result$n_nonconverged_permutations = if (all(is.na(permutation_converged))) {
    NA_integer_
  } else {
    sum(!permutation_converged, na.rm = TRUE)
  }
  result$validity_scope = paste(
    "One omnibus p-value for the stated block-independence null.",
    "Component statistics after LC1 are descriptive and are not rank-null p-values.",
    "Valid only if the statistic, ncomp, c_matrix, standardisation, and solver",
    "settings were fixed independently of the tested alignment; otherwise",
    "repeat the selection inside mb_permutation_test()."
  )
  class(result) = c("mbspls_permutation_test", class(result))

  not_converged = !is.na(result$observed_converged) & !result$observed_converged
  if (any(not_converged)) {
    warning(warningCondition(
      sprintf(
        paste(
          "The observed MB-sPLS fit did not converge within max_iter = %d",
          "for %s; the reported objective is the value after the final",
          "iteration. %s of %d permutation refits did not converge."
        ),
        max_iter,
        paste(names(result$observed_converged)[not_converged], collapse = ", "),
        if (is.na(result$n_nonconverged_permutations)) {
          "An unknown number"
        } else {
          as.character(result$n_nonconverged_permutations)
        },
        n_perm
      ),
      class = "mb_nonconverged_fit"
    ))
  }
  result
}

#' @export
print.mbspls_permutation_test = function(x, ...) {
  NextMethod()
  cat(sprintf(
    "Statistic: %s; ncomp = %d (refitted per permutation: %d); standardize = %s%s\n",
    x$statistic_name, x$ncomp, x$refit_ncomp %||% x$ncomp, x$standardize,
    if (is.null(x$target_block)) "" else paste0("; target block = ", x$target_block)
  ))
  if (isTRUE(x$n_nonconverged_permutations > 0L) ||
    any(!x$observed_converged, na.rm = TRUE)) {
    cat(sprintf(
      "Convergence: observed fit %s; %d of %d permutation refits did not converge\n",
      if (any(!x$observed_converged, na.rm = TRUE)) "did not converge" else "converged",
      x$n_nonconverged_permutations, x$n_perm
    ))
  }
  cat("Scope: ", x$validity_scope, "\n", sep = "")
  invisible(x)
}

# Signed correlations of the active score pairs and the LC statistic. With
# reference signs the statistic is the mean sign-oriented correlation, which
# is large only for associations in the discovery direction.
.mb_lc_score_statistic = function(
  score_blocks,
  pairs,
  reference_signs,
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
  values = correlations[cbind(pairs[1L, ], pairs[2L, ])]
  if (!length(values) || any(!is.finite(values))) {
    stop(
      "An LC confirmation statistic is undefined because a score is degenerate.",
      call. = FALSE
    )
  }
  statistic = if (!is.null(reference_signs)) {
    mean(values * reference_signs)
  } else if (identical(performance_metric, "frobenius")) {
    sqrt(sum(values^2))
  } else {
    mean(abs(values))
  }
  list(statistic = statistic, correlations = values)
}

# Resolve `reference_signs` to one sign vector per LC, aligned with the active
# score pairs. Pair names may list the two blocks in either order.
.mb_resolve_reference_signs = function(
  reference_signs,
  block_names,
  active_pairs,
  component_names
) {
  if (is.null(reference_signs)) {
    return(NULL)
  }
  n_components = length(component_names)
  if (is.list(reference_signs)) {
    if (length(reference_signs) != n_components) {
      stop(sprintf(
        "A `reference_signs` list must contain one sign vector per LC (%d).",
        n_components
      ), call. = FALSE)
    }
    if (!is.null(names(reference_signs))) {
      if (!setequal(names(reference_signs), component_names) ||
        anyDuplicated(names(reference_signs))) {
        stop(
          "Names of a `reference_signs` list must be the confirmation LC names.",
          call. = FALSE
        )
      }
      reference_signs = reference_signs[component_names]
    }
  } else {
    reference_signs = rep(list(reference_signs), n_components)
  }

  all_pairs = utils::combn(length(block_names), 2L)
  first = block_names[all_pairs[1L, ]]
  second = block_names[all_pairs[2L, ]]
  lookup_keys = c(paste(first, second, sep = ":"), paste(second, first, sep = ":"))
  lookup_pairs = rep(seq_len(ncol(all_pairs)), 2L)
  if (anyDuplicated(lookup_keys)) {
    stop(
      "Score block names are ambiguous in `reference_signs` pair names; rename the blocks.",
      call. = FALSE
    )
  }
  active_index = match(
    paste(block_names[active_pairs[1L, ]], block_names[active_pairs[2L, ]],
      sep = ":"),
    lookup_keys
  )

  signs = lapply(seq_len(n_components), function(component) {
    value = reference_signs[[component]]
    if (!is.numeric(value) || !is.null(dim(value)) || !length(value) ||
      anyNA(value) || any(!value %in% c(-1, 1))) {
      stop("`reference_signs` values must be -1 or 1.", call. = FALSE)
    }
    keys = names(value)
    if (is.null(keys) || anyNA(keys) || any(!nzchar(keys))) {
      stop(
        "`reference_signs` must be named by block pair, e.g. \"blockA:blockB\".",
        call. = FALSE
      )
    }
    matched = lookup_pairs[match(keys, lookup_keys)]
    if (anyNA(matched)) {
      stop(sprintf(
        "Unknown block pairs in `reference_signs`: %s.",
        paste(keys[is.na(matched)], collapse = ", ")
      ), call. = FALSE)
    }
    if (anyDuplicated(matched)) {
      stop("Every block pair may appear only once in `reference_signs`.",
        call. = FALSE)
    }
    pair_signs = rep(NA_real_, ncol(all_pairs))
    pair_signs[matched] = as.numeric(value)
    selected = pair_signs[active_index]
    if (anyNA(selected)) {
      stop(sprintf(
        "`reference_signs` must cover every block pair in the statistic; missing: %s.",
        paste(lookup_keys[active_index[is.na(selected)]], collapse = ", ")
      ), call. = FALSE)
    }
    selected
  })
  stats::setNames(signs, component_names)
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
#' adjustment is always reported across the complete supplied LC family, so no
#' LC can be significant unless `n_LC / (n_perm + 1) <= alpha`; a warning of
#' class `"mb_insufficient_permutations"` is raised otherwise.
#'
#' Frozen weights fix the sign of every block score, so the sign of each
#' pairwise score correlation is meaningful. With `reference_signs`, each LC is
#' tested directionally: the statistic is the mean of the active pairwise
#' correlations multiplied by their expected signs from discovery, and the
#' one-sided test rejects only for association in the discovery direction.
#' Replication additionally requires the observed statistic to be positive,
#' i.e. the sign-oriented correlations to agree with discovery on average:
#' restricted designs (strata or whole units) keep between-stratum or
#' between-unit structure fixed under the null, so a significant upper-tail
#' p-value can occur for a pooled correlation of the opposite sign. It then
#' shows dependence relative to that design, not replication of the discovery
#' sign. `replicated` combines both conditions. Without `reference_signs`, the
#' statistic is unsigned (`performance_metric`), so an association in either
#' direction, including one opposite to discovery, can be significant; the
#' result then establishes dependence, not directional replication. The
#' observed signed pairwise correlations are always returned.
#'
#' Row, stratum, and complete-unit exchangeability follow
#' [mb_permutation_test()]. With a single `permute_blocks` entry, only
#' correlations involving that target score block enter the statistic. With
#' multiple entries, every score-pair relation broken by the permutation enters
#' the statistic. The null requires independence of the permuted score blocks
#' from each other and the joint fixed score blocks within the exchangeability
#' design; zero correlation alone is insufficient.
#'
#' @param scores Named list of finite numeric confirmation-score matrices or
#'   data frames. Rows are aligned observations, columns are the same frozen
#'   LCs in the same order, and blocks contain their respective scores.
#' @param independent_confirmation Must be explicitly `TRUE`, asserting that
#'   the confirmation observations were not used to learn or select any part of
#'   the score transformation or LC family.
#' @param reference_signs Optional expected signs (`1` or `-1`) of the pairwise
#'   block-score correlations, fixed from the discovery data (for example the
#'   signs of the discovery-sample score correlations). Either one numeric
#'   vector named by block pair, such as `c("imaging:clinical" = 1)`, used for
#'   every LC, or a list with one such vector per LC (named by LC or in LC
#'   order). Every pair entering the statistic must be covered; pairs of fixed
#'   blocks are ignored. When supplied, the test is directional and requires
#'   `performance_metric = "mac"`.
#' @param alpha Family-wise significance level used with Holm-adjusted p-values.
#' @inheritParams mb_permutation_test
#' @inheritParams mbspls_permutation_test
#'
#' @return An `mb_lc_confirmation_test` object. `results` has one row per LC
#'   with the fixed-score statistic, exceedance count `b`, raw sampled
#'   permutation p-value, Holm-adjusted p-value, the exact 95% Clopper-Pearson
#'   interval for the exceedance probability (`monte_carlo_conf_low`,
#'   `monte_carlo_conf_high`, the Monte Carlo precision to report), and the
#'   adjusted decision `significant_holm`. With `reference_signs`,
#'   `direction_agrees` states whether the observed statistic (the mean
#'   sign-oriented correlation) is positive, and `replicated` is
#'   `significant_holm & direction_agrees`, the replication decision; both are
#'   `NA` without `reference_signs`. `pairwise_correlations` lists the observed
#'   signed correlation of every active score pair per LC, with its reference,
#'   sign-oriented value and sign agreement when `reference_signs` is
#'   supplied. `direction` is
#'   `"expected_sign"` or `"either"`, and `statistic_name` names the statistic.
#'   `log_permutation_group_size` describes the design shared by all LCs. The
#'   per-LC [mb_permutation_test()] objects are kept in `tests`.
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
  keep_null = FALSE,
  reference_signs = NULL
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
  block_names = names(scores)
  permute_blocks = .mb_resolve_block_selection(
    permute_blocks, block_names
  )
  n_perm = .mb_assert_scalar_integer(n_perm, "n_perm")
  seed = .mb_assert_scalar_integer(seed, "seed", lower = 0L)
  alpha = .mb_assert_probability(alpha, "alpha")
  correlation_method = match.arg(correlation_method)
  performance_metric = match.arg(performance_metric)
  if (length(keep_null) != 1L || is.na(keep_null) || !is.logical(keep_null)) {
    stop("`keep_null` must be one non-missing logical value.", call. = FALSE)
  }

  pairs = utils::combn(length(block_names), 2L)
  active = apply(pairs, 2L, function(pair) {
    any(block_names[pair] %in% permute_blocks)
  })
  pairs = pairs[, active, drop = FALSE]
  directional = !is.null(reference_signs)
  if (directional && identical(performance_metric, "frobenius")) {
    stop(
      paste(
        "A directional test with `reference_signs` uses the mean sign-oriented",
        "correlation; set `performance_metric = \"mac\"`."
      ),
      call. = FALSE
    )
  }
  signs = .mb_resolve_reference_signs(
    reference_signs, block_names, pairs, component_names
  )

  component_seeds = with_seed_local(seed, function() {
    sample.int(.Machine$integer.max, length(component_names), replace = FALSE)
  })
  # The design, and hence its permutation group, is shared by all LCs, so
  # the small-group warning is emitted once per LC family.
  deferred = new.env(parent = emptyenv())
  deferred$small_group_warning = NULL
  tests = withCallingHandlers(
    lapply(seq_along(component_names), function(component) {
      component_blocks = lapply(scores, function(x) {
        x[, component, drop = FALSE]
      }) |>
        stats::setNames(block_names)
      analysis = function(current_scores) {
        .mb_lc_score_statistic(
          score_blocks = current_scores,
          pairs = pairs,
          reference_signs = signs[[component]],
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
    }),
    mb_small_permutation_group = function(condition) {
      if (is.null(deferred$small_group_warning)) {
        deferred$small_group_warning = condition
      }
      invokeRestart("muffleWarning")
    }
  )
  if (!is.null(deferred$small_group_warning)) {
    warning(deferred$small_group_warning)
  }

  raw_p = vapply(tests, `[[`, numeric(1L), "p_value")
  holm_p = stats::p.adjust(raw_p, method = "holm")
  results = data.frame(
    component = component_names,
    statistic = vapply(tests, `[[`, numeric(1L), "statistic"),
    exceedances = vapply(tests, `[[`, integer(1L), "exceedances"),
    p_value_raw = raw_p,
    p_value_holm = as.numeric(holm_p),
    monte_carlo_conf_low = vapply(
      tests, `[[`, numeric(1L), "monte_carlo_conf_low"
    ),
    monte_carlo_conf_high = vapply(
      tests, `[[`, numeric(1L), "monte_carlo_conf_high"
    ),
    significant_holm = holm_p <= alpha,
    stringsAsFactors = FALSE
  )
  # A restricted null can make a pooled correlation of the opposite sign
  # significant in the upper tail, so replication also needs sign agreement.
  results$direction_agrees = if (directional) results$statistic > 0 else NA
  results$replicated = if (directional) {
    results$significant_holm & results$direction_agrees
  } else {
    NA
  }

  pairwise_correlations = do.call(rbind, lapply(seq_along(tests), function(component) {
    correlation = tests[[component]]$observed_result$correlations
    reference_sign = if (directional) signs[[component]] else NA_real_
    data.frame(
      component = component_names[[component]],
      block_1 = block_names[pairs[1L, ]],
      block_2 = block_names[pairs[2L, ]],
      correlation = correlation,
      reference_sign = reference_sign,
      oriented_correlation = correlation * reference_sign,
      agrees = if (directional) correlation * reference_sign > 0 else NA,
      stringsAsFactors = FALSE
    )
  }))

  minimum_holm_p = length(component_names) / (n_perm + 1)
  if (minimum_holm_p > alpha) {
    warning(warningCondition(
      sprintf(
        paste(
          "With %d LCs and n_perm = %d, the smallest attainable Holm-adjusted",
          "p-value is %.3g > alpha = %.3g, so no LC can be significant.",
          "Increase n_perm."
        ),
        length(component_names), n_perm, minimum_holm_p, alpha
      ),
      class = "mb_insufficient_permutations"
    ))
  }

  scope_tail = paste(
    "Not a population-rank test and not valid after reusing confirmation",
    "data for fitting or selection."
  )
  structure(list(
    method = if (directional) {
      paste(
        "Independent-confirmation directional permutation tests for frozen",
        "MB-sPLS LC score associations"
      )
    } else {
      paste(
        "Independent-confirmation permutation tests for frozen MB-sPLS LC",
        "score dependence in either direction"
      )
    },
    results = results,
    pairwise_correlations = pairwise_correlations,
    tests = tests,
    direction = if (directional) "expected_sign" else "either",
    reference_signs = signs,
    alpha = alpha,
    adjustment = "holm",
    minimum_raw_p_value = 1 / (n_perm + 1),
    n_perm = n_perm,
    log_permutation_group_size = tests[[1L]]$log_permutation_group_size,
    permute_blocks = permute_blocks,
    fixed_blocks = setdiff(block_names, permute_blocks),
    correlation_method = correlation_method,
    performance_metric = performance_metric,
    statistic_name = if (directional) {
      "mean_oriented_correlation"
    } else if (identical(performance_metric, "frobenius")) {
      "frobenius_norm_correlation"
    } else {
      "mean_absolute_correlation"
    },
    seed = seed,
    component_seeds = component_seeds,
    validity_scope = if (directional) {
      paste(
        "Directional confirmation of fixed learned score associations:",
        "one-sided tests of the mean sign-oriented correlation in the",
        "discovery direction. An LC replicates only if it is significant and",
        "its observed statistic is positive (`replicated`).", scope_tail
      )
    } else {
      paste(
        "Direction-agnostic dependence of fixed learned scores: association",
        "in either direction, including opposite to discovery, can be",
        "significant. Not a directional replication test; supply",
        "`reference_signs` from discovery for that.", scope_tail
      )
    }
  ), class = "mb_lc_confirmation_test")
}

#' @export
print.mb_lc_confirmation_test = function(x, ...) {
  cat(x$method, "\n", sep = "")
  if (identical(x$direction, "expected_sign")) {
    cat("Direction: one-sided in the discovery direction (reference signs)\n")
  } else {
    cat(paste(
      "Direction: either (unsigned statistic); not a directional",
      "replication test\n"
    ))
  }
  results = x$results
  if (!identical(x$direction, "expected_sign")) {
    results = results[, setdiff(names(results), c("direction_agrees", "replicated")), drop = FALSE]
  }
  print(results, row.names = FALSE)
  if (!is.null(x$pairwise_correlations)) {
    pairwise = x$pairwise_correlations
    if (!identical(x$direction, "expected_sign")) {
      pairwise = pairwise[, c("component", "block_1", "block_2", "correlation")]
    }
    cat("Observed signed score correlations:\n")
    print(pairwise, row.names = FALSE)
  }
  cat(paste(
    "Monte Carlo precision: exact 95% Clopper-Pearson interval for the",
    "exceedance probability b/B.\n"
  ))
  cat("Scope: ", x$validity_scope, "\n", sep = "")
  invisible(x)
}
