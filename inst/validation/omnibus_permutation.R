# Targeted calibration and power regression for mbspls_permutation_test().
#
# Run against the installed package, for example:
#   Rscript -e 'library(mlr3mbspls); source(system.file("validation", "omnibus_permutation.R", package = "mlr3mbspls"))'
#
# This simulation checks the implemented calculation in small reference
# scenarios. It is not evidence that arbitrary exchangeability designs or
# analysis callbacks are valid.

suppressPackageStartupMessages(library(mlr3mbspls))

simulation_count = as.integer(Sys.getenv(
  "MBSPLS_VALIDATION_N_SIM", unset = "200"
))
permutation_count = as.integer(Sys.getenv(
  "MBSPLS_VALIDATION_N_PERM", unset = "99"
))
if (!is.finite(simulation_count) || simulation_count < 20L ||
  !is.finite(permutation_count) || permutation_count < 19L) {
  stop("Validation counts are invalid.", call. = FALSE)
}

set.seed(20260831)

run_independent = function(iteration, signal = FALSE) {
  n = 36L
  latent = stats::rnorm(n)
  if (signal) {
    block_a = cbind(
      latent + stats::rnorm(n, sd = 0.5),
      stats::rnorm(n)
    )
    block_b = cbind(
      latent + stats::rnorm(n, sd = 0.5),
      stats::rnorm(n)
    )
  } else {
    block_a = matrix(stats::rnorm(n * 2L), nrow = n)
    block_b = matrix(stats::rnorm(n * 2L), nrow = n)
  }

  mbspls_permutation_test(
    blocks = list(a = block_a, b = block_b),
    statistic = "global_lc1",
    n_perm = permutation_count,
    max_iter = 60L,
    seed = 10000L + iteration,
    analysis_seed = 91L,
    keep_null = FALSE
  )$p_value
}

run_grouped_null = function(iteration) {
  n_unit = 18L
  unit = rep(seq_len(n_unit), each = 2L)
  visit = rep(c("baseline", "followup"), n_unit)
  visit_effect = rep(c(-1, 1), n_unit)
  unit_a = stats::rnorm(n_unit)
  unit_b = stats::rnorm(n_unit)
  block_a = cbind(
    rep(unit_a, each = 2L) + 1.5 * visit_effect +
      stats::rnorm(2L * n_unit, sd = 0.5),
    stats::rnorm(2L * n_unit)
  )
  block_b = cbind(
    rep(unit_b, each = 2L) + 1.5 * visit_effect +
      stats::rnorm(2L * n_unit, sd = 0.5),
    stats::rnorm(2L * n_unit)
  )

  mbspls_permutation_test(
    blocks = list(a = block_a, b = block_b),
    statistic = "global_lc1",
    n_perm = permutation_count,
    exchangeability_unit = unit,
    within_unit = visit,
    max_iter = 60L,
    seed = 30000L + iteration,
    analysis_seed = 71L,
    keep_null = FALSE
  )$p_value
}

run_confirmation = function(iteration, signal = FALSE) {
  n = 48L
  ncomp = 3L
  block_a = matrix(stats::rnorm(n * ncomp), nrow = n)
  block_b = matrix(stats::rnorm(n * ncomp), nrow = n)
  if (signal) {
    latent = stats::rnorm(n)
    block_a[, 1L] = latent + stats::rnorm(n, sd = 0.35)
    block_b[, 1L] = latent + stats::rnorm(n, sd = 0.35)
  }
  colnames(block_a) = colnames(block_b) = paste0("LC", seq_len(ncomp))

  result = mb_lc_confirmation_test(
    scores = list(a = block_a, b = block_b),
    independent_confirmation = TRUE,
    permute_blocks = "b",
    n_perm = permutation_count,
    seed = 50000L + iteration,
    keep_null = FALSE
  )
  if (signal) {
    result$results$p_value_holm[[1L]]
  } else {
    min(result$results$p_value_holm)
  }
}

null_p = vapply(seq_len(simulation_count), run_independent, numeric(1L))
signal_p = vapply(seq_len(max(50L, simulation_count %/% 2L)), function(i) {
  run_independent(i + 1000L, signal = TRUE)
}, numeric(1L))
grouped_null_p = vapply(seq_len(simulation_count), run_grouped_null,
  numeric(1L))
confirmation_null_p = vapply(
  seq_len(simulation_count), run_confirmation, numeric(1L)
)
confirmation_signal_p = vapply(
  seq_len(max(50L, simulation_count %/% 2L)),
  function(i) run_confirmation(i + 2000L, signal = TRUE),
  numeric(1L)
)

summarize_rate = function(p_values, label) {
  rejected = sum(p_values <= 0.05)
  interval = stats::binom.test(rejected, length(p_values))$conf.int
  data.frame(
    check = label,
    estimate = rejected / length(p_values),
    lower_95 = interval[[1L]],
    upper_95 = interval[[2L]],
    n = length(p_values),
    minimum_p = min(p_values),
    zero_p = sum(p_values == 0),
    stringsAsFactors = FALSE
  )
}

results = rbind(
  summarize_rate(null_p, "iid null rejection rate"),
  summarize_rate(grouped_null_p, "whole-unit null rejection rate"),
  summarize_rate(signal_p, "omnibus strong-signal rejection rate"),
  summarize_rate(
    confirmation_null_p,
    "confirmation-family null rejection rate"
  ),
  summarize_rate(
    confirmation_signal_p,
    "confirmation LC1 strong-signal rejection rate"
  )
)
print(results, row.names = FALSE, digits = 4)

minimum_allowed = 1 / (permutation_count + 1)
stopifnot(
  all(c(
    null_p, signal_p, grouped_null_p,
    confirmation_null_p, confirmation_signal_p
  ) >= minimum_allowed),
  !any(c(
    null_p, signal_p, grouped_null_p,
    confirmation_null_p, confirmation_signal_p
  ) == 0),
  mean(null_p <= 0.05) <= 0.10,
  mean(grouped_null_p <= 0.05) <= 0.10,
  mean(signal_p <= 0.05) >= 0.80,
  mean(confirmation_null_p <= 0.05) <= 0.10,
  mean(confirmation_signal_p <= 0.05) >= 0.80
)
