#' Extract mean bootstrap weights with CI/frequency filtering
#'
#' @param model A single mbspls model containing $weights_boot_draws and $weights.
#' @param component Integer LC index (e.g., 1).
#' @param filter_method One of c("ci","frequency").
#'   - "ci": keep features whose CI is >= 0 or <= 0.
#'   - "frequency": keep features with selection freq >= filter_level using model$weights_selectfreq.
#' @param filter_level Numeric threshold:
#'   - if filter_method == "ci": confidence level in `(0,1)`, default 0.95.
#'   - if filter_method == "frequency": minimum freq in `[0,1]`, default 0.5.
#' @return A tibble with rows (block, feature) that pass the filter, including
#'         mean, ci_low, ci_high, and freq (if available), sorted by block and |mean| desc.
#'
#' @export
mbspls_extract_bootstrap_means <- function(
  model,
  component = 1,
  filter_method = c("ci","frequency"),
  filter_level = NULL
) {
  requireNamespace("dplyr")
  requireNamespace("tibble")

  filter_method <- match.arg(filter_method)
  comp_lab <- sprintf("LC_%02d", component)

  # defaults for filter_level
  if (is.null(filter_level)) {
    filter_level <- if (filter_method == "ci") 0.95 else 0.5
  }

  # sanity
  if (is.null(model$weights_boot_draws)) {
    stop("Model must include $weights_boot_draws (long data frame with columns ",
         "`component`,`block`,`feature`,`replicate`,`weight`).")
  }
  if (is.null(model$weights) || is.null(model$weights[[comp_lab]])) {
    stop("Model must include reference $weights for ", comp_lab, " for sign alignment.")
  }

  # --- bootstrap draws for the requested component
  boot_df <- as.data.frame(model$weights_boot_draws)
  boot_df <- boot_df[boot_df$component == comp_lab, , drop = FALSE]
  if (!nrow(boot_df)) stop("No bootstrap draws for ", comp_lab, ".")

  # --- reference weights for per-replicate sign alignment
  blocks_present <- unique(boot_df$block)
  ref_tbl <- dplyr::bind_rows(lapply(blocks_present, function(b) {
    w <- model$weights[[comp_lab]][[b]]
    if (is.null(w)) stop("Block '", b, "' not found in model$weights[[", comp_lab, "]].")
    tibble::tibble(block = b, feature = names(w), refw = as.numeric(w))
  }))

  # join reference and compute replicate-wise sign (align by correlation with ref)
  boot_df <- dplyr::left_join(boot_df, ref_tbl, by = c("block","feature"))
  sign_df <- boot_df |>
    dplyr::group_by(replicate, block) |>
    dplyr::summarise(
      corr = suppressWarnings(stats::cor(weight, refw, use = "complete.obs")),
      .groups = "drop"
    ) |>
    dplyr::mutate(sign = dplyr::if_else(is.na(corr) | corr >= 0, 1, -1))

  boot_df <- dplyr::left_join(boot_df, sign_df[, c("replicate","block","sign")],
                              by = c("replicate","block")) |>
    dplyr::mutate(weight_aligned = sign * weight)

  # --- summarise means and (optionally) CIs
  # CI level (for "ci" filter) -> quantile probs
  cl <- if (filter_method == "ci") {
    if (!is.numeric(filter_level) || filter_level <= 0 || filter_level >= 1)
      stop("For filter_method='ci', filter_level must be in (0,1), e.g., 0.95.")
    filter_level
  } else 0.95
  lower_p <- (1 - cl)/2
  upper_p <- 1 - lower_p

  sum_df <- boot_df |>
    dplyr::group_by(block, feature) |>
    dplyr::summarise(
      mean   = mean(weight_aligned, na.rm = TRUE),
      ci_low = stats::quantile(weight_aligned, lower_p, na.rm = TRUE, names = FALSE),
      ci_high= stats::quantile(weight_aligned, upper_p, na.rm = TRUE, names = FALSE),
      .groups = "drop"
    )

  # --- frequency table (if available)
  freq_tbl <- NULL
  if (!is.null(model$weights_selectfreq)) {
    freq_tbl <- model$weights_selectfreq
    if (inherits(freq_tbl, "data.table")) freq_tbl <- as.data.frame(freq_tbl)
    if (all(c("component","block","feature","freq") %in% names(freq_tbl))) {
      freq_tbl <- freq_tbl[freq_tbl$component == comp_lab,
                           c("block","feature","freq"), drop = FALSE]
    } else {
      freq_tbl <- NULL
    }
  }

  out <- dplyr::left_join(sum_df, freq_tbl, by = c("block","feature"))

  # --- apply filter
  if (filter_method == "ci") {
    out <- out |>
      dplyr::filter(ci_low >= 0 | ci_high <= 0) |>
      dplyr::filter(abs(mean) > 1e-3)  # drop near-zeros
  } else { # frequency
    if (is.null(out$freq)) {
      stop("Requested filter_method='frequency' but model$weights_selectfreq is missing ",
           "or does not have columns component/block/feature/freq.")
    }
    if (!is.numeric(filter_level) || filter_level < 0 || filter_level > 1) {
      stop("For filter_method='frequency', filter_level must be in [0,1], e.g., 0.5.")
    }
    out <- dplyr::filter(out, freq >= filter_level)
  }

  # order: by block then |mean| desc
  out <- out |>
    dplyr::arrange(block, dplyr::desc(abs(mean))) |>
    dplyr::mutate(component = comp_lab, .before = 1)

  data.table::as.data.table(out)
}
