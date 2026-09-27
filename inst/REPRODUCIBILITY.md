# Reproducibility protocol

1. Save immutable participant identifiers and all outer/inner split indices.
2. Fit every learned preprocessing step only on analysis-fold observations.
3. Store fitted preprocessing state and the exact predictor names and order,
   including the resolved block columns and the training means used for
   centring (for example `$state$blocks` and `$state$center` of
   `PipeOpMBsPLS`).
4. Record every seed and know what it controls:
   - MB-sPLS fits are deterministic: they start from a fixed
     cross-covariance initialisation and draw no random numbers. `seed_train`
     only affects the permutations of the training diagnostic, and
     `analysis_seed` does not change an MB-sPLS statistic.
   - `seed_bootstrap` defaults to `NULL`, so bootstrap replicates draw from the
     session RNG and are reproducible after `set.seed()`. Sequential and
     parallel runs then use different draws. Set `seed_bootstrap` to give every
     replicate its own L'Ecuyer-CMRG stream, independent of `workers`; the
     streams are stored in the fitted state (`rng_streams`).
   - `mb_permutation_test()`, `mbspls_permutation_test()` and
     `mb_lc_confirmation_test()` take a permutation `seed`.
     `mb_permutation_test()` also resets `analysis_seed` before every analysis
     call, with the L'Ecuyer-CMRG generator, so callbacks that use forked
     parallelism (`parallel::mclapply()`) stay reproducible. For a stochastic
     callback, fix the analysis seed before looking at the data and never tune
     it.
   - `seed_validation` assigns one L'Ecuyer-CMRG stream per component to the
     prediction-side diagnostics. `mb_rng_streams()` creates
     scheduling-independent streams for other parallel jobs.
   - Seeded calls restore the caller's RNG kinds and state and use explicit
     normal and sampling algorithms. Use R's default Inversion normal
     generator if caller-stream restoration is needed: `.Random.seed` does not
     preserve the spare normal cached by Box-Muller.
5. Save participant-level out-of-fold predictions before computing metrics.
   Score the package measures with `store_models = TRUE` so that every
   resampling iteration is matched to its own prediction payload.
6. Record failed fits and resamples instead of silently dropping them. Keep
   solver convergence (`converged`, `iterations`) and every non-convergence,
   few-units or few-replicates warning. Summarise batchtools runs with
   `collect_mbspls_nested_cv()` at its default `allow_partial = FALSE` unless a
   partial result is explicitly intended.
7. Save `sessionInfo()`, package sources, compiler information and the dependency
   lockfile used for the run, including the revision of the GitHub-only
   `neuroCombat` package when ComBat is used.
8. Align PLS component signs and explicitly match components before aggregating
   loadings across resamples. For MB-sPLS, align signs per block.
9. Keep bootstrap uncertainty separate from permutation/randomisation tests.
10. Record the installed `mlr3mbspls` version and, when it was installed from a
    source archive, that archive's SHA-256 checksum.
11. To check the permutation calculations of the installed package, run the
    installed `validation/null_behavior.R` and `validation/omnibus_permutation.R`
    scripts. They cover small reference scenarios and do not validate a
    study-specific exchangeability design or a data-adaptive full pipeline.
12. For the scientific analysis, retain the permutation seed, analysis seed,
    exchangeability vectors, callback source or fixed specification (statistic,
    `ncomp`, `c_matrix`, standardisation, `max_iter`, `tol`), exact scalar
    statistic, discovery/confirmation split, `reference_signs` for a
    directional confirmation, and the complete pre-specified LC family. Report
    Monte Carlo precision as the exact Clopper-Pearson interval for the
    exceedance probability (`monte_carlo_conf_low`, `monte_carlo_conf_high`),
    not as a Wald standard error, and report the permutation group size when
    few exchangeable units or small strata are involved.
