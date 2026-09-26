# Reproducibility protocol

1. Save immutable participant identifiers and all outer/inner split indices.
2. Fit every learned preprocessing step only on analysis-fold observations.
3. Store fitted preprocessing state and the exact predictor names and order.
4. Use deterministic independent RNG streams for parallel jobs; retain them.
   `mb_rng_streams()` creates scheduling-independent L'Ecuyer-CMRG streams, and
   bootstrap stability selection stores its per-replicate streams when seeded.
   Seeded calls use explicit normal and sampling algorithms. Use R's default
   Inversion normal generator if caller-stream restoration is needed:
   `.Random.seed` does not preserve the spare normal cached by Box-Muller.
5. Save participant-level out-of-fold predictions before computing metrics.
6. Record failed fits and resamples instead of silently dropping them.
7. Save `sessionInfo()`, package sources, compiler information and the dependency
   lockfile used for the run.
8. Align PLS component signs and explicitly match components before aggregating
   loadings across resamples.
9. Keep bootstrap uncertainty separate from permutation/randomisation tests.
10. Rebuild and test the package from the source archive used for analysis.
11. Record the source archive SHA-256, `R CMD build` output, full
    `R CMD check` log, installed-package smoke test, and archive inventory.
12. Run the installed `validation/null_behavior.R` regression when changing
    permutation code. Its fixed-weight exchangeable-null simulation checks the
    add-one calculation and a broad type-I bound; it does not validate a
    study-specific exchangeability design or a data-adaptive full pipeline.
13. Run the installed `validation/omnibus_permutation.R` regression when
    changing complete-analysis inference. It checks iid-null calibration,
    whole-unit-null calibration, strong-signal sensitivity, independent-
    confirmation family-wise error and sensitivity, and the finite p-value floor.
    Retain the permutation seed, fixed `analysis_seed`, exchangeability vectors,
    callback source, exact scalar statistic, discovery/confirmation split, and
    complete pre-specified LC family used in the scientific analysis.
