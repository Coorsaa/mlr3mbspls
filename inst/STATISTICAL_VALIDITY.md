# Statistical validity requirements for mlr3mbspls

This document defines the inferential contract for multiblock sparse PLS/PCA
workflows. It is part of the installed package because these requirements are
not optional presentation advice: violating them changes the estimand or makes
reported uncertainty invalid.

## 1. Unit of analysis and exchangeability

The row of a feature matrix is not automatically an independent observation.
Before any split, bootstrap, permutation, or cross-validation operation,
identify all dependence units: participant, family, household, recruiting
centre, scanner, acquisition batch, treatment group, therapist, ward, or time
series. All rows from one indivisible unit must remain in the same assessment
fold. A bootstrap must sample the appropriate independent unit. A permutation
must operate only within an exchangeability scheme justified by the design.

For repeated measurements, visit-level random splitting is leakage. For
multisite data, ordinary random cross-validation estimates performance on a
mixture of already-seen sites; it does not estimate transport to a new site.
Use leave-site-out or explicitly grouped outer resampling when site transfer is
the intended claim.

Use `mb_assert_disjoint_groups()` to audit a split,
`mb_permute_group_labels()` for a group-level label shuffle, and
`mb_cluster_bootstrap()` for whole-group sampling. The nested-CV entry points
and the sequential tuners reject splits that share rows or a value of the
task's mlr3 `group` role between analysis and assessment partitions.

`PipeOpMBsPLSBootstrapSelect` resamples whole groups of the task's `group`
role by default. An explicit `bootstrap_groups` vector (a named vector is
aligned to task row IDs) must be equal to or coarser than that role; to
resample finer units, remove the role before the operator. Training stops if
only one unit can be resampled (overall or in every stratum), because every
replicate would then equal the training data, and warns when there are fewer
than 10 units or fewer distinct bootstrap samples than replicates. The
permutation functions accept `exchangeability_unit`, `within_unit` and
`strata`, report the size of the resulting permutation group, and warn when
`n_perm` reaches it. These mechanisms make a supplied design enforceable, but
they cannot identify the scientifically correct grouping variable.

## 2. Every learned operation belongs inside resampling

The following operations must be fitted using analysis-fold data only and then
applied unchanged to the assessment fold:

- missing-data imputation and missingness-indicator construction;
- centring, scaling, transformations and outlier rules;
- variance, prevalence, reliability or univariate feature filters;
- batch correction, harmonisation and residualisation;
- block weighting and block selection;
- feature selection and sparse loading estimation;
- component-number and sparsity tuning;
- class balancing, synthetic sampling and case weights;
- probability calibration and decision-threshold selection.

A transformation is not unsupervised merely because it ignores the outcome.
Using assessment-fold predictor distributions still leaks information about the
assessment sample. Harmonisation requires particular care: fitting a site or
batch model on all observations can make held-out observations influence their
own representation.

The package operators follow this rule when they are placed inside a
resampled graph. `PipeOpMBsPLS`, `PipeOpMBsPLSXY` and `PipeOpMBsPCA` centre
every block column with its training mean and apply the stored means at
prediction; the sequential tuners centre each inner fold with the means of its
training rows. `PipeOpSiteCorrection` estimates all correction parameters from
the training rows and never re-estimates them from prediction data. It rejects
target columns wherever it reads them at prediction time; ComBat `covariates`
may name the target because they are used at training only. Preserving an
outcome or group variable during batch correction can still exaggerate
downstream group differences in unbalanced designs (Nygaard et al., 2016). Give
a fitting `subgroup` as task row IDs or a column specification so that it
selects the same observations in every fold.

## 3. Nested resampling is required for performance estimation

Any hyperparameter chosen from data—including component counts, block weights,
sparsity budgets, stopping rules, preprocessing variants and thresholds—must
be selected in an inner loop. Predictions used for the final performance
estimate must be generated only by outer folds or by an untouched external
cohort. Reusing inner-loop performance as the final estimate is optimistically
biased.

The tuners' `result_y` and the inner scores reported by nested CV are such
inner-loop scores. Each component's sparsity budget is chosen to maximise the
score on the same inner folds, so these values are optimistic. The
early-stopping rules of `TunerSeqMBsPLS` and `TunerSeqMBsPCA` are heuristics
that reuse the data that selected the budget; `TunerSeqMBsPLS` also pools
dependent fold p-values with a Stouffer combination. Their `perm_alpha` is a
cutoff, not an error rate. The MB-sPCA rule tests cross-block association only
and is skipped with a warning when fewer than two blocks are retained.

`mbspls_nested_cv()` and `mbspls_nested_cv_batchtools()` refit the complete
configured graph, including bootstrap stability selection, in every outer fold,
so the outer estimate covers that pipeline. `collect_mbspls_nested_cv()`
refuses to summarise unfinished or failed outer folds unless
`allow_partial = TRUE`; a partial summary describes only the folds that
finished and can be biased when failures depend on the data.

The MB-sPLS and MB-sPCA measures score each resampling iteration on the
prediction payload of its own trained model. They therefore need
`store_models = TRUE` in `resample()`, `benchmark()` and the mlr3tuning
functions. A component whose held-out block scores have fewer than two
non-degenerate blocks has no defined cross-block association; `mbspls.mac` and
`mbspls.mac_evwt` score it as zero.

Repeated cross-validation does not create new independent participants.
Uncertainty calculations must account for overlap among folds and repetitions;
fold-level values are not independent replicates. Prefer participant-level
out-of-fold predictions and a resampling procedure defined at the independent
sampling unit.

For the same reason, `aggregate_mbspls_payloads()` does not combine conditional
fold-wise p-values by default. Its Stouffer/Fisher option requires
`allow_p_combination = TRUE`, warns, and remains labelled exploratory; neither
combining p-values nor forcing them to be monotone supplies a dependence or
multiplicity correction.

## 4. Bootstrap output is not automatically a hypothesis test

An ordinary nonparametric bootstrap approximates sampling uncertainty around
the fitted data-generating process. It is not generally a null distribution.
For a non-negative metric such as accuracy, AUROC or RMSE, counting bootstrap
values on one side of zero is not a valid test and can produce a meaningless
near-zero value.

Report the observed estimate, bootstrap bias, standard error, interval method,
confidence level and number of successful replicates. A null hypothesis test
requires a null-generating mechanism appropriate to the estimand, such as a
valid randomisation/permutation design, a restricted model, or a bootstrap
constructed under the null.

`mb_bootstrap_summary()` implements this descriptive contract with type-8
sample quantiles for percentile and basic intervals. The prediction-side
`val_test = "bootstrap"` payload uses the same fields and keeps legacy p-value
columns only as explicit `NA` values.

When bootstrapping PLS loadings or coefficients, match components and align
their signs to a reference solution before averaging or constructing
intervals. PLS component signs are algebraically arbitrary; unaligned
replicates can cancel despite representing the same solution. The reported
MB-sPLS criteria (mean absolute or root-sum-of-squares cross-block
correlation) do not change when the sign of a single block is flipped, and a
bootstrap refit can reach a solution whose relative block orientation differs
from the training fit, so signs must be aligned block by block.
`PipeOpMBsPLSBootstrapSelect` matches replicate components to the training
components by an exact assignment and aligns signs block by block. Within one
fit, the solver orients every block score towards the consensus of the other
blocks (it maximises the covariance with their mean standardised score) and
flips signs only jointly, so the score signs of a fitted model are meaningful;
with three or more blocks an individual pair can still be negatively
correlated. Directional confirmation with `reference_signs` (section 5)
relies on these fitted signs. If blocks are re-oriented for reporting with
`mbspls_flip_weights()`, take `reference_signs` from the discovery scores of
the same re-oriented model that scores the confirmation data.

The intervals in its `weights_ci` are percentile intervals of the sign-aligned
replicate weights at the stored level `1 - alpha`. They are stability
summaries, not confidence intervals with fixed-parameter coverage: the weights
are sparse and selected, replicates must pass a score-correlation acceptance
gate (`min_score_cor`), and components with few accepted replicates are
flagged (`components_below_effective_floor`). With `selection_method = "ci"`,
a feature is selected when its interval lies strictly above or below zero and
its absolute bootstrap mean exceeds `magnitude_threshold`; this is a stability
rule, not a test of a non-zero population weight. With the default
`stable_weight_source = "training"`, the stable support is the intersection of
the training support and the bootstrap selection.

## 5. Permutation tests

A permutation test must repeat the complete data-dependent pipeline. It is not
sufficient to permute already-computed predictions, residual summaries, or the
last fitted score while leaving selected features and tuned hyperparameters
fixed. Each permutation must repeat preprocessing, feature selection, model
selection, calibration and thresholding exactly as under the observed labels.

The package's component-wise training diagnostic fixes the fitted
preprocessing/hyperparameters and must not be reported as a full-pipeline test.
Fitted states store its p-values with a `p_value_scope`, and
`mbspls_model_summary()` reports them as `conditional_p_value`.
Prediction-side `val_test = "permutation"` fixes trained weights. It is a
conditional diagnostic when prediction data were reused or their role is
unclear. It can test a fixed learned score association only when the prediction
rows are genuinely untouched confirmation observations, every transformation
was frozen from independent discovery data, and row-wise exchangeability is
valid. It does not support grouped confirmation designs or multiplicity
adjustment by itself.

Use `mb_permutation_test()` to permute raw aligned blocks and rerun one supplied
complete-analysis callback for every shuffle. Use
`mbspls_permutation_test()` for a fixed, pre-specified MB-sPLS omnibus analysis;
it refits standardisation and the MB-sPLS fit on every permutation (all
requested components for the `*_sum` statistics, the first component for the
`*_lc1` statistics). If sparsity, component count, filtering, or other settings
were selected from the tested alignment, they must instead be selected again
inside the generic callback. `mb_permutation_pvalue()` remains the lower-level
summary for an already generated valid null distribution.

The MB-sPLS solver starts deterministically, so an MB-sPLS statistic does not
depend on `analysis_seed`. It is the cross-block objective at the solver's
solution, which is a local optimum of a non-convex criterion and not
necessarily the global maximum. Observed and permuted data are fitted by the
same deterministic procedure, so the test is valid for the statistic as
computed. For a stochastic callback in `mb_permutation_test()`, the analysis
seed is part of the definition of the statistic: different seeds can give
materially different p-values. Fix and record it before looking at the data;
choosing among seeds after seeing results is an uncorrected multiple-testing
procedure.

For LC-specific confirmation, use `mb_lc_confirmation_test()` on block-score
matrices computed for genuinely untouched observations with all preprocessing,
weights, loadings, deflation, component count, and selection frozen from
independent data. The function requires an explicit independence assertion,
supports row/stratum/whole-unit exchangeability, and reports Holm-adjusted
p-values across the complete supplied LC family. A directional replication
claim requires `reference_signs`, the expected signs of the pairwise score
correlations fixed from discovery; the one-sided test then rejects only for
association in the discovery direction. With more than two blocks the
directional statistic is the mean of the sign-oriented pairwise correlations,
so a rejection shows association in the discovery direction on average, not
for every pair; check the returned `pairwise_correlations` before claiming
that each pair replicated. Without `reference_signs` the statistic is unsigned
and the test establishes dependence in either direction, including one
opposite to discovery. The observed signed correlations are always returned.
The test is not a population-rank test and is invalid if the confirmation data
influenced which LCs were fitted or reported.

The permutation scheme must preserve the design. Examples include shuffling at
participant rather than visit level, restricting permutations within strata,
or permuting treatment labels only as allowed by the randomisation. Nuisance
variables need the same care. Permuting within site tests independence of the
blocks conditional on site, assuming rows are exchangeable within each site;
an unrestricted shuffle tests marginal independence, which a shared site or
scanner shift alone can reject. Residualising on a nuisance variable and then
shuffling freely is not automatically valid. With few exchangeable units or
small strata, the permutation group, not `n_perm`, limits the attainable
p-value to about `1 / |G|`; the functions report the group size and warn when
`n_perm` reaches it.

For a sampled permutation distribution with `B` random permutations, use the
finite Monte Carlo correction `(b + 1) / (B + 1)`, count ties as at least as
extreme, and state the alternative and the centre used for a two-sided
statistic. Report Monte Carlo precision as the exact Clopper-Pearson interval
for the exceedance probability, estimated by `b / B`
(`monte_carlo_conf_low`, `monte_carlo_conf_high`). For `b = 0` and `B = 99`,
its upper bound is about 0.037. The Wald standard error is zero at `b = 0`
or `b = B`, where the package returns it as `NA`; do not report it as the
precision.

The direct test returns one omnibus p-value. Its first-component statistic can
test complete cross-block independence, but that p-value is not proof that a
uniquely identifiable population LC exists. Raw permutations destroy every
shared component and therefore do not generate the sequential rank null needed
to establish that LC2 or a later component remains after earlier population
components. Generic later-LC p-values are not supported. Report the null,
statistic, exchangeability restrictions, complete analysis callback or fixed
specification (including `max_iter` and `tol`), permutation seed, analysis
seed, replicate count, p-value, and Monte Carlo interval alongside effect
sizes.

## 6. Metrics, calibration and clinical utility

Pre-specify a primary metric linked to the intended use. For imbalanced
psychiatric outcomes, accuracy can conceal complete minority-class failure.
Report discrimination, precision-recall behaviour where relevant, sensitivity,
specificity, predictive values at clinically justified thresholds, and proper
scores such as log loss or Brier score. Report calibration intercept and slope
and provide uncertainty. Thresholds selected to maximise a metric are tuned
parameters and belong inside nested resampling.

Statistical discrimination is not clinical utility. Decision-curve or other
explicit utility analysis must use defensible consequences of false positives,
false negatives and intervention burden. A high AUROC alone does not justify a
clinical decision rule.

## 7. Prognosis is not treatment-effect prediction

A model that predicts outcome under observed care is prognostic. It does not
identify which treatment is better for an individual. Precision-treatment
claims require a causal treatment-effect estimand, positivity, consistency,
adequate confounding control or randomisation, treatment-by-covariate modelling,
and evaluation of treatment-policy value. Do not reinterpret a main-effect
classifier or PLS score as an individual treatment effect.

## 8. Multiblock PLS/PCA interpretation

Sparse PLS loadings are selected, sample-dependent quantities. They are not
ordinary regression coefficients with fixed-feature standard errors. Correlated
predictors can exchange importance across resamples while predictive behaviour
changes little. Report selection frequencies, sign-aligned stability,
block-level stability and uncertainty rather than declaring a single selected
feature list to be a biological signature.

The fitted weights are one local optimum of a non-convex criterion, found from
a deterministic start. With strong sparsity, pure noise, or many more features
than rows, other local optima can reach a higher objective, so a sparse
solution can depend on the budgets and on small changes in the data. Check
sensitivity to the sparsity budgets and to resampling before interpreting
individual features, and treat non-convergence warnings (`converged = FALSE`)
as a reason to inspect the fit.

Component labels and signs are not inherently identifiable across resamples.
Align components using an explicit matching criterion, check for swaps or near
ties, and mark ambiguous matches rather than forcing an interpretation. For
MB-sPLS, align signs per block across resamples (section 4). Sparsity
parameters must be described in the same direction used by the
implementation. In this package, `c_<block>` and `c_matrix` are L1 budgets of
unit-L2 weight vectors in `[1, sqrt(p)]`: a larger value allows more non-zero
weights and therefore less sparsity.

## 9. Reproducibility

Store the complete resampling instances, seeds or independent RNG streams,
software versions, preprocessing state, feature schema and out-of-fold
predictions. A single global seed is insufficient for parallel resampling if
results depend on worker count or scheduling. Use independent deterministic
streams and avoid mutating the caller's RNG state from package internals.

MB-sPLS fits draw no random numbers; the seeds of the package govern
permutations and bootstrap replicates only. `mb_rng_streams()` creates
L'Ecuyer-CMRG streams without changing the caller's RNG context. Bootstrap
stability selection assigns one stream to each replicate when `seed_bootstrap`
is supplied; with the default `NULL` it draws from the session RNG. The
installed `REPRODUCIBILITY.md` lists what to record.

## 10. Minimum evidence before clinical use

Clinical use requires more than internal cross-validation. At minimum, perform
external validation in the intended population and care pathway, quantify
site/scanner and temporal transport, assess calibration and subgroup
uncertainty, document missing-data handling, conduct prospective evaluation,
and establish monitoring for drift and calibration failure. Security, privacy,
human-factors, quality-system and applicable regulatory review remain separate
obligations.

The installed `MODEL_CARD.md` turns these obligations into a study-specific
checklist.

## References

- Phipson B, Smyth GK (2010). Permutation p-values should never be zero:
  calculating exact p-values when permutations are randomly drawn.
  <https://doi.org/10.2202/1544-6115.1585>
- Clopper CJ, Pearson ES (1934). The use of confidence or fiducial limits
  illustrated in the case of the binomial.
  <https://doi.org/10.1093/biomet/26.4.404>
- Holm S (1979). A simple sequentially rejective multiple test procedure.
  Scandinavian Journal of Statistics, 6(2), 65-70.
  <https://www.jstor.org/stable/4615733>
- Winkler AM, Renaud O, Smith SM, Nichols TE (2020). Permutation inference for
  canonical correlation analysis.
  <https://doi.org/10.1016/j.neuroimage.2020.117065>
- Danyluik M et al. (2025). Evaluating permutation-based inference for partial
  least squares analysis of neuroimaging data.
  <https://doi.org/10.1162/imag_a_00434>
- Hall P, Wilson SR (1991). Two guidelines for bootstrap hypothesis testing.
  <https://doi.org/10.2307/2532163>
- Nygaard V, Rødland EA, Hovig E (2016). Methods that remove batch effects
  while retaining group differences may lead to exaggerated confidence in
  downstream analyses. <https://doi.org/10.1093/biostatistics/kxv027>
- Cawley GC, Talbot NLC (2010). On over-fitting in model selection and
  subsequent selection bias in performance evaluation.
  <https://www.jmlr.org/papers/v11/cawley10a.html>
- Varma S, Simon R (2006). Bias in error estimation when using
  cross-validation for model selection.
  <https://doi.org/10.1186/1471-2105-7-91>
- Roberts DR et al. (2017). Cross-validation strategies for data with
  temporal, spatial, hierarchical, or phylogenetic structure.
  <https://doi.org/10.1111/ecog.02881>
- Wolff RF et al. (2019). PROBAST: A tool to assess the risk of bias and
  applicability of prediction model studies.
  <https://doi.org/10.7326/M18-1376>
- Moons KGM et al. (2025). PROBAST+AI: an updated quality, risk of bias,
  and applicability assessment tool for prediction models using regression
  or artificial intelligence methods.
  <https://doi.org/10.1136/bmj-2024-082505>
- Kent DM et al. (2020). The Predictive Approaches to Treatment effect
  Heterogeneity (PATH) statement. <https://doi.org/10.7326/M18-3667>
- Vickers AJ, Elkin EB (2006). Decision curve analysis: a novel method for
  evaluating prediction models.
  <https://doi.org/10.1177/0272989X06295361>
