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
`mb_cluster_bootstrap()` for whole-group sampling. The bootstrap stability
operator accepts `bootstrap_groups`; a named vector is aligned to task row IDs.
These mechanisms make a supplied design enforceable, but they cannot identify
the scientifically correct grouping variable.

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

## 3. Nested resampling is required for performance estimation

Any hyperparameter chosen from data—including component counts, block weights,
sparsity budgets, stopping rules, preprocessing variants and thresholds—must
be selected in an inner loop. Predictions used for the final performance
estimate must be generated only by outer folds or by an untouched external
cohort. Reusing inner-loop performance as the final estimate is optimistically
biased.

Repeated cross-validation does not create new independent participants.
Uncertainty calculations must account for overlap among folds and repetitions;
fold-level values are not independent replicates. Prefer participant-level
out-of-fold predictions and a resampling procedure defined at the independent
sampling unit.

For the same reason, `aggregate_mbspls_payloads()` does not combine conditional
fold-wise p-values by default. Its legacy Stouffer/Fisher option now requires an
explicit opt-in and remains labelled exploratory; neither combining p-values nor
forcing them to be monotone supplies a dependence or multiplicity correction.

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

`mb_bootstrap_summary()` implements this descriptive contract. The
prediction-side `val_test = "bootstrap"` payload uses the same fields and keeps
legacy p-value columns only as explicit `NA` values. Bootstrap intervals for
selected sparse loadings are stability summaries; selection and component
matching complicate ordinary fixed-parameter coverage interpretations.

When bootstrapping PLS loadings or coefficients, align component signs to a
reference solution before averaging or constructing intervals. PLS component
signs are algebraically arbitrary; unaligned replicates can cancel despite
representing the same solution.

## 5. Permutation tests

A permutation test must repeat the complete data-dependent pipeline. It is not
sufficient to permute already-computed predictions, residual summaries, or the
last fitted score while leaving selected features and tuned hyperparameters
fixed. Each permutation must repeat preprocessing, feature selection, model
selection, calibration and thresholding exactly as under the observed labels.

The package's component-wise training diagnostic fixes the fitted
preprocessing/hyperparameters and must not be reported as a full-pipeline test.
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
it refits standardisation and the requested sequential fit on every
permutation. If sparsity, component count, filtering, or other settings were
selected from the tested alignment, they must instead be selected again inside
the generic callback. `mb_permutation_pvalue()` remains the lower-level summary
for an already generated valid null distribution.

For LC-specific confirmation, use `mb_lc_confirmation_test()` on block-score
matrices computed for genuinely untouched observations with all preprocessing,
weights, loadings, deflation, component count, and selection frozen from
independent data. The function requires an explicit independence assertion,
supports row/stratum/whole-unit exchangeability, and reports Holm-adjusted
p-values across the complete supplied LC family. Its null concerns replication
of each fixed score association. It is not a population-rank test and is invalid
if the confirmation data influenced which LCs were fitted or reported.

The permutation scheme must preserve the design. Examples include shuffling at
participant rather than visit level, restricting permutations within strata,
or permuting treatment labels only as allowed by the randomisation. For a
sampled permutation distribution with `B` random permutations, use the finite
Monte Carlo correction `(b + 1) / (B + 1)`, count ties as at least as extreme,
and state the alternative and the centre used for a two-sided statistic.

The direct test returns one omnibus p-value. Its first-component statistic can
test complete cross-block independence, but that p-value is not proof that a
uniquely identifiable population LC exists. Raw permutations destroy every
shared component and therefore do not generate the sequential rank null needed
to establish that LC2 or a later component remains after earlier population
components. Generic later-LC p-values are not supported. Report the null,
statistic, exchangeability restrictions, complete analysis callback, seed,
replicate count, p-value, and Monte Carlo precision alongside effect sizes.

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

Component labels and signs are not inherently identifiable across resamples.
Align components using an explicit matching criterion, check for swaps or near
ties, and mark ambiguous matches rather than forcing an interpretation.
Sparsity parameters must be described in the same direction used by the
implementation—for example, whether a larger value means a larger L1 budget
and therefore less sparsity, or a larger penalty and therefore more sparsity.

## 9. Reproducibility

Store the complete resampling instances, seeds or independent RNG streams,
software versions, preprocessing state, feature schema and out-of-fold
predictions. A single global seed is insufficient for parallel resampling if
results depend on worker count or scheduling. Use independent deterministic
streams and avoid mutating the caller's RNG state from package internals.

`mb_rng_streams()` creates L'Ecuyer-CMRG streams without changing the caller's
RNG context. Bootstrap stability selection assigns one retained stream to each
replicate when `seed_bootstrap` is supplied.

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
- Winkler AM, Renaud O, Smith SM, Nichols TE (2020). Permutation inference for
  canonical correlation analysis.
  <https://doi.org/10.1016/j.neuroimage.2020.117065>
- Danyluik M et al. (2025). Evaluating permutation-based inference for partial
  least squares analysis of neuroimaging data.
  <https://doi.org/10.1162/imag_a_00434>
- Hall P, Wilson SR (1991). Two guidelines for bootstrap hypothesis testing.
  <https://doi.org/10.2307/2532163>
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
