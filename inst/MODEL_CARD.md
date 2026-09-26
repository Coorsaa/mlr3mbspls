# Model card checklist for multiblock prediction studies

This checklist is an analysis record, not a certificate. Complete it for each
dataset, target, estimand, and deployment setting. Use `not applicable` only
with a written rationale; software cannot infer these study-design choices.
For health prediction work, assess development quality and evaluation risk of
bias against the current
[PROBAST+AI framework](https://doi.org/10.1136/bmj-2024-082505) as well as
study-specific rules.

## Intended use and claim

- Intended population, care pathway, decision point, and user:
- Diagnostic, prognostic, descriptive, or causal treatment-effect estimand:
- Outcome definition, ascertainment window, prediction horizon, and censoring:
- Actions triggered by the output and consequences of false decisions:
- Explicitly prohibited uses:
- Evidence level: exploratory, internally validated, externally validated,
  prospectively evaluated, or deployed under monitoring:

A diagnostic or prognostic predictor does not identify heterogeneous treatment
effects. Individual treatment-selection claims require a treatment-effect
estimand and defensible assumptions about randomisation or confounding,
positivity, consistency, interference, and treatment versions.

## Cohort and dependence

- Inclusion/exclusion criteria and cohort flow:
- Recruitment setting, dates, prevalence, spectrum, and reference standard:
- Independent sampling unit (participant, family, site, scanner, ward, etc.):
- Repeated-measure timing and rule keeping all dependent rows together:
- Site/scanner/batch distribution and any site-outcome confounding:
- Medication, treatment, referral, and care-pathway variables:
- Demographic and socioeconomic composition:
- Missing-data amount, causes, timing, and missing-not-at-random sensitivity:

Record the exact outer and inner split indices. If transport to unseen sites or
future time periods is claimed, use a resampling design that directly targets
that setting rather than ordinary random cross-validation.

## Leakage and model selection

- Which operations are fitted separately in each analysis split:
- Imputation, transformations, variance filters, and block weights:
- Site/batch correction and whether assessment data influence its fit:
- Feature selection, component count, sparsity, and all other tuning:
- Class balancing, case weights, calibration, and threshold selection:
- Confirmation that final performance uses only outer-fold or external
  predictions, never the inner tuning scores:
- Confirmation that any permutation reruns the complete data-dependent pipeline
  using a design-valid exchangeability scheme:

## Performance and uncertainty

- Pre-specified primary metric and clinical rationale:
- Discrimination and precision-recall metrics:
- Sensitivity, specificity, predictive values, and threshold source:
- Proper scores and calibration intercept/slope with uncertainty:
- Participant-level out-of-fold or external predictions retained:
- Bootstrap sampling unit, interval method, and effective replicate count:
- Subgroup sample sizes, calibration, uncertainty, and fairness definition:
- External, temporal, and site/scanner transport results:
- Decision-curve or other utility analysis with defensible consequences:

Bootstrap intervals describe uncertainty and are not automatically hypothesis
tests. A p-value does not replace effect size, calibration, transportability,
or clinical net benefit.

## MB-sPLS interpretation and stability

- Component matching and sign-alignment rule across resamples:
- Ambiguous component matches, swaps, and rejected replicates:
- Feature and block selection frequencies and sign-aligned intervals:
- Sensitivity to correlated predictors, sparsity choices, and preprocessing:
- Multiplicity strategy and independent replication status:
- Language preventing selected loadings from being labelled causal mechanisms,
  biomarkers, or individually actionable features without separate evidence:

## Operations, governance, and monitoring

- Software/package source checksum, versions, compiler, and RNG streams:
- Data provenance, access control, privacy review, and threat model:
- Prospective workflow and human-factors evaluation:
- Monitoring population shift, missingness, calibration, subgroup performance,
  failures, and overrides:
- Recalibration/retraining triggers, rollback plan, and accountable owner:
- Applicable medical-device, quality-system, ethics, and regulatory review:

Until those obligations are satisfied, treat outputs as research results rather
than clinical decision support.
