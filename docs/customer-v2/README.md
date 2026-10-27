# Customer Business V2

The default Customer entry is a four-stage business workbench: setup, sample
planning, monitoring and analysis. Classic Customer remains in the sidebar for
existing workspaces. Geography and Time Series retain their existing routes.

## Run and explore

Use Python 3.11 and install `requirements.txt`, then run
`streamlit run app.py`. Choose **Customer**, then **Explore synthetic demo** to
exercise the complete workflow. The demo is explicitly labeled and a new
experiment clears its data. Use **New customer experiment** for real inputs.

1. Save the business question, one Control, Test options and metric definitions.
2. Upload CSV, paste Excel TSV, or edit the historical statistics grid. A row is
   Partner × Segment × Metric, with optional historical Arm. Confirm the future
   population mix, calculate sample targets and freeze a version.
3. Upload one row per randomized unit, or separate aggregate outcome / assigned
   count / pre-treatment baseline tables. Source snapshot dates are preserved.
4. Calculate a fixed-horizon readout. View Overall, Partner and Segment effects,
   guardrail conclusions, data health and advanced statistical details.

Existing results without a prospective plan can use **I already have results**
after setup. ID, Arm and Outcome suffice. Such readouts are exploratory: no
sample target, intended assignment ratio, SRM pass or prospective power is
invented.

## Inputs

Historical minimum: `partner, segment, metric, sample_mean, sample_variance,
sample_number`. N is the number of independent randomized units. Variance is
VAR.S (ddof=1); SD / STDEV.S is accepted and squared. Missing Segment is labeled
All customers; missing Partner is labeled Not provided. Binary metrics require
exact `successes` and `sample_number`, rather than reconstructing successes
from a rounded rate. Normal planning requires N >= 2. Assumption-only mode can
omit continuous historical N and is labeled as limited evidence.

Customer records: `id, arm, <metric columns>`, optionally `partner, segment,
assignment_date, snapshot_date, outcome_window_complete, baseline_...`.
IDs are read as strings. Duplicate IDs require distinct dated snapshots with
unchanged assignment and population. The latest snapshot per ID replaces its
previous record. Conflicting or duplicated snapshots are rejected.

Assigned N counts all randomized units, Mature N counts completed windows,
Observed N counts mature nonmissing results for the selected metric. Missing
outcomes are never zeros and never remove units from assignment counts. Maturity
is explicit or derived from assignment date and the longest configured outcome
window plus maturity delay. Without either, maturity remains unverified and a
final decision is flagged for review.

Aggregate mode separates assignment counts from per-metric moments, preventing
double-counting across metrics. Additional pre-treatment summaries are required
for SMD. Aggregate data cannot verify duplicate IDs or individual outliers.

Templates and the complete synthetic example are in `sample_data/customer_v2_*`.

## Statistical contract

- The target is a fixed-weight Partner × Segment population. Every arm uses the
  same frozen weights. Estimate = sum(w × mean difference), sampling variance =
  sum(w² × sample variance / N) across both arms and strata. Between-stratum
  differences in mean are not added to this stratified estimator's variance.
- New Test arms inherit Control variance unless actual arm history or an
  explicit SD multiplier is supplied. Historical differences in mean do not
  become future treatment effects. Historical moment uncertainty is not fully
  propagated by the plug-in power calculation.
- Recommended allocation solves each Test / Control pair at its Primary SD
  ratio, takes the largest Control, then resolves each Test against that fixed
  Control. All included metric comparisons and rounded integer stratum counts
  are rechecked. This is not a global total-sample optimum.
- The default family includes all Test-versus-Control comparisons for Primary,
  all Guardrails and planned Secondary metrics, across independent decision
  populations. A Guardrail can be excluded from sizing without dropping it
  from inference protection. Planning uses alpha / family.
  Final p-values use Holm; intervals use conservative Bonferroni protection.
  Exploratory slices form a separate family. Power is per comparison, not the
  probability that every arm and metric succeeds together.
- Continuous inference uses stratified Welch-Satterthwaite t methods. Adequate
  Binary cells use a fixed-weight pooled-null z-test and unpooled normal
  risk-difference interval. One sparse stratum uses Fisher exact and a separate
  approximate Newcombe interval. A sparse weighted multi-stratum overall test
  is withheld, with exact cell tests reported. Binary boundary-rate normal
  sample planning is refused instead of silently clipping a rate.
- Guardrails use a saved deterioration margin in absolute metric units
  (binary .01 = 1 percentage point). Normal / Welch non-inferiority p-values are
  distinct from two-sided efficacy p-values. Conservative family intervals give
  Within tolerance / Beyond tolerance / Inconclusive. A non-significant efficacy
  p-value is never labeled safe. Operational alerts are separate descriptive
  thresholds and cannot authorize an efficacy win.
- SRM compares Assigned N with frozen per-stratum allocation. Expected Overall
  counts sum actual stratum enrollment × intended shares. Equal shares use
  Pearson goodness-of-fit (or exact binomial for a sparse two-arm case).
  Differing stratum allocations use a conditional sum-of-multinomials Monte
  Carlo calibration (49,999 draws, fixed seed), rather than a false multinomial
  model. Sparse multi-arm diagnostics can remain Insufficient data. Diagnostic
  thresholds account conservatively for the displayed Overall / stratum family.
- SMD uses pre-treatment features only: mean difference / sqrt((vC + vT) / 2).
  Categorical levels use indicator proportions and Bernoulli variances. Zero
  variance and missing features have explicit states. The .1 reference line is
  diagnostic, not proof of failed randomization. Covariate timing is an input
  assumption, not something these tables can independently verify.
- Missing frozen strata block the corresponding inference. Weights are not
  silently renormalized. Only predeclared independent decision populations have
  a power guarantee. A significant segment versus a non-significant segment
  is not an interaction test. Ratios, survival outcomes and correlated repeated
  records require a different statistical model.

## Reports and persistence

Every stage can export a server-generated PDF, protocol/setup JSON and aggregate
CSV from the same calculation object. Reports include source moments, weights,
allocation steps, family definitions, tests, confidence intervals, diagnostics,
limitations, source times and fingerprints. Customer IDs are excluded. The PDF
is a calculation memo, not proof of approval or completed audit. An embedded
open-source font supports Latin and common Simplified Chinese without Office,
printers, browser plug-ins or host font installation.

Frozen versions are append-only files in `.customer_protocols/` by default.
On Databricks, set `CUSTOMER_PROTOCOL_DIR` to an access-controlled persistent
Volume mounted for the app service principal. The sidebar can reopen saved
versions or import exported protocol JSON. Imports recalculate derived targets
and power checks before accepting a prospective plan.

The default local directory is a development store. Organization SSO/access
policy, production Volume permissions, retention and an approved durable shared
storage location must be configured in the target Databricks workspace; this
repository does not provision them. Browser session state is not durable raw
customer storage. Raw records must be reuploaded after a session restart.

Generate sample PDFs and synthetic input files with:
`python scripts/customer_v2_examples.py`.

## Primary references

- [SciPy Welch t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
- [SciPy Fisher exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)
- [statsmodels two-proportion z-test](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html)
- [Microsoft SRM diagnosis](https://www.microsoft.com/en-us/research/publication/diagnosing-sample-ratio-mismatch-in-online-controlled-experiments-a-taxonomy-and-rules-of-thumb-for-practitioners/)
- [Austin baseline balance diagnostics](https://pubmed.ncbi.nlm.nih.gov/19757444/)

Visual reference: `design-concept.png`. Implementation intentionally preserves
the existing global sidebar and uses editable tables to support an arbitrary
number of arms and metrics, rather than the concept's two-arm form.
