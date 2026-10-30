# Experiment Design & Measurement Platform

A business-first Streamlit application for planning, monitoring, analyzing, and documenting experiments across three common operating models:

- **Customer experiments**: randomized tests across customers or accounts
- **Geographic tests**: experiments with pre-assigned Test and Control markets
- **Time-series experiments**: campaign measurement when no randomized control group is available

The interface is designed for desktop demos and business users. Each workflow starts with a visual data story, keeps the main path simple, and places technical controls inside advanced settings.

**Customer Business V2 is now the default Customer workspace.** It provides a
four-step setup, planning, monitoring and analysis flow, separate Partner and
Segment dimensions, versioned sample plans, SRM / pre-treatment SMD diagnostics,
and native PDF calculation memos. Existing Customer workflows remain available
through **Classic workspace**. See [the V2 guide](docs/customer-v2/README.md) for
input definitions, statistical methods, templates and Databricks persistence.

## What The Platform Does

### Experiment Command Center

The home page gives teams a shared view of the experiment portfolio:

- Lifecycle status from Planning through Completed
- Owner, planned launch, expected readout, and next action
- Experiment Pulse for sample progress, traffic pace, allocation health, data freshness, and missing-data alerts
- Saved configurations with version history, copy, and reopen actions
- Frozen Launch / Iterate / Stop decision snapshots with reviewer and rationale
- Printable one-page HTML business reports and staged rollout plans

Saved experiments are stored locally in `.experiment_portfolio.json`. This file is intentionally ignored by Git.

### Classic Customer Experiments

Use this workflow for acquisition policy, proactive credit-line increase, offers, pricing, retention, messaging, card design, product experiences, and other customer-level strategies.

The opening page immediately shows:

- Historical outcome by strategy
- Customer and account coverage
- Product / brand coverage
- Customer-group mix
- Available customer history and a one-row-per-account preview

The planning flow is:

1. **Bring Data**: upload customer history or use one of four demo scenarios.
2. **Choose Audience**: optionally plan separately by one business grouping such as FICO band, risk tier, or revenue band.
3. **Pick Outcome**: configure the primary outcome; secondary metrics and guardrails remain optional.
4. **Define Options**: create named variants or ordered numeric strategy values.
5. **Get Plan**: receive required accounts, traffic allocation, flow per option, and expected test duration.

Customer planning supports longitudinal unit-period data and cross-sectional one-row-per-unit data. Numeric strategies can estimate arm-specific historical means and standard deviations using closest-test-value assignment, business-defined bins, or exact values. The default design sizes each treatment against control using pairwise Neyman allocation, keeps the largest pairwise control requirement, and then recalculates each treatment with that fixed shared control. Every final comparison is checked for target power. Generalized Neyman fixed ratios remain an advanced alternative. Traffic is allocated toward a common completion time.

The customer analysis flow accepts aggregate results from a completed randomized test and presents arm-level lift and uncertainty. The business team makes the decision; individual customer integrity, assignment balance, SRM and guardrails cannot be verified from the summary input.

### Geographic Tests

Use this workflow when the business already supplies fixed Test and Control DMAs or other markets. The platform does not manufacture or optimize market assignments.

The first page previews the market panel visually before configuration, including geographic coverage, Test / Control composition, outcome history, and source data. Planning then estimates the market count and rollout duration supported by the supplied groups and effect scenario.

Analysis uses panel Difference-in-Differences with:

- DMA and date fixed effects
- DMA-clustered uncertainty
- Pre-trend diagnostics
- Placebo intervention checks
- Leave-one-DMA-out sensitivity
- Optional market-size weighting and existing pair metadata

### Time-Series Experiments

Use this workflow for campaigns or policy changes where no randomized control group exists. Daily, weekly, and monthly data are supported.

The opening page shows the full historical series, campaign timing, available pre/post observations, detected frequency, predictor availability, and raw source columns.

The planning flow estimates detectable effect and campaign duration from historical pseudo-interventions. The expected campaign effect is a user-supplied business scenario, not a forecast of an unknown future effect. Results are evaluated through the longest selected duration and include calibration and limited-history safeguards.

The analysis flow uses a single page: select a data source, inspect the preview, choose a metric, Partners, optional grouping and campaign launch date, then calculate. CSV and Excel uploads and pasted or manually entered tables are supported; Databricks is a placeholder. When multiple rows share a date, choose and confirm the aggregation appropriate for the metric and non-overlapping populations. Missing-period and model options are in Advanced settings. Both analyses run together:

- **Before-and-after comparison**: an explicitly calculated descriptive comparison of matching-length periods. It does not adjust for trend, seasonality, or concurrent changes.
- **Estimated impact using a model**: an explicitly run structural state-space forecast using pre-campaign history and optional unaffected predictors. It reports expected outcome without intervention, incremental impact, uncertainty, backtesting, and model-reliability diagnostics.

The page reports a descriptive comparison even when insufficient history prevents model fitting, and identifies the missing model explicitly. After any data or settings change, calculate again before using the result. The current structural model uses `statsmodels`; it is not a full Bayesian sampler.

## Statistical Design Notes

- Continuous and binary outcomes use direction-aware power calculations.
- Multi-arm customer inference uses Holm-adjusted comparisons and simultaneous confidence intervals.
- Numeric customer strategies use group- and strategy-specific historical variability where support is available.
- Geography analysis assumes Test / Control assignment is fixed before upload.
- Time-series planning reports detection capability under a stated effect scenario; it cannot know the true campaign effect before launch.
- Interim Experiment Pulse directions are operational signals and are not presented as final causal conclusions.

## Quick Start

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://127.0.0.1:8501](http://127.0.0.1:8501).

## Suggested Demo Story

For a short presentation:

1. Open the **Command Center** to show the experiment portfolio and items requiring attention.
2. Open **Customer** to show historical strategy/outcome evidence, then enter Plan to see the business-guided workflow.
3. Open **Geography** to show the market map and fixed Test / Control story.
4. Open **Time Series** to show the campaign timeline and the distinction between planning and impact analysis.
5. Return to the Command Center and open a saved experiment, Experiment Pulse, or a frozen decision report.

## Data Inputs

CSV and Parquet uploads are supported.

### Customer

Required fields depend on the workflow, but typically include:

- Customer or account ID
- Outcome column
- Assigned treatment arm for completed-result analysis
- Optional product / brand, audience group, acquisition date, observation period, and historical strategy fields

### Geography

- Date
- DMA or market identifier
- Outcome
- Fixed Test / Control group
- Optional covariates, market weights, pair IDs, campaign spend, and exposure

### Time Series

- Date
- Outcome
- Optional unaffected comparison signals, campaign flag, spend, and operational exposure

Internal Databricks controls are UI placeholders in this MVP; Unity Catalog loading is not connected yet.

## Customer: Summary Planning and Simple Results

The default planning path is **Summary statistics / Partners**. It accepts one
row per Partner and arm, entered in the table or uploaded as CSV:

```csv
Partner,Arm,Mean,SD,N
Partner A,Control,100,30,1000
Partner A,Option A,100,35,1000
Partner A,Option B,100,45,1000
```

- `Arm`: control or test option label.
- `Mean`: historical average of the continuous outcome over a common window.
- `SD`: sample standard deviation (`STDEV.S` / `ddof=1`). Alternatively supply
  `Variance` (`VAR.S`), equal to SD squared. Conflicting SD/variance values fail
  validation. Future treatment effects are **not** inferred from historical means.
- `Partner`: optional; legacy `Group` is also accepted. Omitted means one audience.
- `N`: optional historical independent-account count for planning (at least two
  when supplied). Observed N is required for summary analysis. Repeated account
  periods must first be aggregated to one outcome per account.

The example table and downloadable template contain synthetic assumptions.
Replace them before using the plan. Primary outcome, worthwhile effect, eligible
accounts per traffic period, and control are business inputs. Advanced defaults
are 5% false-positive rate, 80% detection chance, Holm comparison protection,
0% expected missing outcomes, and pairwise Neyman allocation.

The default multi-arm algorithm is:

1. Size each treatment/control pair at the Neyman ratio `n_t / n_c = SD_t / SD_c`.
2. Use the largest pairwise control requirement as the one shared control.
3. Fix that control and solve the minimum integer treatment count for every arm.
4. Verify target power for every final comparison, then inflate for expected
   missing outcomes and derive traffic shares. Planning uses conservative
   per-comparison alpha; analysis uses Holm-adjusted p-values and conservative
   simultaneous intervals. This transparent construction is not claimed to be
   a globally minimum-total-sample optimizer.

Choose **Separate plans per Partner** for Partner-specific decisions. Choose
**One combined plan** for a single overall decision across selected Partners.
The combined planner uses the same planned Partner mix in every arm and includes
both within-Partner and between-Partner variability. Mix defaults follow eligible
traffic and can be edited in advanced settings. Every Partner must supply the
same arm labels and control. Recruitment targets by Partner use integer rounding
that preserves each arm total. Combined targets do not power individual Partner
claims; small targets only approximate the requested mix after rounding.

The default analysis path is **3-column results**:

```csv
ID,Arm,Outcome
C001,Control,100
C002,Control,110
C003,Option A,115
C004,Option A,120
```

Upload one row per randomized account, choose control, then calculate. Column
names can be remapped in advanced settings. Duplicate/blank IDs, missing labels,
non-finite outcomes, and arms with fewer than two observations are blocked.
Continuous outcomes use Welch unequal-variance comparisons; 0/1 outcomes are
automatically recognized and use independent-proportion tests (Fisher's exact
when expected cell counts are small) with Newcombe intervals. Zero control rates
show undefined relative lift rather than infinity. Each option is compared only
with control; results do not rank a definitive winner among treatments.

Optional `Partner` gives separate Partner results, with joint decision protection
on by default. Three columns alone cannot establish randomized assignment,
outcome maturity, or equal Partner mix. Allocation mismatch is checked only with
an explicitly supplied split or matching saved plan, never an assumed equal split.
Results show effect, uncertainty, data checks, and plain-language interpretation.
Changes to data/settings invalidate results; Save retains the calculation time.

Detailed customer workflows and observed group-summary analysis remain available.
Summary planning currently supports continuous outcomes; use the detailed records
workflow to plan binary outcomes or additional metrics. Business owners record
final decisions in the Command Center.

Examples: `sample_data/customer_partner_summary.csv` and
`sample_data/customer_three_column_results.csv` (synthetic values).

## Windows Setup and Time-Series Troubleshooting

For a transferable ZIP, run `python scripts/build_portable.py` on the source
computer. It creates a dated package in `dist/` and refreshes
`dist/ab_testing_app_portable.zip`; ZIPs from older runs do not update when source
code or GitHub changes. The package includes `BUILD_INFO.json` with file hashes
so copied or mixed versions can be identified. Local environments, experiment
records, credentials, and Git metadata are excluded.

On Windows, extract the whole package into a **new folder**, open its
`ab_testing_app` directory, and double-click `start_windows.bat`. It builds a
separate `.venv-windows`, installs the pinned dependencies, checks package
integrity, and tests opening Home -> Time Series before starting the server.
The check detects Python exceptions; it does not test the browser renderer.
If a step fails, the console remains open with the error text. Install 64-bit
Python 3.11 with the Python launcher first. Do not run the app inside the ZIP or
reuse a server still running from another checkout.

Use **Python 3.11**, and recreate the virtual environment on each computer. Do not
copy a macOS `.venv` to Windows. In Windows Command Prompt, from this project:

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

For troubleshooting, use the same interpreter that starts Streamlit:

```bat
.venv-windows\Scripts\python.exe scripts\check_installation.py --smoke-test
```

If you used the manual `.venv` setup instead, substitute `.venv` for
`.venv-windows`. Copy the diagnostic output and full error, not only the last
line. A successful check does not establish that a browser-side error is fixed.

Time Series date labels use an explicit English month/day/year formatter without
calling `Timestamp.strftime`, `date.strftime`, or locale-dependent formatting.
This also covers the landing-page period label and data preview, not only the
planning and analysis screens.
Streamlit is pinned to 1.39.0, which fixes a dataframe-width React maximum-update-depth crash present in 1.37.1. Parquet support is explicitly included through `pyarrow`. Invalid uploads clear
prepared time-series data so later tabs cannot analyze stale input. Missing model
outcomes fail with an actionable message. Campaign planning exceptions display
an error without terminating the page, and planner fit caching is bounded.

## Monthly Planning Sample

`sample_data/time_series_monthly_campaign_planning.csv` contains 84 month-start observations from January 2019 through December 2025 for the **Time Series -> Plan Campaign** workflow.

- Date: `date`
- Outcome: `credit_card_applications`
- Candidate predictors: `control_product_applications`, `organic_search_index`, `consumer_demand_index`, and `fed_funds_rate`
- Optional operational exposure: `eligible_customers`

The app should detect Monthly frequency with no missing periods and offer 2, 3, 4, 6, 9, and 12-month candidate durations. Select predictors only when they will remain available after launch and cannot be affected by the campaign.

## Project Structure

```text
app.py                              Streamlit application and workflow UI
src/experiment_platform/core.py     Shared configuration, data access, and validation
src/experiment_platform/customer.py Customer design, power, analysis, and decisions
src/experiment_platform/geography.py Geographic planning and impact analysis
src/experiment_platform/timeseries.py Time-series planning and counterfactual analysis
src/experiment_platform/charts.py   Shared Plotly visualizations
src/experiment_platform/portfolio.py Saved experiments, Pulse, reports, and decisions
sample_data/                        Example time-series input
tests/                              Unit and workflow regression tests
app.yaml                            Databricks Apps entrypoint
```

Only the files above are needed for development. Local virtual environments,
Python caches, Git metadata, and saved local portfolio state are intentionally
excluded from transfer and deployment packages.

Run the test suite with:

```bash
pytest -q
```

## Databricks App

### Customer experiment results from summary statistics

Customer → Analyze results uses a long-format table, one row per
non-overlapping population × strategy × metric. Include `Strategy`, `Metric`,
`Sample number`, `Sample mean`, and unbiased `Sample variance` (with denominator
n−1). Optional categorical columns such as `Partner`, `FICO group`, and
`PD score group` identify the joint population. Download the in-app CSV template
for an example. Results first show overall charts for all selected metrics,
then segment charts and decision insights. A downloadable results PDF includes
the analysis settings, comparisons, insights, and calculation formulas.

Confirm that rows being pooled contain distinct customers in the same
observation window. This cannot be verified from summary data. The app pools
counts and sample moments, including between-group variation, then uses
Welch's test for continuous metrics and a two-proportion z-test or Fisher's
exact test for binary metrics. Overall p-values can be Holm-adjusted across
Tests within each metric. Subgroups and additional metrics are exploratory;
there is no adjustment across metrics. Overall results use observed arm mixes,
not the fixed population weights from the planning model. Summary inputs do
not verify randomization, SRM, missing outcomes, or duplicate customer IDs.

### Customer decision insights

Customer analysis generates descriptive insights from the current results.
The interface does not call a language model or expose a model endpoint button.
Comparisons between Tests are descriptive unless separately tested; the
insights are not a launch approval.

`app.yaml` starts Streamlit on port `8000`:

```yaml
command:
  - streamlit
  - run
  - app.py
  - --server.address=0.0.0.0
  - --server.port=8000
```

## MVP Boundaries

The following are not yet implemented:

- Unity Catalog table loading and production authentication
- Shared database-backed portfolio storage and multi-user permissions
- Full Bayesian structural time-series sampling
- Approved DMA polygon geometry; the current map uses market centroids
- Randomization-based geographic power simulation for large candidate-market pools
- Covariate-adjusted, count/rate, ratio, and time-to-event customer estimators
- Production audit logging, deployment hardening, and automated data pipelines
