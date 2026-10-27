# Customer Business V2 acceptance record

Scope: desktop Customer setup, planning, monitoring, analysis and reports.
Mobile layout was explicitly removed from scope by the user.

## Environment

- Python 3.11; pinned Streamlit 1.39 / pandas 2.2.2 / scipy 1.13.1.
- Browser plugin not available. Playwright used the installed Google Chrome in
  headless mode against `http://127.0.0.1:8503/` at 1536 × 1024.
- Native PDF generation: ReportLab, embedded BusinessSansSC / Noto Sans SC.
- PDF inspection: Poppler page renders, pypdf text checks, pdfplumber bounds.
- An actual Windows or Databricks production host was not available.

## Browser acceptance

| Check | Outcome |
| --- | --- |
| Page identity | Experiment Platform at the expected localhost URL |
| Meaningful content / blank-page check | Passed |
| Framework / Streamlit exception overlay | None |
| Browser console / page errors | None in the completed desktop flows |
| Setup and multiple arms / metrics | Saved and carried into planning |
| Historical data, future mix, sample requirements | Calculated and final power verified |
| Freeze and monitoring handoff | Version and expected allocation preserved |
| Snapshot validation | Assigned / Mature / Observed counts distinct |
| Analysis and confidence intervals | Overall and subgroup views populated |
| Native PDF downloads | Monitor and final-analysis downloads exercised |
| Independent three-column CSV upload | ID / Arm / Outcome mapped; effect +10 recovered |

The CSV browser script initially selected a file before the upload control's
rerender was complete. Waiting for the control and uploaded filename resolved
the automation timing issue; no parsing or statistical change was needed.

## Automated checks

Final full-suite result: **184 passed** (`python -m pytest -q`).

The full regression suite includes Geography, Time Series, Windows-compatible
dates, portable packaging, classic Customer and V2. V2 coverage includes:

- CSV and TSV equivalence, exact binary successes, SD conversion and row errors.
- Pairwise Neyman / largest shared Control / final integer power checks.
- Independent Segment sizing, fixed ratios and confirmatory Guardrail families.
- Raw versus same-definition aggregate inference; fixed-weight gold calculations.
- Missing frozen strata, zero variance, sparse Binary and exact fallback states.
- Unequal and stratum-specific SRM; continuous / categorical baseline SMD.
- Duplicate snapshots, missing outcomes and source-date maturity.
- Append-only versions, tampered imports, UI result invalidation and stable dates.
- Setup editor persistence, frozen weights surviving navigation, and population
  edits requiring a new version.
- Native PDF extraction, Chinese input and exclusion of customer IDs.

## Visual reference and fidelity ledger

Reference: `design-concept.png`, generated with the built-in Image Gen tool.
The reference and final browser screenshots were inspected with `view_image`
in the same QA pass at the reference's native 1536 × 1024 desktop dimensions.

| Point | Reference / rendered comparison | Resolution |
| --- | --- | --- |
| Product header | Teal C mark, Experiment Platform, customer context | Preserved |
| Workflow copy | Four ordered business stages | Exact stage labels preserved |
| Content hierarchy | Navy titles, short prompts, white sections, right summary | Preserved |
| Typography | Large readable headings, smaller labels and supporting copy | Inspected across forms, charts and result cards |
| Palette | Pale slate background, navy text, teal actions | Scoped V2 styling; global workflows retain their palette |
| Table anatomy | Editable rows, readable headers, modest borders | Implemented with native editable tables |
| Containers / spacing | Restrained radii, open spacing, short stage pages | Preserved within the existing application shell |
| Existing navigation | Concept omits the global sidebar | Intentional: retain Home / Geography / Time Series navigation |
| Multiple arms | Concept sketches two arm form fields | Intentional: editable arm table supports arbitrary named tests |
| Advanced metrics | More fields than the compact concept table | Intentional: required definitions / windows / tolerance controls |

Above-the-fold copy was checked against the visual reference and functional
requirements. Additional owner/version, explicit synthetic-data notices,
scientific limitations and existing global navigation are intentional. No
decorative marketing copy or image-based UI was added. Buttons, tables, filters,
plots and downloads execute actual application behavior.

## PDF acceptance

The delivered synthetic design, monitor and analysis examples contain 6, 13
and 22 pages respectively. Every page was rendered and inspected in page
contact sheets; representative first pages, dense tables and appendices were
also checked at readable resolution. No text characters fell outside the
specified page margin bounds. Embedded-font renders produced no missing-font
errors. Tables split with repeated headers; charts and technical appendices
remain inside the page. Full-precision inputs are also available in JSON.

## Boundaries

Production Databricks identity policy, Volume permissions and durable storage
location require configuration in the target workspace. No production deployment
is claimed. Sparse multi-stratum Binary overall tests, ratios of totals, survival
outcomes, correlated repeated measurements and sequential efficacy stopping
are not silently approximated as ordinary independent-customer t/z tests.
See the main V2 guide for the detailed statistical contract.
