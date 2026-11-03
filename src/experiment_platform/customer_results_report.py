"""Self-contained, aggregate-only HTML customer experiment results report."""
from html import escape
from math import isfinite


def build_results_report(report: dict) -> bytes:
    def esc(value):
        return escape(str(value), quote=True)

    def num(value, digits=3, scale=1):
        try:
            value = float(value) * scale
            return f'{value:,.{digits}f}' if isfinite(value) else 'Not available'
        except (TypeError, ValueError):
            return 'Not available'

    def table(headers, rows):
        headings = ''.join(f'<th scope="col">{esc(x)}</th>' for x in headers)
        body = ''.join('<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows)
        return f'<div class="table-wrap"><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>'

    def probability(value):
        try:
            p = float(value)
            if not isfinite(p): return 'Not available'
            return '<0.0001' if p < 0.0001 else f'{p:.4f}'
        except (TypeError, ValueError):
            return 'Not available'

    def comparisons(rows, binary):
        scale = 100 if binary else 1
        data = []
        for row in rows:
            baseline = row['Control mean / rate']
            lift = num(row['Difference'] / baseline, 2, 100) + '%' if baseline is not None and baseline > 0 else 'Not available'
            data.append([row['Test option'], f"{row['Control N']:,}", f"{row['Test N']:,}",
                         num(row['Control mean / rate'], 2, scale), num(row['Test mean / rate'], 2, scale),
                         num(row['Difference'], 2, scale), lift,
                         f"{num(row['CI lower'], 2, scale)} to {num(row['CI upper'], 2, scale)}",
                         num(row['Test statistic']), probability(row['Raw p-value']), probability(row['Adjusted p-value']),
                         'Significant' if row['Significant'] else 'Not significant', row['Method']])
        return table(['Test', 'Control N', 'Test N', 'Control mean', 'Test mean', 'Difference', 'Lift', 'Difference CI',
                      'Test statistic', 'Raw p', 'Displayed p', 'Result', 'Method'], data)

    title = report.get('name') or 'Customer experiment'
    metrics = report['metrics']
    options = [['Source', report['source']], ['Control strategy', report['control']],
               ['Partners', ', '.join(map(str, report.get('partners') or [])) or 'All selected rows'],
               ['Group by', report.get('group_by') or 'No segment breakdown'],
               ['Significance level', f"{report['alpha']:.1%}"],
               ['Multiple-Test protection', report['protection']],
               ['Calculated at (UTC)', report.get('calculated_at') or 'Not recorded'],
               ['Analysis fingerprint', report.get('fingerprint') or 'Not recorded']]
    sections = []
    for metric in metrics:
        binary = metric['type'] == 'Binary'
        overall = metric['overall']
        tests = len(overall)
        level = 100 * (1 - report['alpha'] / tests) if report['protection'] == 'Holm' and tests else 100 * (1 - report['alpha'])
        unit = 'Averages are percentages and differences are percentage points' if binary else 'Metric units'
        parts = [f"<section><h2>{esc(metric['name'])}</h2><p>{esc(metric['role'])} · {esc(metric['type'])} · Preferred direction: {esc(metric['direction'])}</p>",
                 f"<p class='note'>Overall comparison intervals: {level:g}% confidence. {unit}. Displayed p-values {'use Holm adjustment' if report['protection'] == 'Holm' else 'are unadjusted'}.</p>",
                 '<h3>Overall results</h3>', comparisons(overall, binary)]
        if metric['groups']:
            parts.append(f"<h3>Results by {esc(report.get('group_by') or 'segment')}</h3><p>Each segment uses its own Control. These comparisons are exploratory and unadjusted; intervals use {100*(1-report['alpha']):g}% confidence.</p>")
            for label, rows in metric['groups']:
                parts.append(f'<h4>{esc(label)}</h4>')
                parts.append(comparisons(rows, binary))
        for issue in metric['issues']:
            parts.append(f"<p class='note'>Not calculated: {esc(issue)}</p>")
        insight = metric['insight']
        parts.append(f"<h3>Decision insights</h3><p>{esc(insight['primary'])}</p><ul>")
        for item in insight.get('comparisons', []):
            parts.append(f"<li><strong>{esc(item['test'])} — {esc(item['status'])}:</strong> {esc(item['detail'])}</li>")
        parts.append('</ul>')
        for item in insight.get('observations', []):
            parts.append(f'<p>{esc(item)}</p>')
        parts.append(f"<p>{esc(insight.get('next_step', ''))}</p></section>")
        sections.append(''.join(parts))

    raw_columns = report.get('input_columns', [])
    raw_rows = report.get('input_rows', [])
    input_table = table(raw_columns, [[r.get(col, '') if r.get(col) is not None else '' for col in raw_columns] for r in raw_rows]) if raw_columns else '<p>Input rows were not included.</p>'
    binary_methods = ''
    if any(m['type'] == 'Binary' for m in metrics):
        binary_methods = '<h3>Binary metrics</h3><p>Rate = success count / N. The two-proportion z-test uses the pooled Control and Test rate under the no-difference hypothesis. Sparse counts use Fisher’s exact test; difference intervals use the Newcombe method. The comparison table states the method actually used.</p>'
    banner = '<p class="banner">SYNTHETIC DEMO — data and results are illustrative.</p>' if report.get('synthetic') else ''
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} — Customer Experiment Results</title><style>
body{{font:15px/1.55 Arial,sans-serif;color:#183b52;max-width:1300px;margin:0 auto;padding:38px 30px 70px}}header{{border-top:7px solid #00aeef;padding-top:20px}}h1{{font-size:32px;margin:0;color:#143c55}}h2{{color:#006b9b;border-bottom:1px solid #d7e4eb;padding-bottom:8px;margin-top:35px}}h3{{color:#126282;margin-top:24px}}h4{{margin-bottom:5px}}.eyebrow{{color:#0074a3;letter-spacing:.12em;font-size:12px;font-weight:bold}}.note,.banner{{background:#eaf6fb;border-left:4px solid #00aeef;padding:12px}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #d7e4eb;vertical-align:top}}th{{background:#eaf6fb;color:#143c55}}tbody tr:nth-child(even){{background:#f7fafb}}code{{background:#f2f7f9;padding:2px 4px}}@page{{size:landscape;margin:12mm}}@media print{{body{{padding:0}}.table-wrap{{overflow:visible}}table{{font-size:8px;table-layout:fixed}}th,td{{padding:3px;overflow-wrap:anywhere}}thead{{display:table-header-group}}tr{{break-inside:avoid}}}}
</style></head><body><header><p class="eyebrow">EXPERIMENT PLATFORM / CUSTOMER RESULTS</p><h1>Experiment Results Report</h1><p>{esc(title)}</p></header>{banner}
<section><h2>1. Analysis overview</h2><p>This report compares each Test with the selected Control using the supplied aggregate data. Statistical significance alone does not establish business value.</p>{table(['Setting','Value'], options)}</section>
<section><h2>2. How to read the results</h2><p><strong>Difference</strong> is Test mean minus Control mean. A positive value means a higher observed metric. <strong>Confidence interval</strong> is a range for the difference, not an interval for each group mean. <strong>p-value</strong> measures how surprising the data would be under the no-difference hypothesis; it is not the probability that a strategy works.</p></section>
{''.join(sections)}
<section><h2>3. Data used in this calculation</h2><p>These are the selected aggregate input rows after Partner and metric selection. They are not individual customer records. Rows must represent non-overlapping customers within each metric and use a consistent observation window.</p>{input_table}</section>
<section><h2>4. Calculation logic and formulas</h2><h3>Step 1 — Combine non-overlapping summary rows</h3>
<p>Combined N = sum of row N. Combined mean = sum of (row N × row mean) / combined N.</p><p>Combined sample variance = sum of [(row N − 1) × row variance + row N × (row mean − combined mean)²] / (combined N − 1). The second term accounts for differences in row means.</p>
<h3>Step 2 — Compare each Test with Control</h3><p>Difference = Test mean − Control mean. Relative lift = Difference / Control mean × 100%, when the Control mean is positive.</p>
<p>For continuous metrics the app uses a two-sided Welch t-test: SE = √(Test variance / Test N + Control variance / Control N); t = Difference / SE. Let vT = Test variance / Test N and vC = Control variance / Control N. Welch degrees of freedom = (vT + vC)² / [vT² / (Test N − 1) + vC² / (Control N − 1)]. The difference confidence interval is Difference ± t critical value × SE.</p>{binary_methods}
<h3>Step 3 — Interpret uncertainty and multiple comparisons</h3><p>With Holm selected, overall p-values are adjusted across Tests within each metric. Overall intervals use Bonferroni-adjusted confidence levels and can be more conservative than Holm p-values, so their conclusions can differ. Without protection, p-values and intervals are unadjusted. Additional metrics and segment comparisons are exploratory; they are not adjusted across metrics or segments.</p>
<p>Overall calculations use the observed customer mix in each strategy; they do not enforce planning weights. Comparisons between Tests or differences between segments are not formally tested. Repeated early checks need a separate sequential design.</p></section>
<section><h2>5. Review before deciding</h2><p>Summary data cannot independently verify unique customers, overlapping rows, randomization, SRM, sample completion, outcome maturity, baseline balance, missing outcomes, guardrails or costs. Confirm these before making a business decision. Retain the source and settings for audit.</p></section></body></html>'''.encode('utf-8')
