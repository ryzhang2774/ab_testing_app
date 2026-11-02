"""Portable customer design report; uses only the Python standard library."""
from html import escape


def build_html_summary(plan):
    def text(value):
        return escape(str(value), quote=True)

    def table(headers, rows):
        head = ''.join(f'<th>{text(value)}</th>' for value in headers)
        body = ''.join('<tr>' + ''.join(f'<td>{text(value)}</td>' for value in row) + '</tr>' for row in rows)
        return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    groups = plan.get('group_plans') or [{'segment': 'Overall', 'plan': plan}]
    c = plan['config']
    metric = next(m for m in c['metrics'] if m['role'] == 'Primary')
    total = sum(int(row['recruitment_n']) for group in groups for row in group['plan']['arm_totals'])
    sections = []
    for group in groups:
        g = group['plan']
        roles = {a['name']: a['role'] for a in g['config']['arms']}
        rows = [[roles[r['arm']], r['arm'], f"{int(r['recruitment_n']):,}"] for r in g['arm_totals']]
        comparisons = [[r['metric'], r['test'], r['control'], f"{r['baseline']:,.4g}", f"{r['detectable_change_or_margin']:,.4g}", f"{r['power']:.1%}"] for r in g['comparisons']]
        inputs = [[r['partner'], r['segment'], r['arm'], r['metric'], f"{r['sample_mean']:,.6g}", f"{r['sample_variance']:,.6g}", f"{int(r['sample_number']):,}"] for r in g['history']]
        weights = [[w['partner'], w['segment'], f"{w['weight']:.2%}"] for w in g['weights']]
        targets = [[r['partner'], r['segment'], roles[r['arm']], r['arm'], f"{int(r['recruitment_n']):,}"] for r in g['targets']]
        sections.append(f"<section><h2>{text(group['segment'])}</h2>"
                        + table(['Role', 'Strategy', 'Required customers'], rows)
                        + '<h3>Comparison verification</h3>' + table(['Metric', 'Test', 'Control', 'Control baseline', 'MDE or margin', 'Power'], comparisons)
                        + '<h3>Population weights</h3>' + table(['Partner', 'Group', 'Weight'], weights)
                        + '<h3>Required customers by Partner</h3>' + table(['Partner', 'Group', 'Role', 'Strategy', 'Required N'], targets)
                        + '<h3>Historical inputs used</h3>' + table(['Partner', 'Group', 'Strategy', 'Metric', 'Mean', 'Variance', 'Historical N'], inputs)
                        + f"<p>Comparison family: {g['family_size']}; alpha per comparison: {g['planning_alpha']:.5g}.</p></section>")
    synthetic = '<p class="notice">Synthetic demo: illustrative data and results.</p>' if c.get('synthetic') else ''
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Experiment design summary</title>
<style>body{{font:15px/1.5 Arial,sans-serif;color:#183b52;max-width:1100px;margin:40px auto;padding:0 24px}}h1,h2{{color:#006b9b}}section{{margin:35px 0}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #d7e4eb}}th{{background:#eaf6fb}}.notice{{background:#eaf6fb;padding:12px}}@media print{{body{{margin:0}}section{{break-inside:avoid}}}}</style></head><body>
<h1>Experiment design summary</h1><p>{text(c['name'])}</p>{synthetic}
<p><strong>Primary metric:</strong> {text(metric['name'])} &nbsp; <strong>MDE:</strong> {text(metric['effect'])} {text(metric['effect_scale'])} &nbsp; <strong>Power:</strong> {c['power']:.0%} &nbsp; <strong>Nominal alpha:</strong> {c['alpha']:.1%} &nbsp; <strong>Total customers:</strong> {total:,}</p>
<p>Group by: {text(plan.get('input_grouping', 'Not recorded'))}. Guardrail: {text(next((m['name'] for m in c['metrics'] if m['role'] == 'Guardrail'), 'None'))}. Multiple comparison protection: {'On' if c['multiplicity'] else 'Off'}. Historical inputs and sample sizes are planning assumptions, not observed experiment effects.</p>
{''.join(sections)}
<section><h2>Calculation method</h2><p>Each Test is compared with its group's Control. Control historical counts determine Partner weights within the group. The relative MDE is applied to the weighted Control mean.</p>
<p>Variance of difference = sum over Partners of weight² × (Control variance / Control N + Test variance / Test N). Welch degrees of freedom and a noncentral t distribution determine power.</p>
<p>Pairwise Neyman allocation starts from the Test-to-Control standard deviation ratio. The largest required Control is shared within that group. Each Test N is the shared Control N multiplied by its ratio, rounded up; the final sample sizes are checked. If enabled, Bonferroni protection covers the comparison family across the selected groups.</p>
<p>Required recruitment N = analyzable N / (1 − missing-outcome fraction), rounded up separately for each cell. Historical variance and MDE assumptions do not establish a causal effect.</p></section>
<section><h2>Reproducibility</h2><p>Calculation time: {text(plan.get('saved_at', 'Not recorded'))}; source: {text(plan.get('input_source', 'Not recorded'))}. Retain the source data and review assumptions before launch.</p></section></body></html>'''.encode('utf-8')
