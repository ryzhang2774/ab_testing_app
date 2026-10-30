"""Generate a review-ready, traceable customer experiment design package."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

from .business_config import control_name


BLUE = colors.HexColor('#006B9B')
NAVY = colors.HexColor('#143C55')
CYAN = colors.HexColor('#00AEEF')
PALE = colors.HexColor('#EAF6FB')
LINE = colors.HexColor('#D7E4EB')
MUTED = colors.HexColor('#526A79')
WHITE = colors.white


def build_design_package(plan: dict) -> bytes:
    font_dir = Path(__file__).parent / 'assets' / 'fonts'
    regular, math = 'CustomerPlanSans', 'CustomerMathSans'
    if regular not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(regular, str(font_dir / 'BusinessSansSC-Regular.ttf')))
    if math not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(math, str(font_dir / 'DejaVuSans.ttf')))
    styles = getSampleStyleSheet()
    defs = {
        'Kicker': dict(fontName=regular, fontSize=8, leading=11, textColor=BLUE, spaceAfter=8),
        'PackageTitle': dict(fontName=regular, fontSize=24, leading=30, textColor=NAVY, spaceAfter=7, wordWrap='CJK'),
        'Subtitle': dict(fontName=regular, fontSize=11, leading=16, textColor=MUTED, spaceAfter=10, wordWrap='CJK'),
        'Section': dict(fontName=regular, fontSize=15, leading=19, textColor=NAVY, spaceBefore=4, spaceAfter=9, keepWithNext=True),
        'Subsection': dict(fontName=regular, fontSize=10.5, leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=5, keepWithNext=True),
        'BodyPackage': dict(fontName=regular, fontSize=9, leading=13.5, textColor=NAVY, spaceAfter=6, wordWrap='CJK'),
        'SmallPackage': dict(fontName=regular, fontSize=8, leading=11.5, textColor=MUTED, spaceAfter=3, wordWrap='CJK'),
        'CellPackage': dict(fontName=regular, fontSize=8, leading=11, textColor=NAVY, wordWrap='CJK'),
        'HeadPackage': dict(fontName=regular, fontSize=7.5, leading=10, textColor=WHITE, wordWrap='CJK'),
        'CellRight': dict(fontName=regular, fontSize=8, leading=11, textColor=NAVY, alignment=TA_RIGHT, wordWrap='CJK'),
        'EquationPackage': dict(fontName=math, fontSize=10, leading=17, textColor=NAVY, wordWrap='CJK'),
    }
    for name, values in defs.items():
        styles.add(ParagraphStyle(name=name, **values))
    def p(value, style='BodyPackage'):
        return Paragraph(escape(str(value)).replace('\n', '<br/>'), styles[style])
    def table(rows, widths, numeric_columns=()):
        values = []
        for i, row in enumerate(rows):
            values.append([p(cell, 'HeadPackage' if i == 0 else 'CellRight' if j in numeric_columns else 'CellPackage') for j, cell in enumerate(row)])
        result = Table(values, colWidths=widths, repeatRows=1, hAlign='LEFT')
        result.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),BLUE), ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE, colors.HexColor('#F5F9FB')]),
            ('LINEBELOW',(0,0),(-1,0),.6,BLUE), ('LINEBELOW',(0,1),(-1,-1),.3,LINE),
            ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6),
        ]))
        return result
    def section(number, title, description=None):
        output = [p(f'{number}  /  {title}', 'Section')]
        if description:
            output.append(p(description, 'SmallPackage'))
        return output
    def fmt(value, digits=10):
        if value is None:
            return 'Not available'
        if isinstance(value, (float, int)):
            return f'{value:,.{digits}g}'
        return str(value)
    def equation(title, expression, description):
        box = Table([[Paragraph(expression, styles['EquationPackage'])]], colWidths=[495], hAlign='LEFT')
        box.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE), ('LINEBEFORE',(0,0),(0,-1),2,CYAN),
                                 ('LEFTPADDING',(0,0),(-1,-1),10), ('RIGHTPADDING',(0,0),(-1,-1),10),
                                 ('TOPPADDING',(0,0),(-1,-1),7), ('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        return KeepTogether([p(title,'Subsection'), box, Spacer(1,4), p(description,'SmallPackage')])
    def on_page(canvas, doc):
        width, height = A4
        canvas.saveState()
        canvas.setFillColor(BLUE)
        canvas.rect(0, height-8, width, 8, fill=1, stroke=0)
        canvas.setFont(regular, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(50, height-30, 'EXPERIMENT PLATFORM  /  CUSTOMER DESIGN PACKAGE')
        canvas.setStrokeColor(LINE)
        canvas.line(50, 42, width-50, 42)
        canvas.drawString(50, 28, 'Planning estimate  •  No approval or final result is implied')
        canvas.drawRightString(width-50, 28, f'{doc.page}')
        canvas.restoreState()

    c = plan['config']
    primary = next(m for m in c['metrics'] if m['role'] == 'Primary')
    effect_label = (f"{primary['effect']:g}% of the Control baseline" if primary['effect_scale'] == 'Relative %'
                    else f"{primary['effect']:g} metric units ({primary['effect_scale']})")
    control = control_name(c)
    roles = {a['name']: a['role'] for a in c['arms']}
    totals = plan['arm_totals']
    target = sum(int(r['recruitment_n']) for r in totals)
    control_n = next(int(r['recruitment_n']) for r in totals if r['arm'] == control)
    weights = plan['weights']
    partners = list(dict.fromkeys(str(w['partner']) for w in weights))
    groups = list(dict.fromkeys(str(w['segment']) for w in weights))
    comparisons = plan['comparisons']
    source = plan.get('input_source', 'Not recorded')
    grouping = plan.get('input_grouping', 'Not recorded')
    story = [Spacer(1,14), p('CUSTOMER EXPERIMENT   /   PLANNING', 'Kicker'),
             p('Experiment Design Package', 'PackageTitle'), p(c['name'], 'Subtitle')]
    if c.get('synthetic'):
        banner = Table([[p('SYNTHETIC DEMO  |  The data and results in this package are illustrative.', 'CellPackage')]], colWidths=[495])
        banner.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('LINEBEFORE',(0,0),(0,-1),3,CYAN),('PADDING',(0,0),(-1,-1),9)]))
        story += [banner, Spacer(1,12)]
    story += section('01', 'Sample size recommendation', 'Use these targets to plan random assignment. The total includes the Control and every Test option.')
    sample_rows = [['Role','Option','Analyzable N','Required N','Allocation']]
    for r in totals:
        sample_rows.append([roles.get(r['arm'],'Test'), r['arm'], f"{int(r['analyzable_n']):,}", f"{int(r['recruitment_n']):,}", f"{r['recruitment_n']/target:.1%}"])
    sample_rows.append(['Total','All options',f"{sum(int(r['analyzable_n']) for r in totals):,}",f'{target:,}','100.0%'])
    story += [table(sample_rows,[53,170,85,90,97],numeric_columns=(2,3,4)), Spacer(1,10),
              p(f'The shared Control ({control}) needs {control_n:,} customers. Required N includes any configured reserve for missing outcomes.', 'BodyPackage'),
              p('This is a prospective sample-size estimate, not a result or an approval to launch.', 'SmallPackage'), Spacer(1,12)]
    story += section('02', 'Design at a glance')
    overview = [['Design item','Recorded setting'],
                ['Primary metric',primary['name']], ['Minimum detectable effect',effect_label],
                ['Group by',grouping], ['Partners',', '.join(partners)], ['Selected groups',', '.join(groups)],
                ['Control',control], ['Target power / significance',f"{c['power']:.0%} per comparison / {c['alpha']:.1%} nominal alpha"]]
    story += [table(overview,[160,335]), Spacer(1,8), p('The observation window and metric definitions must be confirmed with the data owner if not supplied. No eligibility or recruitment timeline is inferred.', 'SmallPackage')]

    story += [PageBreak()] + section('03','Statistical settings and assumptions')
    guardrails = [m for m in c['metrics'] if m['role'] == 'Guardrail']
    settings = [['Setting','Value / interpretation'],
                ['Experiment owner',c.get('owner') or 'Not provided'],
                ['Data source',source], ['Calculation time (UTC)',plan.get('saved_at','Not recorded')],
                ['Randomization unit',c['unit']], ['Decision scope',c['decision_scope']],
                ['Allocation','Pairwise Neyman with a shared Control'], ['Missing-outcome reserve',f"{c['attrition']:.1%}"],
                ['Multiple comparison protection',f"{'On' if c['multiplicity'] else 'Off'}; family size {plan['family_size']}; alpha per comparison {plan['planning_alpha']:.6g}"],
                ['Target power',f"{c['power']:.1%} for each planned comparison"],
                ['Guardrail', '; '.join(f"{m['name']} ({'included in plan; margin ' + fmt(m['margin']) if m['include_in_plan'] else 'monitoring only'})" for m in guardrails) if guardrails else 'Not selected']]
    if plan.get('group_boundaries'):
        settings.append(['Score boundaries',f"{plan['group_boundaries']} (boundary included in lower group)"])
    story += [table(settings,[160,335]), Spacer(1,10)]
    story += [p('What the plan assumes', 'Subsection'),
              p('The historical Control counts define the Partner and group weights. The experiment must maintain that same mix in Control and Test options.'),
              p('The variance inputs are historical estimates. Historical differences between strategies are not interpreted as causal effects; the MDE is the business-specified target.'),
              p('Power is a per-comparison target, not the probability that all Tests or metrics succeed together. The calculation does not fully incorporate uncertainty in historical moments.'),
              p('The design uses fixed-horizon inference. Repeated early significance checks require a separately specified method. Confirm randomization, assignment integrity and observation-window maturity before analysis.')]
    story += section('04','Calculation verification','Each Test is compared with the same Control. The final integer allocation has been checked against the selected target power.')
    pair_rows = [['Test option','Initial control N','Initial test N','Starting ratio']]
    for r in plan['pairwise']:
        pair_rows.append([r['test'],f"{r['pair_control_n']:,}",f"{r['pair_test_n']:,}",f"{r['neyman_ratio']:.3f}"])
    story += [table(pair_rows,[195,100,100,100],numeric_columns=(1,2,3)), Spacer(1,8)]
    verify_rows = [['Metric / test','Control N','Test N','Difference*','Power']]
    for r in comparisons:
        verify_rows.append([f"{r['metric']} / {r['test']}",f"{r['control_n']:,}",f"{r['test_n']:,}",fmt(r['detectable_change_or_margin'],6),f"{r['power']:.1%}"])
    story += [table(verify_rows,[195,75,75,75,75],numeric_columns=(1,2,3,4)), Spacer(1,5),
              p('* Difference is the planned detectable effect or guardrail tolerance, expressed in metric units. These are model assumptions, not observed effects.', 'SmallPackage')]

    story += [PageBreak()] + section('05','Calculation formulas','Notation is shown for the continuous-metric planning model used in this package.')
    equations = [
        ('A. Combine non-overlapping historical groups',
         'N = Σ n<sub>i</sub>       Mean = Σ (n<sub>i</sub> × mean<sub>i</sub>) / N<br/>Variance = Σ [(n<sub>i</sub> − 1) × variance<sub>i</sub> + n<sub>i</sub> × (mean<sub>i</sub> − Mean)<super>2</super>] / (N − 1)',
         'n is historical customer count; the second variance term captures differences between historical group means.'),
        ('B. Define target population and effect',
         'w<sub>h</sub> = Control historical N<sub>h</sub> / total Control historical N<br/>Baseline = Σ (w<sub>h</sub> × Control mean<sub>h</sub>)<br/>Relative effect in metric units = |Baseline| × MDE (%) / 100',
         'h denotes a Partner and group. The user-specified absolute MDE is used directly.'),
        ('C. Variance and Welch-Satterthwaite degrees of freedom',
         'V = Σ [w<sub>h</sub><super>2</super> × (s<sub>Control,h</sub><super>2</super> / n<sub>Control,h</sub> + s<sub>Test,h</sub><super>2</super> / n<sub>Test,h</sub>)]<br/>df = V<super>2</super> / Σ [v<sub>j</sub><super>2</super> / (n<sub>j</sub> − 1)]',
         's² is historical variance, n is the planned analyzable sample and each v is one variance component in V.'),
        ('D. Neyman ratio and shared Control',
         'Test / Control starting ratio = √(Σ w<sub>h</sub> × s<sub>Test,h</sub><super>2</super>) / √(Σ w<sub>h</sub> × s<sub>Control,h</sub><super>2</super>)',
         'Size each Test against Control; keep the largest Control requirement, then recalculate each Test and verify the final integer sample sizes.'),
        ('E. Power and comparison protection',
         'Noncentrality = effect / √V<br/>Adjusted alpha = alpha / number of comparisons in the family',
         'A noncentral t distribution is used for primary-metric power. The alpha adjustment applies only when protection is enabled. Guardrail planning uses its margin and one-sided non-inferiority power with a conservative critical value.'),
        ('F. Missing-outcome reserve',
         'Recruitment target = round up [analyzable sample / (1 − missing-outcome fraction)]',
         'The reserve is applied separately to each Partner, group and option.'),
    ]
    for args in equations:
        story.append(equation(*args))

    story += [PageBreak()] + section('06','Population mix','Weights are based on historical Control sample counts and are shared across experiment options.')
    weight_rows = [['Partner',grouping,'Weight']]
    for row in weights:
        weight_rows.append([row['partner'],row['segment'],f"{row['weight']:.2%}"])
    story += [table(weight_rows,[165,245,85],numeric_columns=(2,)), Spacer(1,10)]
    story += section('07','Historical inputs','These are the pooled statistics used by the calculation, not a copy of individual customer records.')
    input_rows = [['Partner / group','Option','Metric','Mean','Variance','N']]
    for r in plan['history']:
        input_rows.append([f"{r['partner']} / {r['segment']}",r['arm'],r['metric'],fmt(r['sample_mean'],9),fmt(r['sample_variance'],9),f"{int(r['sample_number']):,}"])
    story += [table(input_rows,[120,105,93,60,67,50],numeric_columns=(3,4,5)), Spacer(1,10)]
    story += section('08','Reproducibility and review notes')
    story += [p(f"Experiment ID: {c.get('experiment_id','Not recorded')}"),
              p(f"Historical input fingerprint: {plan.get('data_fingerprint','Not recorded')}"),
              p(f"Statistical engine code identifier: {plan.get('code_version','Not recorded')}"),
              p('Retain the original source file, data definitions and application version in an approved location. The engine identifier does not cover the interface or input adapters. This package records calculation inputs and assumptions; it does not verify the source, randomization, review or approval.', 'SmallPackage')]
    result = BytesIO()
    doc = SimpleDocTemplate(result, pagesize=A4, leftMargin=50, rightMargin=50, topMargin=46, bottomMargin=55,
                            title='Experiment Design Package', author='Experiment Platform')
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return result.getvalue()
