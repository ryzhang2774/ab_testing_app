"""Create a shareable, aggregate-only customer experiment results report."""
from io import BytesIO
from math import isfinite
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

BLUE = colors.HexColor('#006B9B')
NAVY = colors.HexColor('#143C55')
MUTED = colors.HexColor('#526A79')
LINE = colors.HexColor('#D7E4EB')
PALE = colors.HexColor('#EAF6FB')


def build_results_report(report: dict) -> bytes:
    """Return a PDF using the exact comparisons and insights displayed in the UI."""
    font = 'CustomerResultsSans'
    if font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font, str(Path(__file__).parent / 'assets/fonts/BusinessSansSC-Regular.ttf')))
    styles = getSampleStyleSheet()
    for name, values in {
        'RT': dict(fontSize=23, leading=29, textColor=NAVY, spaceAfter=8),
        'RK': dict(fontSize=8, leading=11, textColor=BLUE, spaceAfter=8),
        'RS': dict(fontSize=14, leading=19, textColor=NAVY, spaceBefore=12, spaceAfter=7, keepWithNext=True),
        'RH': dict(fontSize=10, leading=14, textColor=BLUE, spaceBefore=8, spaceAfter=5, keepWithNext=True),
        'RB': dict(fontSize=9, leading=13, textColor=NAVY, spaceAfter=6),
        'RM': dict(fontSize=8, leading=11, textColor=MUTED, spaceAfter=5),
        'RC': dict(fontSize=7.5, leading=10, textColor=NAVY),
        'RCH': dict(fontSize=7.2, leading=9, textColor=colors.white),
    }.items():
        styles.add(ParagraphStyle(name=name, fontName=font, wordWrap='CJK', **values))
    def p(value, style='RB'):
        return Paragraph(escape(str(value)).replace('\n', '<br/>'), styles[style])
    def table(rows, widths):
        t = Table([[p(cell, 'RCH' if i == 0 else 'RC') for cell in row] for i, row in enumerate(rows)],
                  colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), BLUE), ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F9FB')]),
            ('LINEBELOW', (0,1), (-1,-1), .3, LINE),
            ('LEFTPADDING', (0,0), (-1,-1), 5), ('RIGHTPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6)]))
        return t
    def number(value, scale=1):
        try:
            value = float(value) * scale
            return f'{value:,.2f}' if isfinite(value) else 'N/A'
        except (ValueError, TypeError):
            return 'N/A'
    def probability(value):
        try:
            value = float(value)
            return ('<0.001' if value < .001 else f'{value:.3f}') if isfinite(value) else 'N/A'
        except (ValueError, TypeError):
            return 'N/A'
    def comparison_table(rows, binary):
        scale = 100 if binary else 1
        output = [['Test strategy', 'N\nC / T', 'Average\nC / T', 'Change', 'Difference\nconfidence interval', 'p-value', 'Result']]
        for row in rows:
            output.append([row['Test option'], f"{int(row['Control N']):,}\n{int(row['Test N']):,}",
                number(row['Control mean / rate'],scale)+'\n'+number(row['Test mean / rate'],scale),
                number(row['Difference'],scale),
                number(row['CI lower'],scale)+' to '+number(row['CI upper'],scale),
                probability(row['Adjusted p-value']), 'Significant' if row['Significant'] else 'Not significant'])
        return table(output, [87,55,74,54,99,54,72])
    def average_plot(rows, binary, control):
        # The graph is descriptive; confidence intervals for differences remain in the table.
        values = [(str(control), float(rows[0]['Control mean / rate']))]
        values += [(str(row['Test option']), float(row['Test mean / rate'])) for row in rows]
        if len(values) > 12 or any(not isfinite(v) for _, v in values):
            return None
        values.sort(key=lambda item: item[1])
        scale = 100 if binary else 1
        values = [(label, value * scale) for label,value in values]
        lo, hi = min(0, *(v for _,v in values)), max(0, *(v for _,v in values))
        span = hi-lo or 1
        left, width = 150, 275
        zero = left + width * (-lo) / span
        drawing = Drawing(495, 22*len(values)+12)
        drawing.add(Line(zero, 4, zero, 22*len(values)+4, strokeColor=LINE))
        for i,(label,value) in enumerate(values):
            y = 22*(len(values)-i-1)+6
            end = left + width*(value-lo)/span
            drawing.add(String(0,y+3,label[:24]+('...' if len(label)>24 else ''),fontName=font,fontSize=8,fillColor=NAVY))
            drawing.add(Rect(min(zero,end),y,max(abs(end-zero),.5),13,fillColor=BLUE if label == str(control) else colors.HexColor('#00AEEF'),strokeColor=None))
            drawing.add(String(min(max(end+5, left), 435),y+3,number(value),fontName=font,fontSize=8,fillColor=NAVY))
        return drawing
    def footer(canvas, doc):
        width, height = A4
        canvas.saveState()
        canvas.setFillColor(BLUE); canvas.rect(0,height-8,width,8,fill=1,stroke=0)
        canvas.setFont(font,8); canvas.setFillColor(MUTED)
        canvas.drawString(50,height-30,'EXPERIMENT PLATFORM  /  CUSTOMER RESULTS REPORT')
        canvas.setStrokeColor(LINE); canvas.line(50,42,width-50,42)
        canvas.drawString(50,28,'Results for review  |  Not an approval or rollout decision')
        canvas.drawRightString(width-50,28,str(doc.page)); canvas.restoreState()
    stream = BytesIO()
    doc = SimpleDocTemplate(stream, pagesize=A4, leftMargin=50, rightMargin=50, topMargin=48, bottomMargin=55,
                            title='Customer Experiment Results Report', author='Experiment Platform')
    story = [Spacer(1,12), p('CUSTOMER EXPERIMENT  /  RESULTS', 'RK'),
             p('Experiment Results Report','RT'), p(report.get('name') or 'Customer experiment','RB')]
    if report.get('synthetic'):
        banner = Table([[p('SYNTHETIC DEMO  |  These data and results are illustrative.', 'RB')]],colWidths=[495])
        banner.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('LINEBEFORE',(0,0),(0,-1),3,BLUE),
                                    ('LEFTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),8)]))
        story += [banner,Spacer(1,8)]
    story += [p('01  /  Analysis overview','RS'),
              table([['Setting','Value'], ['Input', report['source']+'  |  '+report['input_format']],
                     ['Control strategy',report['control']], ['Partners', ', '.join(map(str, report.get('partners') or [])) or 'All rows in input'],
                     ['Group by',report.get('group_by') or 'No segment breakdown'],
                     ['Metrics', ', '.join(m['name'] for m in report['metrics'])],
                     ['Significance level',str(report['alpha'])], ['Multiple-test protection',report['protection']],
                     ['Results calculated',report.get('calculated_at') or 'Not recorded'],
                     ['Analysis fingerprint',report.get('fingerprint') or 'Not recorded']], [155,340]),
              p('Review required: statistical significance alone does not establish commercial value or approval to launch.','RM')]
    for index, metric in enumerate(report['metrics']):
        binary = metric['type'] == 'Binary'
        intro = ([p('02  /  Overall results', 'RS')] if index == 0 else []) + [p(metric['name'], 'RH'),
                 p(f"{metric['role']}  |  {metric['type']}  |  Preferred direction: {metric['direction']}", 'RM')]
        chart = average_plot(metric['overall'], binary, report['control']) if metric['overall'] else None
        if chart is not None:
            intro += [p('Observed average by strategy (descriptive)', 'RM'), chart, Spacer(1,6)]
        interval_level = 100 * (1 - report['alpha'] / len(metric['overall'])) if report['protection'] == 'Holm' and metric['overall'] else 100 * (1 - report['alpha'])
        intro += [p(f'Overall difference confidence level: {interval_level:g}%.', 'RM'),
                  p('C = Control; T = Test. '+('Averages are percentages; changes and intervals are percentage points.' if binary else 'Averages, changes and intervals are in metric units.')+' p-values follow the selected overall protection.', 'RM'),
                  comparison_table(metric['overall'],binary)]
        if len(metric['overall']) <= 8:
            story.append(KeepTogether(intro))
        else:
            story.extend(intro)
    if any(metric['groups'] or metric['issues'] for metric in report['metrics']):
        story += [p('03  /  Results by segment','RS'),
                  p(f"Each segment is compared with its own Control. Segment results are exploratory and unadjusted; difference confidence level: {100*(1-report['alpha']):g}%.",'RM')]
        for metric in report['metrics']:
            story.append(p(metric['name'],'RH'))
            for label, rows in metric['groups']:
                story += [p(label,'RH'),comparison_table(rows, metric['type'] == 'Binary')]
            for issue in metric['issues']:
                story.append(p('Not calculated: '+str(issue),'RM'))
    story.append(p('04  /  Decision insights','RS'))
    for metric in report['metrics']:
        story += [p(metric['name'],'RH'),p(metric['insight']['primary'])]
        for item in metric['insight']['comparisons']:
            story.append(p(f"{item['test']}  |  {item['status']}  |  {item['detail']}",'RM'))
        for item in metric['insight']['observations']:
            story.append(p(item,'RM'))
    story.append(p('05  /  Detailed test statistics','RS'))
    for metric in report['metrics']:
        story.append(p(metric['name'],'RH'))
        detailed = [['Scope / test','Method','Statistic','Raw p','Displayed p']]
        for label, rows in [('Overall',metric['overall'])]+metric['groups']:
            for row in rows:
                detailed.append([f"{label} / {row['Test option']}",row['Method'],number(row['Test statistic']),probability(row['Raw p-value']),probability(row['Adjusted p-value'])])
        story.append(table(detailed,[170,120,65,70,70]))
    story += [p('06  /  Methods and formulas','RS'),
              p('Each comparison uses the selected metric and observed customers. The formulas below describe the calculations used in this report.'),
              p('Combining non-overlapping summaries','RH'),
              p('Combined N = sum of each row N'),
              p('Combined mean = sum of (row N x row mean) / combined N'),
              p('Combined sample variance = sum of [(row N - 1) x row variance + row N x (row mean - combined mean)^2] / (combined N - 1)'),
              p('Combining rows is valid only when they represent non-overlapping customers measured over a consistent outcome window. Sample variance is the unbiased variance with N - 1 in its denominator.','RM'),
              p('Continuous metrics: Welch two-sample t-test','RH'),
              p('Difference = Test mean - Control mean'),
              p('Lift (%) = 100 x Difference / Control mean, when Control mean is positive'),
              p('SE = sqrt(Test variance / Test N + Control variance / Control N)'),
              p('t statistic = Difference / SE'),
              p('Welch degrees of freedom = (vT + vC)^2 / [vT^2 / (Test N - 1) + vC^2 / (Control N - 1)], where vT = Test variance / Test N and vC = Control variance / Control N.'),
              p('Difference confidence interval = Difference +/- t critical value x SE. The two-sided p-value comes from the t distribution using the Welch degrees of freedom.'),
              p('Multiple comparisons and interpretation','RH'),
              p('If Holm protection is selected, overall p-values are adjusted across Tests within each metric. Overall intervals use a Bonferroni-adjusted confidence level: 100 x (1 - alpha / number of Tests)%. Consequently, p-values and intervals can lead to different conclusions. Without protection, the confidence level is 100 x (1 - alpha)%.'),
              p('Segment comparisons are unadjusted and exploratory. Additional metrics are exploratory and are not adjusted across metrics. Confidence intervals describe the Test minus Control difference, not individual group averages.'),
              p('Overall results reflect the observed population mix. Test-to-Test comparisons and differences between segments are not formally tested. Repeated early significance checks require a separately planned sequential method.'),
              p('Randomization, SRM, sample completion, outcome maturity, baseline balance, missing outcomes, guardrails and costs have not been verified by this report. Review them before a decision.'),
              p('Keep the source data and analysis settings for audit and reproducibility. The report contains aggregate results, not customer-level records.','RM')]
    if any(metric['type'] == 'Binary' for metric in report['metrics']):
        story += [p('Binary metrics','RH'),
                  p('Rate = successes / N. Difference = Test rate - Control rate. The two-proportion z-test uses pooled rate = (Test successes + Control successes) / (Test N + Control N) and SE under the null = sqrt(pooled rate x (1 - pooled rate) x (1 / Test N + 1 / Control N)); z = Difference / SE under the null.'),
                  p('Sparse counts use Fisher exact test; binary difference intervals use the Newcombe method. See the detailed table for the method used for each comparison.')]
    if report['input_format'] == 'Summary statistics':
        story.append(p('Summary data cannot independently verify unique customers, overlapping rows, assignment integrity or missing outcomes.','RM'))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return stream.getvalue()
