from io import BytesIO
import pandas as pd
from pypdf import PdfReader
from src.experiment_platform.customer_summary_results import normalize_summary, calculate_summary
from src.experiment_platform.customer_ui import _result_rows
from src.experiment_platform.customer_decision import decision_context, rule_based_decision
from src.experiment_platform.customer_results_report import build_results_report


def test_report_uses_aggregate_results_and_labels_binary_units():
    data, _ = normalize_summary(pd.DataFrame([
        {'Segment':'A','Strategy':'Control','Metric':'Conversion','Metric type':'Binary','Sample number':100,'Success count':10},
        {'Segment':'A','Strategy':'Test & more','Metric':'Conversion','Metric type':'Binary','Sample number':100,'Success count':20},
    ]))
    overall, groups, audit, issues = calculate_summary(data,'Conversion','Control','Segment',.05,'Holm')
    rows = _result_rows(overall,'Overall','Binary',data,'strategy','Conversion','Control',None,.05)
    grouped = [(label,_result_rows(result,label,'Binary',data,'strategy','Conversion','Control','Segment',.05)) for label,result in groups.items()]
    context = decision_context(rows,pd.concat([r for _,r in grouped]),'Conversion','Control','Higher',.05,'Holm',False,binary=True)
    report = {'name':'Real example','source':'Upload file','input_format':'Summary statistics','synthetic':False,'control':'Control','group_by':'Segment','alpha':.05,'protection':'Holm','metrics':[
        {'name':'Conversion','role':'Primary metric','type':audit['metric_type'],'direction':'Higher','overall':rows.to_dict('records'),
         'groups':[(k,r.to_dict('records')) for k,r in grouped],'issues':issues,'insight':rule_based_decision(context)}]}
    pdf = build_results_report(report)
    text = '\n'.join(p.extract_text() for p in PdfReader(BytesIO(pdf)).pages)
    assert 'Real example' in text and 'Test & more' in text
    assert '10.00' in text and '20.00' in text and 'percentage points' in text
    assert '95%' in text and 'Segment' in text
    assert 'SYNTHETIC DEMO' not in text and 'cannot independently verify' in text
    assert 'Randomization, SRM' in text
    assert 'Combined sample variance =' in text and 'Welch degrees of freedom' in text
    assert 't statistic = Difference / SE' in text and 'Newcombe method' in text
    report['synthetic'] = True
    demo_text = '\n'.join(p.extract_text() for p in PdfReader(BytesIO(build_results_report(report))).pages)
    assert 'SYNTHETIC DEMO' in demo_text
