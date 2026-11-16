"""Self-contained, escaped HTML audit trail for matched geographic experiments."""
from html import escape
import pandas as pd
from .time_series_html_report import _document, _table, _num


def build_geo_design_html(design, history, cfg, plan=None):
    def table(frame):
        return _table(list(frame.columns),frame.where(pd.notna(frame),'Not available').to_numpy().tolist())
    body = _table(['Dataset','Primary outcome','Planned launch','Random seed'],[[
        cfg.dataset_name,cfg.outcome_column,cfg.planned_launch_date,design['seed']]])
    body += _table(['Algorithm','Matching calculated at','Historical input SHA-256'],[[
        design['algorithm_version'],design['created_at'],design['input_fingerprint']]])
    body += '<h2>Design overview</h2>' + _table(['Matched pairs','Assigned DMAs','Independent randomization units','Auxiliary variables'],[[
        len(design['pairs']),len(design['assignment']),design['independent_units'],', '.join(design['covariates']) or 'None']])
    body += _table(['Matching window','Untouched validation window','Distance limit','Spillover cluster'],[[
        design['training_start']+' to '+design['training_end'],design['validation_start']+' to '+design['validation_end'],
        design['max_distance'],design['cluster_column'] or 'None provided']])
    body += '<h2>Candidate and business constraints</h2>' + _table(['Candidates','Mandatory markets','Excluded markets'],[[
        ', '.join(design['candidates']),', '.join(design['required']) or 'None',', '.join(design['excluded']) or 'None']])
    body += '<h2>Market assignment</h2>' + table(design['assignment'])
    body += '<h2>Why these markets were paired</h2>' + table(design['pairs'])
    body += '<h2>Training balance across assigned DMAs</h2>'+table(design['balance'])
    body += '<p>SMD = (Test feature mean − Control feature mean) / square root of ((Test feature sample variance + Control feature sample variance) / 2). Each DMA contributes one training feature value. This is a descriptive DMA-level check; few markets limit its interpretation. No random seeds are searched to improve balance after assignment.</p>'
    body += '<h2>Reserve and excluded markets</h2>'
    body += table(design['reserves']) if not design['reserves'].empty else '<p>Every eligible candidate was paired.</p>'
    body += '<h2>Untouched historical validation</h2>'
    v = design['validation']
    body += _table(['MAE','RMSE','WAPE','A/A effect per period','A/A p-value','A/A alpha','Reference randomizations'],[[
        _num(v['mae']),_num(v['rmse']),_num(v['wape']),_num(v['aa_effect']),_num(v['aa_pvalue'],5),v['alpha'],v['randomizations']]])
    body += '<p>'+escape(v['status']+'. '+v['note'])+'</p>'
    body += table(v['predictions'])
    if plan:
        body += '<h2>Campaign planning assumptions</h2>' + _table(['Absolute effect per Test DMA/period','Target power','Two-sided alpha','Ramp periods','Outcome delay','Maximum requested duration'],[[
            plan['assumed_effect_absolute'],plan['target_power'],plan['alpha'],plan['ramp_periods'],plan['outcome_delay_periods'],plan['max_duration_requested']]])
        output = plan['table'].rename(columns={'Required Test DMAs':'Selected Test DMAs','Required Control DMAs':'Selected Control DMAs','Required Total DMAs':'Selected total DMAs'})
        body += '<h2>Historical detection simulation</h2><p>'+escape(plan['method_note'])+'</p>'+table(output)
    body += '''<h2>How matching is calculated</h2>
<p>Only pre-launch data is used. The earlier window determines matching and standardization. The later window is held out; its outcome values never determine pair distances, selected pairs or random assignment. Completeness checks use the full pre-launch reporting grid.</p>
<p>For each DMA, the outcome path is divided by its absolute training mean (RMS if its mean is effectively zero). Path distance is the root mean squared difference between these normalized paths. Summary features include the training mean, linear change across the training window and sample standard deviation. Auxiliary variables contribute their training mean and linear change. Each summary feature is standardized by its cross-market training standard deviation; constant features are omitted.</p>
<p>Within each block, distance is the root mean squared feature difference. Each block distance is divided by its median candidate-pair distance (one if the median is effectively zero). Final distance is the weighted sum of these block distances. The weights below are app defaults, not universal industry thresholds.</p>'''
    settings = design['distance_settings']
    body += _table(['Distance block','Weight','Training distance scale'],list(zip(settings['distance_blocks'],settings['block_weights'],settings['block_scales'])))
    body += '''<p>The optimization first maximizes the number of feasible pairs, then minimizes their total matching distance. Every DMA can appear in at most one pair; mandatory markets must be assigned; pairs beyond the distance limit are disallowed. Markets sharing a spillover cluster cannot be placed in opposite arms. This can link several pairs into one assignment component. A seeded fair coin flips each independent component. No rerandomization is performed to improve a historical p-value.</p>
<h2>How validation and testing are calculated</h2>
<p>Predicted Test outcome = observed Control outcome + the average Test-minus-Control difference in the matching window. Prediction error = actual Test − predicted Test. MAE = average absolute error; RMSE = square root of average squared error; WAPE = total absolute error / total absolute actual Test outcome.</p>
<p>For each pair, historical A/A change = (validation Test mean − matching Test mean) − (validation Control mean − matching Control mean). The statistic is the mean of these pair changes. A two-sided randomization p-value compares its absolute value with values obtained by flipping whole independent assignment components, respecting spillover constraints. Exact enumeration includes ties; Monte Carlo uses (extreme draws + 1) / (draws + 1). With few independent components, small p-values may be impossible.</p>
<p>A/A checks are diagnostic. Lack of significance does not establish equivalence. Multiple windows in planning can overlap and are not independent replications. Freeze the design before launch; changing variables, candidates or seeds repeatedly after validation weakens the holdout. Pairing does not establish that spillover is absent; cluster definitions and event reviews require business input. Analysis must use the actual executed assignment.</p>'''
    body += '<h2>Training features used</h2>'+table(design['features'])
    inputs = history.loc[history._dma.isin(design['candidates']),list(dict.fromkeys(['_date','_dma','_outcome',*design['covariates'],*([design['cluster_column']] if design['cluster_column'] else [])]))].copy()
    inputs['_date'] = pd.to_datetime(inputs['_date']).dt.strftime('%Y-%m-%d')
    body += '<h2>Historical input values</h2>'+table(inputs)
    return _document('Geographic Experiment Design Report',body)
