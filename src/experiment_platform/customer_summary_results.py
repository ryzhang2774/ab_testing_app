"""Inference from non-overlapping per-arm experiment summary statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

from .customer import continuous_summary_results
from .customer_plan_ui import pool_statistics


def normalize_summary(frame):
    aliases = {'assigned option':'strategy', 'assigned_option':'strategy', 'assigned strategy':'strategy',
               'arm':'strategy', 'option':'strategy', 'strategy':'strategy', 'metric':'metric',
               'metric type':'metric_type', 'metric_type':'metric_type', 'sample mean':'sample_mean',
               'sample_mean':'sample_mean', 'mean':'sample_mean', 'sample variance':'sample_variance',
               'sample_variance':'sample_variance', 'variance':'sample_variance', 'sample number':'sample_number',
               'sample_number':'sample_number', 'sample size':'sample_number', 'n':'sample_number',
               'sample sd':'sample_sd', 'sample_sd':'sample_sd', 'sd':'sample_sd', 'standard deviation':'sample_sd',
               'success count':'success_count', 'success_count':'success_count', 'successes':'success_count'}
    data = frame.copy()
    data.columns = [aliases.get(str(c).strip().lower(), str(c).strip()) for c in data.columns]
    if data.columns.duplicated().any(): raise ValueError('Duplicate column names after mapping; rename the columns.')
    for col in ['strategy','metric','sample_number']:
        if col not in data: raise ValueError(f'Missing required column: {col}.')
    if data.empty: raise ValueError('Provide summary rows for Control and at least one Test.')
    if 'metric_type' not in data: data['metric_type'] = 'Continuous'
    for col in ['strategy','metric','metric_type']:
        if data[col].isna().any() or data[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{col}: every row needs a value.')
        data[col] = data[col].astype(str).str.strip()
    data['metric_type'] = data['metric_type'].str.title()
    if not data.metric_type.isin(['Continuous','Binary']).all():
        raise ValueError('Metric type must be Continuous or Binary.')
    if (data.groupby('metric').metric_type.nunique() > 1).any():
        raise ValueError('Each metric must have one consistent metric type.')
    reserved = {'strategy','metric','metric_type','sample_number','sample_mean','sample_variance','sample_sd','success_count'}
    dimensions = [c for c in data.columns if c not in reserved]
    if any(str(c).strip().lower() in {'ym','month','year_month','date','snapshot_date'} for c in dimensions):
        raise ValueError('Do not combine monthly snapshots: provide one non-overlapping summary per combination.')
    for col in dimensions:
        if data[col].isna().any() or data[col].astype(str).str.strip().eq('').any(): raise ValueError(f'{col}: fill in every row.')
        data[col] = data[col].astype(str).str.strip()
    if data.duplicated(['strategy','metric']+dimensions).any():
        raise ValueError('Duplicate summary combinations. Remove duplicates or aggregate non-overlapping data first.')
    n = pd.to_numeric(data.sample_number, errors='coerce')
    if not np.isfinite(n).all() or (n < 2).any() or (n % 1 != 0).any():
        raise ValueError('Sample number must be a whole number of at least 2 per row.')
    data['sample_number'] = n.astype(int)
    continuous = data.metric_type.eq('Continuous')
    for col, label in [('sample_sd','Sample SD'),('sample_variance','Sample variance')]:
        if col in data:
            supplied = data[col].notna() & data[col].astype(str).str.strip().ne('')
            if (supplied & pd.to_numeric(data[col], errors='coerce').isna()).any():
                raise ValueError(f'{label} must contain valid numeric values.')
    for col in ['sample_mean','sample_variance','sample_sd','success_count']:
        if col not in data: data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors='coerce')
    sd = data.loc[continuous, 'sample_sd']
    supplied_sd = sd.notna()
    if (sd[supplied_sd] < 0).any() or not np.isfinite(sd[supplied_sd]).all():
        raise ValueError('Sample SD must be finite and non-negative.')
    both = continuous & data.sample_sd.notna() & data.sample_variance.notna()
    if both.any() and not np.allclose(data.loc[both, 'sample_sd'] ** 2, data.loc[both, 'sample_variance'], rtol=1e-4, atol=1e-8):
        raise ValueError('Sample variance and sample SD do not agree. Supply one or correct the values.')
    use_sd = continuous & data.sample_variance.isna() & data.sample_sd.notna()
    data.loc[use_sd, 'sample_variance'] = data.loc[use_sd, 'sample_sd'] ** 2
    if not np.isfinite(data.loc[continuous,'sample_mean']).all() or not np.isfinite(data.loc[continuous,'sample_variance']).all():
        raise ValueError('Continuous metrics need a finite sample mean and sample variance in every row.')
    if (data.loc[continuous,'sample_variance'] < 0).any(): raise ValueError('Sample variance cannot be negative.')
    binary = ~continuous
    successes = data.loc[binary,'success_count']
    if not np.isfinite(successes).all() or (successes < 0).any() or (successes % 1 != 0).any() or (successes > data.loc[binary,'sample_number']).any():
        raise ValueError('Binary metrics need an integer success count between 0 and sample number.')
    rate = successes / data.loc[binary,'sample_number']
    data.loc[binary,'sample_mean'] = rate
    data.loc[binary,'sample_variance'] = rate * (1-rate) * data.loc[binary,'sample_number'] / (data.loc[binary,'sample_number'] - 1)
    return data, dimensions


def _binary_comparisons(pooled, control, alpha, protection):
    indexed = pooled.set_index('strategy')
    if control not in indexed.index or len(indexed) < 2: raise ValueError('Provide Control and at least one Test.')
    c = indexed.loc[control]
    interval_alpha = alpha / max(len(indexed)-1, 1) if protection != 'None' else alpha
    rows = []
    for arm, row in indexed.iterrows():
        nc, nt = int(c.sample_number), int(row.sample_number)
        xc, xt = int(c.success_count), int(row.success_count)
        rate_c, rate_t = xc/nc, xt/nt
        is_control = arm == control
        out = {'Credit Line':arm,'Is Control':is_control,'N':nt,'Mean / Rate':rate_t,'Effect vs Control':0.,
               'Relative Lift':0.,'CI Lower':np.nan,'CI Upper':np.nan,'Statistic':np.nan,'Raw p-value':np.nan,
               'p-value':np.nan,'Adjusted p-value':np.nan,'Significant':False,'Result':'Control','Method':'Control'}
        if not is_control:
            common = (xc+xt)/(nc+nt)
            if min(nc*common, nc*(1-common), nt*common, nt*(1-common)) < 5:
                statistic, p = stats.fisher_exact([[xt,nt-xt],[xc,nc-xc]])
                method = 'Fisher exact test'
            else:
                statistic, p = proportions_ztest([xt,xc],[nt,nc])
                method = 'Two-proportion z-test'
            lo, hi = confint_proportions_2indep(xt,nt,xc,nc,method='newcomb',compare='diff',alpha=interval_alpha)
            delta = rate_t-rate_c
            out.update({'Effect vs Control':delta,'Relative Lift':delta/rate_c if rate_c else np.nan,
                        'CI Lower':lo,'CI Upper':hi,'Statistic':statistic,'Raw p-value':float(p),'Method':method})
        rows.append(out)
    tests = [r for r in rows if not r['Is Control']]
    raw = [r['Raw p-value'] for r in tests]
    adjusted = multipletests(raw, alpha=alpha, method='holm')[1] if protection == 'Holm' and len(raw)>1 else raw
    for row, value in zip(tests, adjusted):
        row.update({'p-value':float(value), 'Adjusted p-value':float(value), 'Significant':bool(value<alpha),
                    'Result':'Significant' if value<alpha else 'Not significant'})
    return pd.DataFrame(rows).sort_values('Is Control',ascending=False,kind='stable').reset_index(drop=True)


def calculate_summary(data, metric, control, group=None, alpha=.05, protection='Holm'):
    selected = data[data.metric.eq(metric)].copy()
    if selected.empty: raise ValueError('No summaries for this metric.')
    metric_type = selected.metric_type.iloc[0]
    if group and group not in selected: raise ValueError('Selected grouping column is missing.')
    def compare(part, method):
        pooled = pool_statistics(part, ['strategy'])
        if metric_type == 'Binary':
            successes = part.groupby('strategy',sort=False)['success_count'].sum().reset_index()
            pooled = pooled.merge(successes,on='strategy',validate='one_to_one')
            return _binary_comparisons(pooled,control,alpha,method)
        result = continuous_summary_results(pooled.rename(columns={'strategy':'Arm','sample_number':'N',
                        'sample_mean':'Mean','sample_variance':'Variance'}),control,alpha,method)
        result['Method'] = 'Welch t-test'
        return result
    overall = compare(selected, protection)
    groups, issues = {}, []
    if group:
        all_arms = set(selected.strategy)
        for label, part in selected.groupby(group,sort=False):
            missing = sorted(all_arms-set(part.strategy))
            if missing: issues.append(f"{label}: no summaries for {', '.join(missing)}")
            try: groups[str(label)] = compare(part,'None')
            except ValueError as exc: issues.append(f'{label}: {exc}')
    audit = {'metric_type':metric_type, 'accounts':int(overall['N'].sum()), 'summary_only':True}
    return overall, groups, audit, issues
