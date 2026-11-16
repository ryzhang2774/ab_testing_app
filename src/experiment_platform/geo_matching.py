"""Pre-launch multivariate geo design with untouched historical validation."""
from itertools import combinations, product
from datetime import datetime
import hashlib
import re
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def matching_candidates(panel, excluded=()):
    """Suggest numeric business covariates, excluding roles, IDs and delivery fields."""
    result = []
    for col in panel:
        name = re.sub(r'([a-z0-9])([A-Z])',r'\1 \2',str(col)).lower().replace('_', ' ')
        if col in excluded or str(col).startswith('_'):
            continue
        if any(word in name for word in ('date', 'week', 'month', 'group', 'assignment', 'pair', 'flag', 'campaign', 'impression', ' id', 'score id')):
            continue
        if set(name.split()) & {'id','uuid','identifier'} or name in {'customerid','accountid','marketid','geoid'}:
            continue
        values = pd.to_numeric(panel[col], errors='coerce')
        if values.notna().mean() >= .8 and np.isfinite(values.dropna()).all() and values.nunique() > 1:
            result.append(col)
    return result


def _standardize(frame):
    sd = frame.std(ddof=0)
    keep = sd.gt(1e-10)
    return (frame.loc[:, keep] - frame.loc[:, keep].mean()) / sd[keep]


def _distances(train, covariates):
    outcome = train.pivot(index='_date', columns='_dma', values='_outcome').astype(float).sort_index().sort_index(axis=1)
    names = list(outcome.columns)
    mean = outcome.mean()
    scale = mean.abs().where(mean.abs() > 1e-8, np.sqrt((outcome**2).mean())).clip(lower=1e-8)
    normalized = outcome.div(scale)
    # Slope spans the full training window, preserving economic scale information.
    time = np.linspace(0, 1, len(outcome))
    slopes = pd.Series(np.polyfit(time, outcome.to_numpy(), 1)[0], index=names)
    features = pd.DataFrame({'Outcome mean': mean, 'Outcome trend': slopes,
                             'Outcome volatility': outcome.std(ddof=1)}, index=names)
    primary = _standardize(features)
    auxiliary = pd.DataFrame(index=names)
    for col in covariates:
        wide = train.pivot(index='_date', columns='_dma', values=col).reindex(columns=names).astype(float)
        auxiliary[col + ' · mean'] = wide.mean()
        auxiliary[col + ' · trend'] = np.polyfit(time, wide.to_numpy(), 1)[0]
    auxiliary_raw = auxiliary.copy()
    auxiliary = _standardize(auxiliary)
    def rms(v):
        return float(np.sqrt(np.mean(v**2))) if len(v) else 0.
    blocks = {}
    for i, j in combinations(range(len(names)), 2):
        shape = rms(normalized.iloc[:, i].to_numpy() - normalized.iloc[:, j].to_numpy())
        blocks[i, j] = [shape, rms(primary.iloc[i].to_numpy() - primary.iloc[j].to_numpy()),
                         rms(auxiliary.iloc[i].to_numpy() - auxiliary.iloc[j].to_numpy())]
    raw = np.asarray(list(blocks.values()))
    # Put the three distance blocks on comparable scales using training data only.
    typical = np.median(raw, axis=0)
    typical = np.where(typical > 1e-10, typical, 1.)
    weights = np.array([.5, .25, .25]) if len(auxiliary.columns) else np.array([2/3, 1/3, 0.])
    costs = {edge: float(np.dot(values / typical, weights)) for edge, values in blocks.items()}
    return names, costs, pd.concat([features,auxiliary_raw],axis=1), outcome, {'distance_blocks': ['Normalized outcome path', 'Outcome level, trend and volatility', 'Auxiliary means and trends'],
        'block_weights': weights.tolist(), 'block_scales': typical.tolist(), 'auxiliary_features': list(auxiliary.columns)}


def _optimal_pairs(names, costs, required, clusters, max_distance):
    """Maximum feasible cardinality, then minimum total distance; never greedy."""
    edges = [(i, j) for i, j in costs if costs[i, j] <= max_distance and
             not (clusters.get(names[i]) and clusters.get(names[i]) == clusters.get(names[j]))]
    if not edges:
        raise ValueError('No compatible pairs remain. Review candidates, distance limit or spillover clusters.')
    n, m = len(names), len(edges)
    matrix = lil_matrix((n + 2*m + n, m+n))
    low, high = [], []
    for i, name in enumerate(names):
        for k, edge in enumerate(edges):
            if i in edge:
                matrix[i, k] = 1
        low.append(1 if name in required else 0)
        high.append(1)
    row = n
    for k, (i, j) in enumerate(edges):
        matrix[row, m+i] = matrix[row, m+j] = 1
        matrix[row, k] = -1
        low.append(0); high.append(np.inf); row += 1
        matrix[row, m+i] = matrix[row, m+j] = 1
        matrix[row, k] = 1
        low.append(-np.inf); high.append(2); row += 1
    roots = {}
    for i, name in enumerate(names):
        cluster = clusters.get(name)
        if cluster and cluster in roots:
            matrix[row, m+i], matrix[row, m+roots[cluster]] = 1, -1
            low.append(0); high.append(0); row += 1
        elif cluster:
            roots[cluster] = i
    matrix = matrix[:row].tocsr()
    constraint = LinearConstraint(matrix, low, high)
    bounds = Bounds(np.zeros(m+n), np.ones(m+n))
    objective = np.r_[-np.ones(m), np.zeros(n)]
    first = milp(objective, integrality=np.ones(m+n), bounds=bounds, constraints=constraint,
                 options={'time_limit': 20, 'mip_rel_gap': 0.})
    if not first.success:
        raise ValueError('A complete optimization could not be certified. Review mandatory markets or spillover constraints, or reduce the candidate set.')
    count = int(np.rint(first.x[:m].sum()))
    cardinality = LinearConstraint(np.r_[np.ones(m), np.zeros(n)][None, :], count, count)
    result = milp(np.r_[[costs[e] for e in edges], np.zeros(n)], integrality=np.ones(m+n),
                  bounds=bounds, constraints=[constraint, cardinality], options={'time_limit':20, 'mip_rel_gap':0.})
    if not result.success:
        raise ValueError('The matching search did not finish with a certified solution. Reduce candidates or review constraints.')
    chosen = [e for e, value in zip(edges, result.x[:m]) if value > .5]
    if len(chosen) < 2:
        raise ValueError('At least two compatible pairs are needed. Add candidates or review the matching constraints.')
    return chosen, np.rint(result.x[m:]).astype(int)


def _components(names, pairs, clusters):
    parent = {name: name for name in names}
    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    def union(a, b):
        parent[root(a)] = root(b)
    for a, b in pairs:
        union(a, b)
    roots = {}
    for name in names:
        cluster = clusters.get(name)
        if cluster:
            if cluster in roots:
                union(name, roots[cluster])
            roots[cluster] = name
    labels = {r:i for i, r in enumerate(sorted({root(name) for name in names}))}
    return {name:labels[root(name)] for name in names}


def build_matched_design(history, covariates=(), candidates=None, required=(), excluded=(),
                         cluster_column=None, seed=91, max_distance=3., holdout_fraction=.25):
    """history must already be strictly before launch. No post outcomes are used."""
    data = history.copy()
    if not {'_date', '_dma', '_outcome'}.issubset(data):
        raise ValueError('Provide a prepared DMA/date/outcome panel.')
    if data.duplicated(['_date', '_dma']).any():
        raise ValueError('Resolve duplicate DMA/date rows before matching.')
    dates = pd.Index(sorted(data._date.unique()))
    if len(dates) > 2:
        differences = np.diff(pd.DatetimeIndex(dates).asi8)
        # Monthly calendars have unequal day lengths; period ordinals detect gaps.
        monthly = pd.DatetimeIndex(dates).to_period('M')
        if len(monthly.unique()) == len(dates) and np.median(differences)/86400e9 >= 27:
            complete = np.diff(monthly.asi8).tolist() == [1]*(len(dates)-1)
        else:
            step = int(pd.Series(differences).mode().iloc[0])
            complete = bool(np.all(differences == step))
        if not complete:
            raise ValueError('The shared reporting calendar has missing or irregular periods. Provide a regular daily, weekly or monthly panel before matching.')
    if not .1 <= holdout_fraction <= .5 or not 0 < max_distance < np.inf:
        raise ValueError('Use a holdout fraction from 10% to 50% and a positive distance limit.')
    all_names = sorted(data._dma.unique())
    candidates = set(all_names if candidates is None else candidates)
    required, excluded = set(required), set(excluded)
    if (candidates | required | excluded) - set(all_names):
        raise ValueError('Candidate and constraint market names must exist in the uploaded data.')
    if required & excluded or not required.issubset(candidates):
        raise ValueError('Mandatory markets must be candidates and cannot also be excluded.')
    covariates = list(dict.fromkeys(covariates))
    if any(col not in data or col in {'_date','_dma','_outcome'} for col in covariates):
        raise ValueError('Select valid auxiliary matching columns.')
    if len(dates) < 12:
        raise ValueError('Use at least 12 pre-launch periods: at least 8 for matching and 4 for untouched validation.')
    holdout = max(4, int(np.ceil(len(dates)*holdout_fraction)))
    if len(dates) - holdout < 8:
        holdout = len(dates) - 8
    training_dates, validation_dates = dates[:-holdout], dates[-holdout:]
    reserves, eligible, clusters = [], [], {}
    for name in all_names:
        rows = data.loc[data._dma.eq(name)]
        reason = 'Not selected as a candidate' if name not in candidates else 'Excluded by user' if name in excluded else None
        if reason is None and (len(rows) != len(dates) or rows._date.nunique() != len(dates)):
            reason = 'Incomplete pre-launch time coverage'
        if reason is None:
            for col in ['_outcome'] + covariates:
                values = pd.to_numeric(rows[col], errors='coerce')
                if values.isna().any() or not np.isfinite(values).all():
                    reason = f'Missing or non-numeric values: {col}'; break
                data.loc[rows.index, col] = values.to_numpy()
        if reason is None and cluster_column:
            if cluster_column not in rows:
                raise ValueError('The selected spillover cluster column is missing.')
            labels = rows[cluster_column].dropna().astype(str).str.strip()
            labels = labels[labels.ne('')].unique()
            if len(labels) > 1:
                raise ValueError(f'{name}: spillover cluster must stay constant over time.')
            if len(labels):
                clusters[name] = labels[0]
        if reason:
            if name in required:
                raise ValueError(f'Mandatory market {name}: {reason}.')
            reserves.append({'DMA':name, 'Reason':reason})
        else:
            eligible.append(name)
    if len(eligible) < 4:
        raise ValueError('At least four eligible markets with complete historical data are needed.')
    usable = data.loc[data._dma.isin(eligible)]
    train = usable.loc[usable._date.isin(training_dates)]
    names, costs, features, outcome, settings = _distances(train, covariates)
    pairs, orientation = _optimal_pairs(names, costs, required, clusters, max_distance)
    paired_names = [(names[i], names[j]) for i,j in pairs]
    units = _components(names, paired_names, clusters)
    rng = np.random.default_rng(seed)
    flips = {unit: int(rng.integers(2)) for unit in sorted(set(units.values()))}
    assigned, details = [], []
    for pair_id, ((i,j),(a,b)) in enumerate(zip(pairs, paired_names), 1):
        role = {}
        for pos, name in [(i,a),(j,b)]:
            role[name] = 'Test' if orientation[pos] ^ flips[units[name]] else 'Control'
            assigned.append({'DMA':name,'Group':role[name],'Pair':str(pair_id),'Randomization unit':units[name]})
        correlation = outcome[a].corr(outcome[b])
        gap = float(abs(outcome[a].mean()-outcome[b].mean()) / max(abs(outcome[a].mean()),abs(outcome[b].mean()),1e-8)*100)
        scales = [abs(outcome[name].mean()) if abs(outcome[name].mean()) > 1e-8 else max(float(np.sqrt((outcome[name]**2).mean())),1e-8) for name in (a,b)]
        path_gap = float(np.sqrt(np.mean((outcome[a]/scales[0]-outcome[b]/scales[1])**2)))
        details.append({'Pair':str(pair_id), 'Test DMA':a if role[a]=='Test' else b,
            'Control DMA':b if role[b]=='Control' else a, 'Matching distance':costs[i,j],
            'Training correlation':float(correlation) if np.isfinite(correlation) else None,
            'Mean gap (%)':gap,
            'Basis':f'Mean gap {gap:.1f}%; normalized path difference {path_gap:.3f}. '
                    + ('Also matched on '+', '.join(covariates)+'.' if covariates else 'Outcome-only matching.')})
    used = {row['DMA'] for row in assigned}
    for name in eligible:
        if name not in used:
            has_edge = any(name in (names[i],names[j]) and cost <= max_distance for (i,j),cost in costs.items())
            reserves.append({'DMA':name,'Reason':'Reserve after global pairing and constraints' if has_edge else 'No match within the distance limit'})
    assignment = pd.DataFrame(assigned).sort_values('DMA').reset_index(drop=True)
    # Independent flips are connected components, not pairs, when clusters link pairs.
    active_units = sorted(assignment['Randomization unit'].unique())
    assignment['Randomization unit'] = assignment['Randomization unit'].map({u:i for i,u in enumerate(active_units)})
    result = {'assignment':assignment,'pairs':pd.DataFrame(details),'reserves':pd.DataFrame(reserves,columns=['DMA','Reason']),
        'algorithm_version':'multivariate-paired-v1','created_at':datetime.now().astimezone().isoformat(timespec='seconds'),
        'input_fingerprint':hashlib.sha256(data.sort_values(['_dma','_date']).to_csv(index=False).encode()).hexdigest(),
        'training_start':str(pd.Timestamp(training_dates[0]).date()),'training_end':str(pd.Timestamp(training_dates[-1]).date()),
        'validation_start':str(pd.Timestamp(validation_dates[0]).date()),'validation_end':str(pd.Timestamp(validation_dates[-1]).date()),
        'training_periods':len(training_dates),'validation_periods':len(validation_dates),'seed':int(seed),
        'covariates':covariates,'cluster_column':cluster_column,'max_distance':float(max_distance),
        'features':features.reset_index(names='DMA'), 'distance_settings':settings,
        'required':sorted(required),'excluded':sorted(excluded),'candidates':sorted(candidates),
        'independent_units':len(active_units),'holdout_fraction':float(holdout_fraction)}
    result['validation'] = validate_matched_design(usable, result)
    balance = []
    roles = assignment.set_index('DMA').Group
    for col in features.columns:
        test = features.loc[roles[roles.eq('Test')].index,col]
        control = features.loc[roles[roles.eq('Control')].index,col]
        difference = float(test.mean()-control.mean())
        sd = float(np.sqrt((test.var(ddof=1)+control.var(ddof=1))/2))
        balance.append({'Feature':col,'Test mean':float(test.mean()),'Control mean':float(control.mean()),
                        'SMD':difference/sd if sd>1e-10 else (0. if abs(difference)<1e-10 else None)})
    result['balance'] = pd.DataFrame(balance)
    return result


def _signs(units, seed, draws=1024):
    if units <= 10:
        return np.array(list(product([-1.,1.], repeat=units))), True
    return np.random.default_rng(seed).choice([-1.,1.],size=(draws,units)), False


def _paired_changes(data, design, dates):
    assignment = design['assignment'].set_index('DMA')
    wide = data.pivot(index='_date',columns='_dma',values='_outcome').sort_index()
    baseline = wide.loc[wide.index <= pd.Timestamp(design['training_end'])].mean()
    observed = wide.loc[dates].mean() - baseline
    changes, units = [], []
    for pair in design['pairs'].to_dict('records'):
        test, control = pair['Test DMA'], pair['Control DMA']
        changes.append(float(observed[test] - observed[control]))
        units.append(int(assignment.at[test,'Randomization unit']))
    return np.array(changes), np.array(units)


def validate_matched_design(history, design, alpha=.05):
    """A/A checks use only the held-out pre-launch observations; never tune pairs here."""
    dates = sorted(history.loc[history._date.gt(pd.Timestamp(design['training_end'])), '_date'].unique())
    assignment = design['assignment']
    rows = []
    # Fixed pre-baseline difference is a simple paired counterfactual, not a fitted test-period model.
    for pair in design['pairs'].to_dict('records'):
        a, b = pair['Test DMA'], pair['Control DMA']
        wide = history.loc[history._dma.isin([a,b])].pivot(index='_date',columns='_dma',values='_outcome').sort_index()
        before = wide.loc[wide.index <= pd.Timestamp(design['training_end'])]
        after = wide.loc[dates]
        expected = after[b] + (before[a]-before[b]).mean()
        for date, actual, predicted in zip(after.index,after[a],expected):
            rows.append({'Pair':pair['Pair'],'Date':str(pd.Timestamp(date).date()),'Actual Test':float(actual),
                         'Predicted Test':float(predicted),'Error':float(actual-predicted)})
    predictions = pd.DataFrame(rows)
    error = predictions.Error.to_numpy()
    denominator = predictions['Actual Test'].abs().sum()
    changes, units = _paired_changes(history,design,dates)
    signs, exact = _signs(design['independent_units'],design['seed'])
    null = signs[:,units] @ changes / len(changes)
    observed = float(changes.mean())
    tail = int((np.abs(null) >= abs(observed)-1e-10).sum())
    p = tail/len(null) if exact else (tail+1)/(len(null)+1)
    return {'mae':float(np.mean(abs(error))),'rmse':float(np.sqrt(np.mean(error**2))),
        'wape':float(abs(error).sum()/denominator) if denominator else None,
        'aa_effect':observed,'aa_pvalue':float(p),'aa_flagged':bool(p<alpha),'alpha':float(alpha),
        'status':'Review historical mismatch' if p<alpha else 'No A/A flag; limited historical validation',
        'randomizations':len(signs),'exact':exact,'predictions':predictions,
        'note':'One untouched historical A/A window. A non-significant result does not prove equivalence or validate the future false-positive rate.'}


def paired_power_table(history, design, durations, effect_absolute, alpha=.05, seed=91):
    """Inject a constant per-DMA effect into held-out windows; test with design-aware flips."""
    dates = sorted(history.loc[history._date.gt(pd.Timestamp(design['training_end'])), '_date'].unique())
    signs, exact = _signs(design['independent_units'],seed)
    rng = np.random.default_rng(seed+173)
    simulations = signs if len(signs)<=256 else rng.choice([-1.,1.],size=(256,design['independent_units']))
    results = []
    for duration in durations:
        if duration > len(dates):
            continue
        starts = np.unique(np.linspace(0,len(dates)-duration,min(5,len(dates)-duration+1),dtype=int))
        powers, null_rates = [], []
        for start in starts:
            change, units = _paired_changes(history,design,dates[start:start+duration])
            permutations = signs[:,units]
            simulated_changes = simulations[:,units] * change
            for effect, collector in [(0.,null_rates),(effect_absolute,powers)]:
                outcomes = simulated_changes + effect
                actual = np.abs(outcomes.mean(axis=1))
                distribution = np.abs(permutations @ outcomes.T / len(change))
                tails = (distribution >= actual[None,:]-1e-10).sum(axis=0)
                pvalues = tails/len(signs) if exact else (tails+1)/(len(signs)+1)
                collector.append(float((pvalues < alpha).mean()))
        results.append({'duration':int(duration),'power':float(np.mean(powers)),
            'null_rejection_rate':float(np.mean(null_rates)),'windows':len(starts),
            'assignments_per_window':len(simulations),'exact_reference':exact})
    return results


def plan_matched_experiment(panel, design, launch, frequency, effect_type, expected_effect,
                            alpha=.05, target_power=.8, max_duration=26, ramp=0, delay=0):
    """Historical sensitivity for the selected design, without extrapolating extra DMAs."""
    from .geography import _duration_candidates, _add_periods
    history = panel.loc[panel._date < pd.Timestamp(launch)]
    assignment = design['assignment']
    tests = assignment.loc[assignment.Group.eq('Test'),'DMA']
    baseline = float(history.loc[history._dma.isin(tests) & history._date.le(pd.Timestamp(design['training_end'])), '_outcome'].mean())
    effect = abs(baseline)*expected_effect/100 if effect_type=='Relative lift' else expected_effect
    if not np.isfinite(effect) or effect <= 0:
        raise ValueError('Use a positive effect scenario and a nonzero baseline for relative lift.')
    durations = _duration_candidates(frequency,min(int(max_duration),design['validation_periods']))
    simulations = paired_power_table(history,design,durations,float(effect),alpha,design['seed'])
    if not simulations:
        raise ValueError('The held-out history is too short for the selected campaign duration. Provide more history or reduce the duration.')
    counts = assignment.Group.value_counts()
    rows, recommended = [], None
    for result in simulations:
        meets = result['power'] >= target_power and result['null_rejection_rate'] <= alpha
        if meets and recommended is None:
            recommended = result['duration']
            status = 'Shortest evaluated option meeting target'
        else:
            status = 'Meets historical target' if meets else 'Below historical detection target'
        rows.append({'Campaign Duration':result['duration'],'Detection Chance':result['power'],
            'Required Test DMAs':int(counts['Test']),'Required Control DMAs':int(counts['Control']),
            'Required Total DMAs':len(assignment),'Earliest Readout':_add_periods(pd.Timestamp(launch),ramp+result['duration']+delay,frequency).date(),
            'Status':status,'Historical windows':result['windows'],'Null rejection rate':result['null_rejection_rate'],
            'Assignments per window':result['assignments_per_window']})
    return {'table':pd.DataFrame(rows),'recommended_duration':recommended,'current_test_dmas':int(counts['Test']),
        'current_control_dmas':int(counts['Control']),'current_total_dmas':len(assignment),'baseline':baseline,
        'assumed_effect_absolute':float(effect),'assumed_effect_relative':effect/abs(baseline) if baseline else np.nan,
        'historical_periods':history._date.nunique(),'lag1_autocorrelation':None,'matched_design':design,
        'alpha':float(alpha),'target_power':float(target_power),'max_duration_requested':int(max_duration),
        'ramp_periods':int(ramp),'outcome_delay_periods':int(delay),
        'method_note':'Historical sensitivity simulation for a fixed paired design. The matching period fixes each market baseline. '
        'Untouched historical windows provide paired changes. A constant per-Test-DMA effect is injected, and a two-sided randomization test '
        'flips whole independent assignment components. The reference is exact up to 10 components, otherwise Monte Carlo with a +1 correction. '
        'At most five overlapping windows and 256 assignments per window estimate detection frequency. '
        'Overlapping windows are dependent; this is a conditional historical estimate, not a guarantee or an independently calibrated future false-positive rate. '
        'No power is extrapolated beyond the held-out history, and no unobserved additional DMAs are assumed.'}
