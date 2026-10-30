"""Illustrative credit-card data shared by the customer planning and analysis demos.

Numbers are synthetic examples, not calibrated portfolio estimates. Monetary
outcomes are per account for one illustrative observation window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PARTNERS = ['Holland', 'PriceLine', 'Princess', 'Carnival']
FICO_GROUPS = ['FICO 650–699', 'FICO 700–749', 'FICO 750+']
PD_GROUPS = ['PD low', 'PD medium', 'PD high']
STRATEGIES = [f'Credit line {amount}' for amount in (1000, 2000, 3000, 4000, 5000)]


def _outcomes(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(frame)
    line = frame['Strategy'].str.extract(r'(\d+)$')[0].astype(float).to_numpy()
    fico = frame['FICO group'].map({FICO_GROUPS[0]: 0, FICO_GROUPS[1]: 1, FICO_GROUPS[2]: 2}).to_numpy()
    pd_index = frame['PD score group'].map(dict(zip(PD_GROUPS, range(3)))).to_numpy()
    risk = frame['PD score group'].map({'PD low': .018, 'PD medium': .055, 'PD high': .13}).to_numpy()
    risk = np.clip(risk * rng.lognormal(0, .15, n), .003, .3)
    shift = frame['Partner'].map(dict(zip(PARTNERS, [-.03, 0, .025, .045]))).to_numpy()
    active = rng.random(n) < np.clip(.83 + .02*fico - .5*risk, .58, .93)
    utilization = rng.beta(2.3, 3.3, n)
    utilization = np.clip(utilization + shift + .035*(1-fico) + np.array([-.035, .015, .075])[pd_index], .015, .98)
    # Different populations have different illustrative borrowing needs and
    # response curves. These assumptions are synthetic, not empirical FICO or
    # PD effects. The upper bound keeps balance within the assigned limit.
    partner_factor = frame['Partner'].map(dict(zip(PARTNERS, [.96, 1.05, .94, 1.02]))).to_numpy()
    capacity = np.array([1650., 2350., 3100.])[fico] * np.array([1.15, 1., .78])[pd_index] * partner_factor
    response_scale = np.array([850., 1000., 1150.])[fico] * np.array([1.08, 1., .88])[pd_index]
    effective_line = np.minimum(line, capacity * (1 - np.exp(-line / response_scale)))
    balance = np.where(active, effective_line * utilization, 0.)
    loss_event = rng.random(n) < np.clip(risk*.34, .001, .15)
    loss = np.where(loss_event, np.minimum(line, rng.gamma(2., 1., n)*(.11*line)), 0.)
    revenue = balance*.085 + rng.normal(0, 18, n) - loss
    return pd.DataFrame({'Baseline revolving balance': np.round(np.where(active, rng.beta(2.3, 3.3, n)*1000, 0), 2),
                         'Revolving Balance': np.round(balance, 2), 'RAR': np.round(revenue, 2),
                         'NCL': np.round(loss, 2), 'Active account (0/1)': active.astype(int)})


def historical_demo(seed: int = 31415) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Every cell has data, but its historical count follows a plausible uneven
    # Partner/FICO/PD mix and varies independently by option.
    pd_mix = [[.13, .32, .55], [.35, .46, .19], [.70, .25, .05]]
    blocks = []
    for p, partner in enumerate(PARTNERS):
        for i, fico in enumerate(FICO_GROUPS):
            for j, pd_group in enumerate(PD_GROUPS):
                for strategy in STRATEGIES:
                    count = int(rng.integers(32, 52) + 300 * [1., .85, .70, .62][p] * pd_mix[i][j])
                    block = pd.DataFrame({'Partner': [partner]*count, 'FICO group': [fico]*count,
                                          'PD score group': [pd_group]*count, 'Strategy': [strategy]*count})
                    blocks.append(pd.concat([block, _outcomes(block, rng)], axis=1))
    records = pd.concat(blocks, ignore_index=True)
    result = records.melt(id_vars=['Partner', 'FICO group', 'PD score group', 'Strategy'],
                          value_vars=['Revolving Balance', 'RAR', 'NCL'], var_name='Metric', value_name='value')
    result = result.groupby(['Partner', 'FICO group', 'PD score group', 'Strategy', 'Metric'], sort=False)['value'].agg(
        **{'Sample mean': 'mean', 'Sample variance': 'var', 'Sample number': 'size'}).reset_index()
    result[['Sample mean', 'Sample variance']] = result[['Sample mean', 'Sample variance']].round(2)
    return result


def customer_demo_results(seed: int = 2718, n: int = 7200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fico = rng.choice(FICO_GROUPS, n, p=[.30, .42, .28])
    pd_mix = np.array([[.13, .32, .55], [.35, .46, .19], [.70, .25, .05]])
    fico_index = pd.Series(fico).map(dict(zip(FICO_GROUPS, range(3)))).to_numpy()
    draws = rng.random(n)
    pd_group = [PD_GROUPS[int(np.searchsorted(np.cumsum(pd_mix[i]), u))] for i, u in zip(fico_index, draws)]
    frame = pd.DataFrame({'Partner':rng.choice(PARTNERS, n, p=[.32, .27, .22, .19]), 'FICO group':fico,
                          'PD score group':pd_group, 'Strategy':rng.choice(STRATEGIES, n, p=[.23, .22, .20, .19, .16])})
    outcomes = _outcomes(frame, rng)
    frame = frame.rename(columns={'Strategy':'Assigned option'})
    frame.insert(0, 'Customer ID', [f'C{i:06d}' for i in range(1, n+1)])
    return pd.concat([frame, outcomes.drop(columns=['Active account (0/1)'])], axis=1)
