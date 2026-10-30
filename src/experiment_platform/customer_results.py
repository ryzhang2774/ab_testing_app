"""Customer-level demo and descriptive, fixed-horizon comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .customer import analyze_customer_records
from .customer_demo import customer_demo_results


def demo_results(seed: int = 2718) -> pd.DataFrame:
    """Synthetic, independent customer records aligned with the planning demo."""
    return customer_demo_results(seed)


def calculate_results(frame, id_column, arm_column, outcome_column, control, group_column=None,
                      alpha=.05, metric_type='Auto', multiplicity_method='Holm', expected_shares=None):
    """Overall inference plus exploratory subgroup readouts, with isolated errors."""
    if group_column and (group_column in {id_column, arm_column, outcome_column} or group_column not in frame):
        raise ValueError('Choose a different column for grouping.')
    overall, audit = analyze_customer_records(frame, id_column, arm_column, outcome_column, control,
                                               alpha=alpha, metric_type=metric_type,
                                               multiplicity_method=multiplicity_method,
                                               expected_shares=expected_shares)
    subgroup, issues = {}, []
    if group_column:
        groups = frame[group_column]
        if groups.isna().any() or groups.astype(str).str.strip().eq('').any():
            raise ValueError(f'{group_column}: fill in all group labels before analysis.')
        all_arms = set(frame[arm_column].astype(str).str.strip())
        for label, part in frame.groupby(group_column, sort=False, dropna=False):
            present = set(part[arm_column].astype(str).str.strip())
            missing = sorted(all_arms - present)
            if missing:
                issues.append(f"{label}: no customers for {', '.join(missing)}")
            try:
                result, _ = analyze_customer_records(part, id_column, arm_column, outcome_column, control,
                    alpha=alpha, metric_type=audit['metric_type'], multiplicity_method='None')
                subgroup[str(label)] = result['All customers']
            except ValueError as exc:
                issues.append(f'{label}: {exc}')
    return overall['All customers'], subgroup, audit, issues
