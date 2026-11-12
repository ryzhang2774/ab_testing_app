"""Pre-launch market assignment for a paired geographic experiment."""
import numpy as np
import pandas as pd


def assign_market_pairs(history, seed, pair_column="None"):
    """Use the prepared pre-launch panel. Each complete pair contributes one market per arm."""
    if history["_dma"].nunique() < 4:
        raise ValueError("Include at least four markets with pre-launch outcomes to plan two groups.")
    if history.duplicated(["_date", "_dma"]).any():
        raise ValueError("Combine duplicate market/date rows before assigning markets.")
    rng = np.random.default_rng(seed)
    if pair_column != "None":
        pairs = history[["_dma", pair_column]].drop_duplicates()
        if pairs[pair_column].isna().any() or pairs["_dma"].duplicated().any() or not pairs.groupby(pair_column).size().eq(2).all():
            raise ValueError("Each supplied pair needs two markets and a stable pair ID for each market.")
        blocks = [(str(pair), sorted(rows["_dma"])) for pair, rows in pairs.groupby(pair_column, sort=True)]
    else:
        common = history.pivot(index="_date", columns="_dma", values="_outcome").replace([np.inf, -np.inf], np.nan).dropna()
        if common.empty:
            raise ValueError("Use a common pre-launch reporting period so the historical market averages are comparable.")
        ranked = common.mean().sort_values(kind="stable").index.tolist()
        # Keep the middle market as a standalone draw if the market count is odd.
        singleton = ranked.pop(len(ranked) // 2) if len(ranked) % 2 else None
        blocks = [(str(i // 2 + 1), ranked[i:i + 2]) for i in range(0, len(ranked), 2)]
        if singleton is not None:
            blocks.append(("", [singleton]))
    rows = []
    for pair, markets in blocks:
        arms = rng.permutation(["Test", "Control"])
        rows.extend((market, arm, pair) for market, arm in zip(markets, arms))
    return pd.DataFrame(rows, columns=["DMA", "Group", "Pair"]).sort_values("DMA").reset_index(drop=True)
