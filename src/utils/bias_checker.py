import numpy as np
import pandas as pd
from typing import Dict, List

def sector_overconcentration(
    weights: np.ndarray,
    sectors: Dict[str, str],
    tickers: List[str],
    sector_max: float = 0.25,
) -> Dict[str, float]:
    sector_weights = {}
    for w, t in zip(weights, tickers):
        s = sectors[t]
        sector_weights[s] = sector_weights.get(s, 0.0) + float(w)
    violations = {s: w for s, w in sector_weights.items() if w > sector_max}
    return violations

def recency_bias(
    selection_history: pd.DataFrame,
    recent_window: int = 30,
    threshold_ratio: float = 2.0,
) -> Dict[str, float]:
    if len(selection_history) < 2 * recent_window:
        return {}
    recent = selection_history.iloc[-recent_window:]
    past = selection_history.iloc[:-recent_window]
    recent_mean = recent.mean()
    past_mean = past.mean()
    ratio = (recent_mean / (past_mean + 1e-8)).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.dropna()
    biased = ratio[ratio > threshold_ratio].to_dict()
    return {str(k): float(v) for k, v in biased.items()}
