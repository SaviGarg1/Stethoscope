from typing import Dict
import pandas as pd


def prepare_input(features: Dict[str, float], numeric_cols: list, cat_cols: list):
    """Return a DataFrame with columns ordered and missing values filled.

    - numeric_cols: list of expected numeric column names
    - cat_cols: list of expected categorical column names
    """
    row = {}
    # fill numeric columns
    for c in numeric_cols:
        row[c] = features.get(c, 0.0)
    # fill categorical columns with a safe default
    for c in cat_cols:
        row[c] = features.get(c, 'Unknown')
    df = pd.DataFrame([row], columns=(numeric_cols + cat_cols))
    return df
