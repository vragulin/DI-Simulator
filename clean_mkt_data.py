""" Generate a set of clean pickle files that don't need to be filtered
    V. Ragulin, 9-Sep-2022
"""

import os
import numpy as np
import pandas as pd
from typing import Optional
import index_math as im
import config as cnf

# Data location for the original pickle files
source_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
result_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp_clean/"
PX_PICKLE = "idx_prices.pickle"
TR_PICKLE = "idx_t_rets.pickle"
W_PICKLE = "idx_daily_w.pickle"


def clean_mkt_data(div_yld, max_px: float = 100.0, min_px: float = 0.001) -> None:
    """ Load data from pickle files and clean it"""

    px = pd.read_pickle(os.path.join(source_dir, PX_PICKLE)).fillna(method='ffill').fillna(0)
    # tri = pd.read_pickle(os.path.join(source_dir, TR_PICKLE)).fillna(method='ffill').fillna(0)
    # w = pd.read_pickle(os.path.join(source_dir, W_PICKLE)).fillna(0)

    # find stocks present for the entire sample and (for fixed share indices) ones that have not risen too much and
    # pull out everything else
    full_indic = (
            (px > px.iloc[0, :] * min_px) &
            (px < px.iloc[0, :] * max_px)
    ).all().values

    # Generate clean series
    px_clean = px.iloc[:, full_indic]
    px_arr = px_clean.values
    px_ret = px_arr[1:, :] / px_arr[:-1, :]

    # Re-generate weights assuming we buy index at t=0 and don't rebalance
    np.random.seed(7)
    w_start = np.exp(np.random.normal(size=(1, px_clean.shape[1])))
    w_start /= np.sum(w_start)

    # Generate total returns assuming a fixed dividend yield override
    t_rets = px_ret + div_yld / 252

    tri_res = np.ones(px_clean.shape)
    tri_res[1:, :] = t_rets.cumprod(axis=0)
    tri_res *= px_arr[0, :]

    w_idx = im.index_weights_over_time(w_start, px_arr / px_arr[0, :])

    w_sum = w_idx.sum(axis=1)
    assert np.isclose(w_sum, 1.0).all()

    # Print checks
    print(f'Max weights = {np.max(w_idx):.3f}')

    # Pack data into datframes and output
    df_px = px_clean
    df_tri = pd.DataFrame(tri_res, index=px_clean.index, columns=px_clean.columns)
    df_w = pd.DataFrame(w_idx, index=px_clean.index, columns=px_clean.columns)

    # Save the data into new pickles
    df_px.to_pickle(os.path.join(result_dir, PX_PICKLE))
    df_tri.to_pickle(os.path.join(result_dir, TR_PICKLE))
    df_w.to_pickle(os.path.join(result_dir, W_PICKLE))

    return None


# Entry point
if __name__ == "__main__":
    div_override = 0.02
    clean_mkt_data(div_override)
    print('Done')
