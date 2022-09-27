""" Generate and save simulation paths
    as pickle files
"""
# Entry point
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

import index_math as im

pd.options.display.float_format = '{:,.4f}'.format

import sim_one_path_mdata as sopm


def calc_path_stats(data_dict: dict, shuffle: Optional[np.array] = None,
                    weights: Optional[np.array] = None,
                    verbose: bool = False) -> tuple:

    """ Generate statistics of a set of paths """
    params = data_dict['params']
    dt = params['dt']

    if shuffle is None:
        d_px = data_dict['d_px'].values
    else:
        d_px = data_dict['d_px'].values[shuffle, :]

    if weights is None:
        w0 = data_dict['w'].values[0]
    else:
        w0 = weights

    n_steps = d_px.shape[0] - 1
    n_stocks = d_px.shape[1]
    px1 = (1 + d_px).cumprod(axis=0)

    idx_vals = px1 @ w0
    idx_rets = np.zeros(n_steps + 1)
    idx_rets[1:] = idx_vals[1:] / idx_vals[:-1] - 1
    idx_return = (252 / dt) * (idx_vals[-1] ** (1 / n_steps) - 1)
    idx_vol = np.sqrt(252 / dt) * np.std(np.log(1 + idx_rets))

    stk_vol = im.stock_vol_avg(d_px, w0, dt/252)

    if verbose:
        print(f"# shares: {n_stocks}", end=", ")
        print(f"Index ret: {idx_return * 100:.2f} ", end=", ")
        print(f"Idx vol: {idx_vol * 100:.2f}", end=", ")
        print(f"Stk wavg vol: {stk_vol * 100:.2f}")

    return idx_return, idx_vol, stk_vol


# ***********************************************************
# Entry point
# ***********************************************************
if __name__ == "__main__":

    # Load input parameters and set up accumulator structures
    input_file = 'inputs/config_mkt_data_test.xlsx'
    params = sopm.load_params(input_file)

    # Files with input data
    # working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp_clean/"
    # data_files = {'px': os.path.join(working_dir, "idx_prices.pickle"),
    #               'tri': os.path.join(working_dir, "idx_t_rets.pickle"),
    #               'w': os.path.join(working_dir, "idx_daily_w.pickle")}
    data_dir = 'data/test_data_100'
    data_files = {'px': os.path.join(data_dir, 'prices.pickle'),
                  'tri': os.path.join(data_dir, 't_rets.pickle'),
                  'w': os.path.join(data_dir, 'daily_w.pickle')}

    # Files for the output path files
    out_dir = 'data/test_data_100/paths'
    path_files = {'base_dict': os.path.join(out_dir, 'base_dict.pickle'),
                  'shuffles': os.path.join(out_dir, 'shuffle.pickle'),
                  'weights': os.path.join(out_dir, 'weights.pickle')}

    n_paths = 3
    path_stats = pd.DataFrame(0, index=range(n_paths),
                              columns=['idx_ret', 'idx_vol', 'stk_vol'])

    # Generate base dictionary
    base_dict = sopm.load_data(data_files, params)

    # Remember that our shuffles are indexed by location, not by days
    base_dict['shuffle_idx_step'] = 1

    n_steps, n_stocks = base_dict['d_px'].shape + np.array((-1, 0))
    path_shuffles = np.zeros((n_paths, n_steps + 1), dtype=np.int_)
    path_weights = np.ones((n_paths, n_stocks)) * base_dict['w'].values[0]

    rng = np.random.default_rng(2022)
    # Generate and shuffles and weights for different paths
    for i in range(n_paths):

        # Generate shuffle and weights rows, update dataframes
        path_shuffles[i, 1:] = rng.choice(range(1, n_steps+1), n_steps,
                                          replace=params['replace'])

        # Calculate statistics of the resulting paths
        idx_ret, idx_vol, stk_vol = calc_path_stats(
                                        base_dict, shuffle=path_shuffles[i],
                                        weights= path_weights[i])

        print(f"path = {i}, idx_ret = {idx_ret * 100:.2f}, "
              f"idx_vol = {idx_vol * 100:.2f}, stk_vol = {stk_vol*100:.2f}")

        path_stats.iloc[i, :] = idx_ret, idx_vol, stk_vol

    # Save path into a file
    print("Saving files...")
    pd.DataFrame(path_shuffles, index=range(n_paths), columns=range(n_steps + 1)
                 ).to_csv(path_files['shuffles'])
    pd.DataFrame(path_weights, index=range(n_paths), columns=base_dict['px'].columns
                 ).to_csv(path_files['weights'])
    with open(path_files['base_dict'], 'wb') as handle:
        pickle.dump(base_dict, handle)

    print("\nDone")
