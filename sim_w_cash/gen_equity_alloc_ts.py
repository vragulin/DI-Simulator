""" Generate plausible equity allocation weights from price and weight data
    To be used in account simuations

    V. Ragulin, Started 8-Nov-22
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config
from typing import Sequence
from sim_one_path_mdata import load_params, load_data


def moving_average(x: Sequence, w: int) -> np.array:
    """ Compute a rolling moving average without a gap in the front
        :param x: data (1-d array or list or other sequence)
        :param w: window length
    """
    conv = np.convolve(x[:, 0], np.ones(w), 'full') / w
    return conv[:len(x)]


def norm_ma(x: Sequence, w: int, adj_start_val: bool = False) -> np.array:
    """ Compute a rolling moving average without a gap in the front
        Normalize std of the output to 1
        :param x: data (1-d array or list or other sequence)
        :param adj_start_val: if True, taper the first w points so that the result starts at zero value
        :param w: window length
    """
    ma = moving_average(x, w)
    norm_shift = np.mean(ma)

    if adj_start_val:
        scale = np.minimum((np.ones(len(x)).cumsum() - 1) / w, 1)
        norm_shift *= scale

    return (ma - norm_shift) / np.std(ma)


def gen_equity_weights(data_dict: dict, alloc_params: dict) -> pd.DataFrame:
    """ Generate equity allocation weight depending on the index
        :param data_dict: context dict with stock prices, dates, weights etc..
        :param alloc_params: dict with strategy parameters
        :result: df index by dates with equity allocation percentage
    """

    # Generate index series
    px_arr = data_dict['px_arr']
    w = data_dict['w_arr'][0, None]
    # w[0, :2] = 0.3
    # w /= np.sum(w)
    idx_vals = px_arr @ w.T
    idx_rets = idx_vals * 0.0
    idx_rets[1:] = np.log(idx_vals[1:] / idx_vals[:-1])

    # Calc Value and Momentum signals
    dt = data_dict['params']['dt']

    # Value = normalized deviation from 5y return
    val_win = int(5 * config.ANN_FACTOR / dt)
    sig_val = -norm_ma(idx_rets, val_win)

    # Momentum = 12m return
    if config.MOMENTUM_SIG_TYPE == 'TRAIL_EX_1':
        mom_win = int(1 * config.ANN_FACTOR / dt) -1
        sig_mom0 = norm_ma(idx_rets, mom_win)
        sig_mom = sig_mom0 * 0
        sig_mom[1:] = sig_mom0[:-1]
    else:  # Use trailing 1y return
        mom_win = int(1 * config.ANN_FACTOR / dt)
        sig_mom = norm_ma(idx_rets, mom_win)

    # Combined signal
    sig = sig_val * alloc_params['w_val'] + sig_mom * alloc_params['w_mom']
    sig /= np.std(sig)

    # Map signal into asset allocation.
    band = 1.5  # std band that corresponds to the range
    eq_alloc = sig / band * alloc_params['bandwidth']

    eq_alloc = np.maximum(eq_alloc, -alloc_params['bandwidth'])
    eq_alloc = np.minimum(eq_alloc, alloc_params['bandwidth'])
    eq_alloc += alloc_params['base_weight']

    # Combine the weights and dates to form a dataframe
    df_out = pd.DataFrame(idx_vals,
                          index=data_dict['dates'][data_dict['dates_idx']],
                          columns=['idx'])
    df_out['eq_alloc'] = eq_alloc
    df_out['sig_val'] = sig_val
    df_out['sig_mom'] = sig_mom

    return df_out


if __name__ == "__main__":

    SAVE_FILE = True

    # Load simulation parameters
    input_file = '../inputs/config_ac_mkt_data.xlsx'

    params = load_params(input_file)

    # Weight Parmeters
    alloc_params = {
        'base_weight': 0.75,
        'bandwidth': 0.25,
        'w_val': 1,
        'w_mom': 1
    }

    # Load simulation data files
    dir_path = Path(config.WORKING_DIR)
    data_files = {'px': dir_path / config.PX_PICKLE,
                  'tri': dir_path / config.TR_PICKLE,
                  'w': dir_path / config.W_PICKLE,
                  'eq_alloc': dir_path / config.EQ_ALLOC_PICKLE}

    data_dict = load_data(data_files, params)

    # Generate weights
    df_eq = gen_equity_weights(data_dict, alloc_params)
    print(df_eq)

    # Save df to file
    if SAVE_FILE:
        # noinspection PyTypeChecker
        df_eq.to_pickle(data_files['eq_alloc'])

    df_eq[['idx','sig_val','sig_mom']].plot(title='Index Values and Signals', legend=True)
    plt.show()

    df_eq['eq_alloc'].plot(title='Equity Allocations', legend=True)
    plt.show()

    # Produce a picture of the index and
    print("Done")
