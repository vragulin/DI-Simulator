"""
    Data loading functions adapted ot the B500 simulation
    by V. Ragulin, started 1-Feb-2023
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from load_mkt_data import vectorize_dict, read_crop_pickle


def load_B500_data(data_files: dict, data_freq: int, filter_params: Optional[dict] = None,
                   range_info: Optional[dict] = None) -> dict:
    """
    load market data from files and do some simple processing
    :param data_files: dictionary with paths to pickle files with data
    :param data_freq:  number of trading days between rebalance points
    :param filter_params: dictionary with parameters controlling how we filter out bad data
                        if None - no filtering, otherwise need fields 'max' and 'min'.
    :param range_info: specify date range we want to load
    :return: dict of arrays with prices, returns, weights to be used for analysis
             no re-scaling or randomization in this function - this will be done in a separate
             function
    """

    # Load data from files
    # px = pd.read_pickle(data_files['px']).fillna(method='ffill').fillna(0)
    px = read_crop_pickle(data_files['px'], range_info=range_info).fillna(method='ffill').fillna(0)

    # Check that stocks are alphabetically sorted, otherwise there may be issues later in the code.
    # TODO - in later versions teach the code to handle unsorted stock
    assert sorted(px.columns.to_list()) == px.columns.to_list(), "Error: stocks in data files not sorted."

    dates = px.index
    dates_idx = range(0, len(px.index), data_freq)
    px.index = range(0, len(px.index))
    px = px.reindex(index=dates_idx)

    # tri = pd.read_pickle(data_files['tri']).fillna(method='ffill').fillna(0)
    tri = read_crop_pickle(data_files['tri'], range_info=range_info).fillna(method='ffill').fillna(0)
    tri.index = range(0, len(tri.index))
    tri = tri.reindex(index=dates_idx)

    # Load weights
    w = read_crop_pickle(data_files['w'], range_info=range_info).fillna(0)
    w.index = range(0, len(w.index))
    w = np.maximum(0, w.reindex(index=dates_idx))

    # Build returns
    d_px = px.pct_change().fillna(0).replace(np.inf, 0)
    d_tri = tri.pct_change().fillna(0).replace(np.inf, 0)

    # Check that we don't have negative dividends, this can happen if we have bad input data
    d_tri = np.maximum(d_tri, d_px)
    div = d_tri - d_px

    # find stocks present for the entire sample and ones that stayed within price bands
    # for fixed share indices large movers can cause problems, because their weights can get too high
    if filter_params is None:
        px_cap, px_floor = 1e10, 1e-10
    else:
        px_cap = filter_params.get('max', 1e10)
        px_floor = filter_params.get('min', 1e-10)

    full_indic: np.array = (
            (px > px.loc[0, :] * px_floor) &
            (px < px.loc[0, :] * px_cap)
    ).all().values

    d_px = d_px.iloc[:, full_indic].reset_index(drop=True)
    n_stocks = d_px.shape[1]
    n_steps = d_px.shape[0] - 1

    # Set random weights
    w = w.iloc[:, full_indic]

    # Rescale weighs to add up to 1
    w = w.div(w.sum(axis=1), axis='rows')

    # Calculate various return series
    px = (1 + d_px).cumprod()
    div = div.iloc[:, full_indic].reset_index(drop=True)
    d_tri = d_px + div
    tri = (1 + d_tri).cumprod()

    # Calculate weights - assume that the loaded dataframe contains actual end-of-period weights
    df_w = pd.DataFrame(w.values, index=px.index, columns=px.columns)

    # Pack data into an output dictionary
    out_dict = {'px': px, 'd_px': d_px, 'div': div, 'tri': tri, 'd_tri': d_tri,
                'w': df_w, 'fixed_weights': None,
                'dates': dates, 'dates_idx': dates_idx}

    return out_dict


def load_params(file: str) -> dict:
    """ load simulation settings """

    warnings.simplefilter(action='ignore', category=UserWarning)
    df_params = pd.read_excel(file, sheet_name='params', index_col='Name')
    params = df_params['Value'].to_dict()

    # Ensure correct data types
    if not params['ret_override_flag']:
        params['ret_override'] = None

    items_to_int = ['dt', 'n_steps', 'harvest_freq', 'donate_freq']
    for item in items_to_int:
        try:
            params[item] = int(params[item])
        except KeyError:
            params[item] = None

    # Initialize a random number generator instance
    params['rng'] = np.random.default_rng(2022)
    return params


def load_data(data_files: dict, params: dict) -> dict:
    """ Load simulation settings from an excel file
    :param data_files: dictiory with locations of pickle_files with historical market data
    :param params: dictionary with global simulation settings (i.e. not specific to mkt data)
    """

    range_info = params.get('range_info')
    # ToDo - need to write a separate data loader to account for the fact that there are zeros and NULLs in the arrays
    #        maybe after I populate the initial missing data with index returns, I will not need to write
    #        a new loaer function.

    mdata = load_B500_data(data_files, params['dt'], range_info=range_info)

    params['n_steps'] = n_steps = len(mdata['d_px']) - 1

    # Pack data into pandas dataframes
    tickers = mdata['d_px'].columns
    w = pd.DataFrame(mdata['w'], index=range(n_steps + 1), columns=tickers)
    div = mdata['div'].reset_index(drop=True)
    d_px = mdata['d_px'].reset_index(drop=True)
    px = mdata['px'].reset_index(drop=True)
    d_tri = mdata['d_tri'].reset_index(drop=True)

    # Pack into a dictionary
    data_dict = {'w': w, 'div': div, 'd_px': d_px, 'px': px,
                 'd_tri': d_tri, 'dates': mdata['dates'],
                 'dates_idx': mdata['dates_idx'],
                 'params': params,
                 'mkt_data': mdata, 'prices_from_pickle': True}

    # Vectorize
    vectorize_dict(data_dict, ['px', 'd_px', 'd_tri', 'div', 'w'])

    return data_dict
