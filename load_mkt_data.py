""" Elm's code (James) to load data from pickle files
    Adapted by Vlad, 15-Aug-2022
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Any
from enum import Enum
import index_math as im
import config as cnf
import datetime
from rescale_stock_paths import rescale_stocks_to_match_idx


def vectorize_dict(data_dict: dict, fields: list) -> None:
    """ Add fields *_arr that are numpy arrays for a data dict in place
    :params data_dict:  original data_dict
    :params fields:  list (or other iterable) of fields to vectorize
    """
    for k in fields:
        try:
            if isinstance(data_dict[k], pd.DataFrame):
                data_dict[k + '_arr'] = data_dict[k].to_numpy()
            else:
                data_dict[k + '_arr'] = data_dict[k]
        except KeyError:
            raise KeyError(f"Field {k} not in data_dict.")


def crop_dframe(df: pd.DataFrame, t_start: Optional[datetime.date] = None,
                t_end: Optional[datetime.date] = None) -> pd.DataFrame:
    """ Crop price or return dataframe if t_start / t_end have been specified
        :param df: original dataframe, index_col = dates
        :param t_start: stard date (closed interval)
        :param t_end: end date (closed interval)
        :return: dataframe with cropped data
    """

    if (t_start is None) and (t_end is None):
        return df
    else:

        d_start = d_end = None
        df1 = df.copy()

        # Convert to date to datetime
        if t_start is not None:
            d_start = datetime.datetime(t_start.year, t_start.month, t_start.day)
            df1 = df1[df1.index >= d_start]

        if t_end is not None:
            d_end = datetime.datetime(t_end.year, t_end.month, t_end.day)
            df1 = df1[df1.index <= d_end]

        return df1


def read_crop_pickle(file: Any, range_info: Optional[dict] = None):
    """ Wrapper around pd.read_pickle() that also crops the output dataframe
        plus fills empty rows
    """

    df0 = pd.read_pickle(file)
    if range_info is None:
        return df0
    else:
        t_start = range_info.get('t_start')
        t_end = range_info.get('t_end')
        df = crop_dframe(df0, t_start=t_start, t_end=t_end)
        return df


def process_mkt_data(input: dict, data_freq: int, randomize: bool = False,
                     replace: bool = False, random_w: bool = False,
                     return_override: Optional[float] = None,
                     vol_override: float = -1,
                     vol_fixed_w: bool = True,
                     stk_res_vol_factor: Optional[float] = None,
                     ann_factor: float = 1.0,
                     rand_seed: Optional[int] = 0,
                     rand_seed_w: Optional[int] = 0) -> dict:
    """ Update market data before using it in the simulation
        If needed randomize (using random_state as an input
        And/or rescale to match target index return, index vol
        And rescale residaual stock vol by a factor.  Do processing in place.

        :param input: dictionary with data to be processed (in place)
        :param data_freq: trading days between data points (aka dt)
        :param randomize: if True, reshuffle returns
        :param replace: if True and do reshuffle with replacement
        :param random_w: if True - generate new random weights
        :param return_override: if not None, set index return to the target return
                                (annual, with freq set by freq of data points)
        :param vol_override: if not negative, set vol of index paths to this value
        :param vol_fixed_w: if True, set set to target index with fixed weights at t=0
                                    otherwise, set to target the vol of the entire series
        :param stk_res_vol_factor: if not None, apply a multiplicative scalar to the
                                    residual volatilities, keep index vol the same
        :param ann_factor: annualization factor
        :param rand_seed: random seed used to reshuffle returns
        :param rand_seed_w: random seed used to generate new returns

        :return: dictionary with moments (mean, std, resid vol etc.) of the rescaled paths
    """

    pass


def load_mkt_data(data_files: dict, data_freq: int, filter_params: Optional[dict] = None,
                  fixed_weights: bool = False, rand_w: bool = True,
                  rand_seed: Optional[int] = None, range_info: Optional[dict] = None) -> dict:
    """
    load market data from files and do some simple processing
    :param data_files: dictionary with paths to pickle files with data
    :param data_freq:  number of trading days between rebalance points
    :param filter_params: dictionary with parameters controlling how we filter out bad data
                        if None - no filtering, otherwise need fields 'max' and 'min'.
    :param rand_w: if True generate random weights, else use weights form the pickle file
    :param rand_seed: random seed used to generate random weights (for replicatbility)
    :param fixed_weights: if True assume fixed index weights for the stocks, otherwise, assume equal shares
                            (and weights evolve in line with stock prices)
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

    # Load weights only if needed
    if not rand_w:
        # w = pd.read_pickle(data_files['w']).fillna(0)
        w = read_crop_pickle(data_files['w'], range_info=range_info).fillna(0)
        w.index = range(0, len(w.index))
        w = np.maximum(0, w.reindex(index=dates_idx))
    else:
        w = None

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
    if rand_w:
        if rand_seed is not None:
            np.random.seed(rand_seed)
        w_arr = np.exp(np.random.normal(size=(1, n_stocks)))
        w = pd.DataFrame(w_arr, index=[0], columns=d_px.columns)
    else:
        w = w.iloc[:, full_indic]

    # Rescale weighs to add up to 1
    w = w.div(w.sum(axis=1), axis='rows')

    # Calculate various return series
    px = (1 + d_px).cumprod()
    div = div.iloc[:, full_indic].reset_index(drop=True)
    d_tri = d_px + div
    tri = (1 + d_tri).cumprod()

    # Calculate weights
    if fixed_weights:
        weights = np.ones((n_steps + 1, 1)) @ w
    else:  # fixed shares
        weights = im.index_weights_over_time(w.values[0], px.values)

    df_w = pd.DataFrame(weights, index=px.index, columns=px.columns)

    # Pack data into an output dictionary
    out_dict = {'px': px, 'd_px': d_px, 'div': div, 'tri': tri, 'd_tri': d_tri,
                'w': df_w, 'fixed_weights': fixed_weights,
                'dates': dates, 'dates_idx': dates_idx}

    return out_dict


# Entry point
if __name__ == "__main__":
    # Set up inputs
    data_files = {
        'px': os.path.join(cnf.working_dir, cnf.PX_PICKLE),
        'tri': os.path.join(cnf.working_dir, cnf.TR_PICKLE),
        'w': os.path.join(cnf.working_dir, cnf.W_PICKLE)
    }

    dt = 60

    filter_params = {
        'max': 1 + cnf.max_rise,
        'min': 1 - cnf.max_drop
    }

    # Pull data
    data_dict = load_mkt_data(data_files, dt, filter_params=filter_params,
                              rand_w=False)

    stats = process_mkt_data(data_dict, dt)

    print('Done')
