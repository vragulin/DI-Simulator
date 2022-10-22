""" Extract from data files SPX stocks with continous 20y history and
    build a new set of pickle files with this data
"""
import pathlib
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import *
from pathlib import Path
from typing import Optional

SOURCE_DIR = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp_clean/"
PX_PICKLE = "idx_prices.pickle"
TR_PICKLE = "idx_t_rets.pickle"
W_PICKLE = "idx_daily_w.pickle"
DEST_DIR = "../data/{subset_code}"

def load_data() -> dict:
    dir_path = Path(SOURCE_DIR)

    px = pd.read_pickle(dir_path / PX_PICKLE)
    tri = pd.read_pickle(dir_path / TR_PICKLE)
    w = pd.read_pickle(dir_path / W_PICKLE)

    inputs = {'px': px, 'tri': tri, 'w': w}
    return inputs


def filter_data(inputs: dict, idx_tickers: Optional[list], start_date: Optional[dt.date],
                save_files: bool = False, subset_code: Optional[str] = None):

    """ Filter simulation subset form the full data set"""

    px, tri, w = inputs['px'], inputs['tri'], inputs['w']

    input_tickers = set(px.columns)

    if idx_tickers is not None:
        good_tickers = input_tickers.intersection(set(idx_tickers))
    else:
        good_tickers = input_tickers

    ticker_list0 = sorted(list(good_tickers))
    tkr0_arr = np.array(ticker_list0)

    # Run another check that the prices are present throughout the entire period
    px1 = px.loc[start_date:, ticker_list0]
    full_indic = np.sum((px1.to_numpy() > 0), axis=0) == len(px1.index)
    tkr_arr = tkr0_arr[full_indic]
    ticker_list = list(tkr_arr)

    # Make sure that new weights add up to 1
    w1 = w.loc[start_date:, ticker_list]
    w_rescaled = w1.divide(w1.sum(axis=1), axis=0)

    out_dict = {'px': px.loc[start_date:, ticker_list],
                'tri': tri.loc[start_date:, ticker_list],
                'w': w_rescaled}

    if save_files:
        dir_path = pathlib.Path(DEST_DIR.format(subset_code=subset_code))
        dir_path.mkdir(parents=True, exist_ok=True)

        for k, f in zip(out_dict, [PX_PICKLE, TR_PICKLE, W_PICKLE]):
            out_dict[k].to_pickle(dir_path / f)

    return out_dict


if __name__ == "__main__":

    n_years = 15
    end_date = dt.date(2022, 6, 1)
    start_date = end_date + relativedelta(years=-n_years)

    inputs = load_data()

    # Load a subset of tickers form file, if specified
    good_tickers = None

    subset_code = f'mkt_data_{n_years}y'

    outputs = filter_data(inputs, good_tickers, start_date=start_date,
                          save_files=True, subset_code=subset_code)

    print('Done')
