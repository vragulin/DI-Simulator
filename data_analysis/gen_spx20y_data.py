""" Extract from data files SPX stocks with continous 20y history and
    build a new set of pickle files with this data
"""
import pathlib
import numpy as np
import pandas as pd
from pathlib import Path

SOURCE_DIR = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
PX_PICKLE = "idx_prices.pickle"
TR_PICKLE = "idx_t_rets.pickle"
W_PICKLE = "idx_daily_w.pickle"

DEST_DIR = r"../data/mkt_data_spx_20y"

START_DATE = "2002-10-01"


def load_data() -> dict:

    dir_path = Path(SOURCE_DIR)

    px = pd.read_pickle(dir_path / PX_PICKLE)
    tri = pd.read_pickle(dir_path / TR_PICKLE)
    w = pd.read_pickle(dir_path / W_PICKLE)

    inputs = {'px': px, 'tri': tri, 'w': w}
    return inputs


def load_good_tickers():
    fpath = Path('../data/mkt_data_spx_20y/SPX_20y.xlsx')
    df = pd.read_excel(fpath)
    tickers = df['Symbol'].values
    return sorted(tickers)

def filter_data(inputs: dict, idx_tickers: list, save_files: bool = False):
    px, tri, w = inputs['px'], inputs['tri'], inputs['w']

    input_tickers = set(px.columns)
    good_tickers = input_tickers.intersection(set(idx_tickers))
    ticker_list0 = sorted(list(good_tickers))
    tkr0_arr = np.array(ticker_list0)

    #Run another check that the prices are present throughout the entire period
    px1 = px.loc[START_DATE:, ticker_list0]
    full_indic = np.sum((px1.to_numpy() > 0), axis=0) == len(px1.index)
    tkr_arr = tkr0_arr[full_indic]
    ticker_list = list(tkr_arr)

    # Make sure that new weights add up to 1
    w1 = w.loc[START_DATE:, ticker_list]
    w_rescaled = w1.divide(w1.sum(axis=1), axis=0)

    out_dict = {'px': px.loc[START_DATE:, ticker_list],
                'tri': tri.loc[START_DATE:, ticker_list],
                'w': w_rescaled}

    if save_files:
        dir_path = pathlib.Path(DEST_DIR)
        for k, f in zip(out_dict, [PX_PICKLE, TR_PICKLE, W_PICKLE]):
            out_dict[k].to_pickle(dir_path / f)

    return out_dict


if __name__ == "__main__":
    inputs = load_data()

    spx_tickers = load_good_tickers()
    outputs = filter_data(inputs, spx_tickers, save_files=True)

    print('Done')
