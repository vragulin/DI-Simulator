"""
    Direct index / tax harvesting simulation.  One path simulation using market data from pickle files
    by V. Ragulin, started 15-Aug-2022
"""
import datetime as dt
import logging
import os
import time
import warnings
from contextlib import suppress

import numpy as np
import pandas as pd

from load_mkt_data import load_mkt_data, vectorize_dict
from pretty_print import df_to_format
from sim_one_path import run_sim


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
    mdata = load_mkt_data(data_files, params['dt'], rand_w=False, range_info=range_info)

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


# ***********************************************************
# Entry point
# ***********************************************************
if __name__ == "__main__":
    timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = 'results/sim_' + timestamp + '.log'

    # Silently remove prior log_files and set up logger
    with suppress(OSError):
        os.remove(log_file)

    logging.basicConfig(level=logging.INFO, filename=log_file,
                        format='%(message)s')
    # Run simulation
    tic = time.perf_counter()  # Timer start

    # Load simulation parameters
    input_file = 'inputs/config_test10_opt.xlsx'
    params = load_params(input_file)

    # Load simulation data files
    dir_path = 'data/test_data_10'
    data_files = {'px': os.path.join(dir_path, 'prices.pickle'),
                  'tri': os.path.join(dir_path, 't_rets.pickle'),
                  'w': os.path.join(dir_path, 'daily_w.pickle')}

    # dir_path = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp_clean/"
    # data_files = {'px': os.path.join(dir_path, 'idx_prices.pickle'),
    #               'tri': os.path.join(dir_path, 'idx_t_rets.pickle'),
    #               'w': os.path.join(dir_path, 'idx_daily_w.pickle')}

    data_dict = load_data(data_files, params)

    sim_stats, step_report = run_sim(data_dict, suffix=timestamp)

    toc = time.perf_counter()  # Timer start
    print(f'Simulation took {(toc - tic):0.4f} sec.\n')

    print("Trade Summary (x100):")
    print(df_to_format(step_report * 100, formats={'_dflt': '{:.2f}'}))

    print("\nSimulation statistics:")
    print(df_to_format(pd.DataFrame(sim_stats * 100, columns=["annualized (%)"]), formats={'_dflt': '{:.6f}'}))

    print("\nDone")
