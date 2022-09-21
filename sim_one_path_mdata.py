"""
    Direct index / tax harvesting simulation.  One path simulation using market data from pickle files
    by V. Ragulin, started 15-Aug-2022
"""
import os
import datetime as dt
import logging
import sys
import time
from contextlib import suppress
from typing import Optional
import warnings
import dateutil.relativedelta as du

import config
from load_mkt_data import load_mkt_data
import numpy as np
import pandas as pd
from sim_one_path import run_sim
from pretty_print import df_to_format


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


def load_data(data_files, params: dict) -> dict:
    """ Load simulation settings from an excel file
    :param params: dictionary with global simulation settings (i.e. not specific to path
    :param randomize: if True, generate random paths
    """
    # Load market data
    fixed_weight_flag = (params['benchmark_type'] == 'fixed_weights')  # the other setting is fixed_shares
    rand_state = params.get('rand_state', 0)

    mdata = load_mkt_data(data_files, params['dt'])

    params['n_steps'] = n_steps = len(mdata['d_px']) - 1

    # Array of rebalance dates (so that we finish arond today)
    # Use simulated values for simplicity since we are randomizing anyway.
    # if params['dt'] in [20, 60]:
    #     month_gap = int(params['dt']/20)
    #     rebal_dates = sorted([config.t_last + du.relativedelta(months=-month_gap * x)
    #                           for x in range(n_steps + 1)])
    # elif params['dt'] == 5:
    #     rebal_dates = sorted([config.t_last + du.relativedelta(weeks=-x)
    #                           for x in range(n_steps + 1)])
    # else:
    #     raise ValueError(f"Rebalance freq dt={params['dt']} not implemented.")

    rebal_dates = [mdata['dates'][i].date() for i in mdata['dates_idx']]

    # Pack data into pandas dataframes
    tickers = mdata['d_px'].columns
    w_idx = pd.DataFrame(mdata['w'], index=range(n_steps + 1), columns=tickers)
    div = mdata['div'].reset_index(drop=True)
    d_px = mdata['d_px'].reset_index(drop=True)
    px = mdata['px'].reset_index(drop=True)
    d_tri = mdata['d_tri'].reset_index(drop=True)

    # Pack into a dictionary
    data_dict = {'w_idx': w_idx, 'div': div, 'd_px': d_px, 'px': px,
                 'd_tri': d_tri, 'dates': rebal_dates, 'params': params,
                 'randomize': params['randomize'],
                 'vol_mult': mdata['vol_mult'],
                 'rng': params['rng'], 'mkt_data': mdata,
                 'prices_from_pickle': True}

    # print("First 5 weights:", data_dict['w_idx'].values[0, :5])
    # print("First 5 returns:", data_dict['d_px'].values[1, :5])

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
    tic = time.perf_counter() # Timer start

    input_file = 'inputs/config_mkt_data.xlsx'
    params = load_params(input_file)
    data_dict = load_data(params, randomize=params['randomize'])

    sim_stats, step_report = run_sim(data_dict, suffix=timestamp)

    toc = time.perf_counter() # Timer start
    print(f'Simulation took {(toc-tic):0.4f} sec.\n')

    print("Trade Summary (x100):")
    print(df_to_format(step_report * 100, formats={'_dflt': '{:.2f}'}))

    print("\nSimulation statistics:")
    print(df_to_format(pd.DataFrame(sim_stats * 100, columns=["annualized (%)"]), formats={'_dflt': '{:.6f}'}))

    print("\nDone")

