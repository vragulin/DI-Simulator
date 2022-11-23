""" Simulate an account that has both index and cash positions.
    Input data comes from pickle files, i.e. similar to sim_one_path_mdata

    Started V. Ragulin - 8-Nov_2022
"""
import datetime
import os
import logging
import warnings
import pandas as pd

from pathlib import Path
from typing import Optional
from contextlib import suppress

import config
from load_mkt_data import vectorize_dict, load_mkt_data
from pretty_print import df_to_format
from sim_account import run_sim_w_cash
from port_lots_class import DispMethod
from sim_one_path_mdata import load_params, load_data

warnings.filterwarnings("ignore")


def load_ac_data(data_files: dict, params: dict, fixed_alloc: Optional[float] = None) -> dict:
    # Wrapper to load both index data and asset allocation
    out_dict = load_data(data_files, params)

    # Load equity weights
    if fixed_alloc is None:
        df_alloc = pd.read_pickle(data_files['eq_alloc'])
    else:
        df_alloc = pd.DataFrame(fixed_alloc,
                                index=out_dict['dates'][out_dict['dates_idx']],
                                columns=['eq_alloc'])

    out_dict['eq_alloc'] = df_alloc[['eq_alloc']]

    # Popular interest rate period and cumulative returns
    int_rate = config.int_rate
    dfr = pd.DataFrame(int_rate, index=df_alloc.index, columns=['int_rate'])

    dt = out_dict['params']['dt']
    dfr['cash_ret'] = dfr['int_rate'] * dt / config.ANN_FACTOR
    dfr['cash_tri'] = (1 + dfr['cash_ret'].shift().fillna(0)).cumprod()

    out_dict['int_rate'] = dfr[['int_rate']]
    out_dict['cash_ret'] = dfr[['cash_ret']]
    out_dict['cash_tri'] = dfr[['cash_tri']]

    # Add numpy array versions of the series to the array
    vectorize_dict(out_dict, ['eq_alloc', 'int_rate', 'cash_ret', 'cash_tri'])

    return out_dict


# ***********************************************************
# Entry point
# ***********************************************************
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = '../results/ac_w_cash/sim_' + timestamp + '.log'

    # Silently remove prior log_files and set up logger
    with suppress(OSError):
        os.remove(log_file)

    logging.basicConfig(level=logging.INFO, filename=log_file,
                        format='%(message)s')

    # Load simulation parameters
    input_file = '../inputs/config_ac_test10.xlsx'
    params = load_params(input_file)

    # Load simulation data files
    dir_path = Path(config.WORKING_DIR)
    data_files = {'px': dir_path / config.PX_PICKLE,
                  'tri': dir_path / config.TR_PICKLE,
                  'w': dir_path / config.W_PICKLE,
                  'eq_alloc': dir_path / config.EQ_ALLOC_PICKLE}

    # data_dict = load_ac_data(data_files, params, fixed_alloc=0.75)
    data_dict = load_ac_data(data_files, params)

    # Set lot disposition method
    data_dict['disp_method'] = DispMethod.LTFO
    print(f"Disp Method = {data_dict['disp_method'].name}")

    # Start the simulation
    sim_stats, step_report = run_sim_w_cash(data_dict, suffix=timestamp)

    # Print results: step report + simulation statistics
    print("\nTrade Summary (x100, %):")
    # print(f"Simulation time = {toc-tic} seconds.")
    print(df_to_format(step_report * 100, formats={'_dflt': '{:.2f}'}))

    print("\nSimulation statistics (%):")
    print(df_to_format(pd.DataFrame(sim_stats * 100, columns=["annualized (%)"]), formats={'_dflt': '{:.2f}'}))
    print("\nDone")
