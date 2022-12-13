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
from load_mkt_data import vectorize_dict #  , crop_dframe
from pretty_print import df_to_format
from sim_account import run_sim_w_cash
from port_lots_class import DispMethod
from sim_one_path_mdata import load_params, load_data

warnings.filterwarnings("ignore")


def load_params_n_range(file: str) -> dict:
    """
    Wrapper to add simulation range info to the params
    :param file: file path where parameter xls or csv file reside
    :return: dictionary with parameter values
    """

    out_dict = load_params(file)
    if config.CROP_RANGE:
        range_info = {'t_start': config.t_start,
                      't_end': config.t_end}
        out_dict['range_info'] = range_info

    return out_dict


def load_ac_data(data_files: dict, params: dict, fixed_alloc: Optional[float] = None,
                 use_bmk: bool = True) -> dict:
    """ Wrapper to load both index data and asset allocation
        To run without a benchmark specify fixed_alloc = -1 (or another value not on the list)
    """
    out_dict = load_data(data_files, params)

    # Load equity weights
    range_info = params.get('range_info')

    if fixed_alloc is None:
        df_alloc = pd.read_pickle(data_files['eq_alloc'])
    else:
        df_alloc = pd.DataFrame(fixed_alloc,
                                index=out_dict['dates'][out_dict['dates_idx']],
                                columns=['eq_alloc'])

    if range_info is not None:
        df_alloc = df_alloc.reindex(out_dict['dates'][out_dict['dates_idx']],
                         method='nearest')

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

    # Check if we have a benchmark for this fixed allocation - we will need that for tracking
    out_dict['bmk_val'] = None

    bmk_dir = Path('../data/benchmarks') / config.sim_code
    bmk_code = data_files['bmk_code']

    if use_bmk:
        if fixed_alloc is None:
            bmk_file = bmk_dir / f'elm_{bmk_code}_passive_tsst.csv'
        elif fixed_alloc == 0.75:
            bmk_file = bmk_dir / f'fix75_{bmk_code}_passive_tsst.csv'
        elif fixed_alloc == 1.0:
            bmk_file = bmk_dir / f'fix100_{bmk_code}_passive_tsst.csv'
        else:
            raise FileNotFoundError(f"Benchmark file not found")

        if bmk_file.exists():
            df_bmk = pd.read_csv(bmk_file, index_col='step')
            out_dict['bmk_val'] = df_bmk[['port_val']]

            # Log benchmark used
            for func in [print, logging.info]:
                func(f"Benchmark file: {bmk_file}")

        else:
            print(f"Warning: benchmark file not found: {bmk_file}")

    # Add numpy array versions of the series to the array
    vect_fields = ['eq_alloc', 'int_rate', 'cash_ret', 'cash_tri', 'bmk_val']
    vectorize_dict(out_dict, vect_fields)

    return out_dict


def log_run_params():
    """ Print run paramters both to the console and to the log file
    """
    for func in [print, logging.info]:
        # Log run parameters
        func("Run Parameters:")
        func(params)
        func('\nData Files location:')
        func(data_files)
        func('\nTax assumptions:')
        func(config.tax)
        func('\nDisposition method:')
        func(data_dict['disp_method'].name)
        func('\nStart Simulation:')


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
    input_file = '../inputs/config_ac_mkt_20y.xlsx'
    params = load_params_n_range(input_file)

    # Load simulation data files
    dir_path = Path(config.WORKING_DIR)
    data_files = {'px': dir_path / config.PX_PICKLE,
                  'tri': dir_path / config.TR_PICKLE,
                  'w': dir_path / config.W_PICKLE,
                  'eq_alloc': dir_path / config.EQ_ALLOC_PICKLE,
                  'bmk_code': 'ret7'}

    # data_dict = load_ac_data(data_files, params, fixed_alloc=1.0)
    data_dict = load_ac_data(data_files, params)

    # Set lot disposition method
    data_dict['disp_method'] = DispMethod.LTFO

    # Start the simulation
    log_run_params()
    sim_stats, step_report = run_sim_w_cash(data_dict, suffix=timestamp)

    # Print results: step report + simulation statistics
    print("\nTrade Summary (x100, %):")
    # print(f"Simulation time = {toc-tic} seconds.")
    print(df_to_format(step_report * 100, formats={'_dflt': '{:.2f}'}))

    print("\nSimulation statistics (%):")
    print(df_to_format(pd.DataFrame(sim_stats * 100, columns=["annualized (%)"]), formats={'_dflt': '{:.2f}'}))
    print("\nDone")
