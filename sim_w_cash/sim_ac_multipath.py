""" Run a multi-path simulation

    Started V. Ragulin 5-Dec-2022
"""

import datetime
import os
import logging
import warnings
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import index_math as im

from pathlib import Path
from codetiming import Timer
from typing import Optional
from contextlib import suppress
from load_mkt_data import vectorize_dict
from sim_one_path import pd_to_csv
from sim_one_path_mdata import load_data
from sim_multi_path_mdata import gen_data_dict_emba
from port_lots_class import DispMethod
from sim_account import run_sim_w_cash
from sim_account_mdata import load_params_n_range
from gen_equity_alloc_ts import norm_ma
from pretty_print import df_to_format

import config  # Use a separate config file for the multi-path simulation

warnings.filterwarnings("ignore")


def load_params_multipath(file: str, n_paths: int = 1, fixed_alloc: Optional[float] = None,
                          disp_method: DispMethod = DispMethod.LTFO) -> dict:
    """
    Wrapper to load single path parameters, plus initialize
    parameters specific to simulation
    :param file: file path where parameter xls or csv file reside
    :return: dictionary with parameter values
    """
    out_dict = load_params_n_range(file)
    out_dict['n_paths'] = n_paths
    out_dict['fixed_alloc'] = fixed_alloc
    out_dict['disp_method'] = disp_method

    return out_dict


def init_path_info(base: dict, params: dict, path_files: dict = {},
                   save: bool = False, load: bool = False) -> dict:
    """
    Generate paths or (if specified) load them from file
    :param base: base price data dictionary (market prices, weights etc.)
    :param params: simulation parameters dictionary
    :param path_files: dictionary of file locations
    :param save: flag if I want save the path info into files specified in the path_files
    :param laod: instead of generating shuffles and weights, load them from files
    :return: path_info dictionary with all info required fro the simulation
    """
    n_paths, n_steps = params['n_paths'], params['n_steps']

    # Create a shuffle array for returns
    shuffles = pd.DataFrame(1, index=range(n_steps), columns=range(1, n_paths + 1))

    if load:
        shuffles = pd.read_csv(path_files['shuffles'], index_col=0)
        weights = pd.read_csv(path_files['weights'], index_col=0)
    else:
        if params['randomize']:
            idx = shuffles[1].cumsum()
            for col in shuffles:
                shuffles[col] = idx.sample(n_steps, replace=params['replace'],
                                           random_state=params['rng'].bit_generator).values
        else:
            shuffle = shuffles.cumsum()

        # Create dataframe of starting weights
        weights = pd.DataFrame(0, index=range(1, n_paths + 1), columns=base['px'].columns)

        if params['random_weights']:
            weights.iloc[:, :] = np.exp(np.random.normal(size=weights.shape))
            weights = weights.div(weights.sum(axis=1), axis=0)
        else:
            weights.iloc[:, :] = base['w_arr'][0, :]

        # If requested, save files
        if save:
            shuffles.to_csv(path_files['shuffles'])
            weights.to_csv(path_files['weights'])

    if save:
        with open(path_files['base_dict'], 'wb') as f:
            pickle.dump(base, f)

    out_dict = {'base_dict': base,
                'shuffles': shuffles,
                'weights': weights}

    return out_dict


def gen_equity_alloc(data_dict: dict) -> pd.DataFrame:
    """ Generate equity allocation weight depending on the index
        :param data_dict: market data, params and other context
        :result: df index by dates with equity allocation percentage
    """

    # Generate index series
    prices = data_dict['px_arr']
    w0 = data_dict['w_arr'][0, :]
    idx_vals = im.index_vals(w0, prices, norm=True)
    idx_rets = idx_vals * 0.0
    idx_rets[1:] = np.log(idx_vals[1:] / idx_vals[:-1])

    # Calc Value and Momentum signals
    dt = params['dt']
    rebal_dates = data_dict['dates'][data_dict['dates_idx']]

    # Value = normalized deviation from 5y return
    val_win = int(5 * config.ANN_FACTOR / dt)
    sig_val = -norm_ma(idx_rets[:, None], val_win, adj_start_val=True)

    # Momentum = 12m return
    if config.MOMENTUM_SIG_TYPE == 'TRAIL_EX_1':
        mom_win = int(1 * config.ANN_FACTOR / dt) - 1
        sig_mom0 = norm_ma(idx_rets[:, None], mom_win)
        sig_mom = sig_mom0 * 0
        sig_mom[1:] = sig_mom0[:-1]
    else:  # Use trailing 1y return
        mom_win = int(1 * config.ANN_FACTOR / dt)
        sig_mom = norm_ma(idx_rets[:, None], mom_win)

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
    df_out = pd.DataFrame(idx_vals, index=rebal_dates, columns=['idx'])

    df_out['eq_alloc'] = eq_alloc
    df_out['sig_val'] = sig_val
    df_out['sig_mom'] = sig_mom

    return df_out


def gen_path_n_alloc_data(i_path: int, path_info: dict, params: dict,
                          log_stats: bool = False) -> dict:
    """
    Generate path data and simulated equity allocation from sim parameters and path info files.

    :param i_path: path number
    :param path_info: dictionary with file locations of where to find path data
    :param params: dictionary with other simualtion parameters
    :param log_stats: if true, print out (or log) moments of the resulting paths
    :return: data_dict
    """

    # Unpack input structure to convenience
    base = path_info['base_dict']
    shuffles = path_info['shuffles']
    weights = path_info['weights']
    dt = params['dt']
    shuffle_step = base['dates_idx'][1]

    # Check that path files frequency is the same as params['dt']
    assert dt == shuffle_step, f"Frequency does not match. Params: {dt}, Files: {shuffle_step}."

    # Calc shuffled market prices, divs, etc (if required)
    shuffle_idx = shuffles[i_path].values.astype(int)

    d_px_arr = base['d_px_arr'] * 0
    d_px_arr[1:, :] = base['d_px_arr'][shuffle_idx, :]

    d_px = pd.DataFrame(d_px_arr, index=base['d_px'].index,
                        columns=base['d_px'].columns).reset_index(drop=True)

    w0 = weights.loc[i_path].values

    # Generate the data_dict from the price changes and the starting weights
    data_dict = gen_data_dict_emba(d_px, w0, base, params)

    # Calc Elm-like equity weights for the full range, as an approximation assume fixed weights
    if params['fixed_alloc'] is None:
        df_alloc = gen_equity_alloc(data_dict)
    else:
        df_alloc = pd.DataFrame(params['fixed_alloc'],
                                index=base['dates'][base['dates_idx']],
                                columns=['eq_alloc'])

    data_dict['eq_alloc'] = df_alloc[['eq_alloc']]

    # Add interest rate info
    int_rate = config.int_rate
    dfr = pd.DataFrame(int_rate, index=df_alloc.index, columns=['int_rate'])

    dt = data_dict['params']['dt']
    dfr['cash_ret'] = dfr['int_rate'] * dt / config.ANN_FACTOR
    dfr['cash_tri'] = (1 + dfr['cash_ret'].shift().fillna(0)).cumprod()

    data_dict['int_rate'] = dfr[['int_rate']]
    data_dict['cash_ret'] = dfr[['cash_ret']]
    data_dict['cash_tri'] = dfr[['cash_tri']]

    # Check if we have a benchmark for this fixed allocation - we will need that for tracking
    # ToDo: think about my strategy for the benchmark
    # Assume that we have save a file with benchmark performance for every path in config.PATHS_DIR
    # Better strategy - load banchmark file during the set-up (i.e. before I start looping over paths)
    # And then just pull the be respective path out of the data frame.

    if params['use_benchmark']:
        data_dict['bmk_val'] = data_dict['bmk_paths'][[i_path]]
    else:
        data_dict['bmk_val'] = None

    # Add numpy array versions of the series to the array
    vect_fields = ['eq_alloc', 'int_rate', 'cash_ret', 'cash_tri', 'bmk_val']
    vectorize_dict(data_dict, vect_fields)

    return data_dict


def process_path_stats(path_stats: list) -> tuple:
    """ Compute average values of simualion results across a list of simulations
        Each simulation statistic is a series
        :param port_paths: list of dicts with portfolio path info
        :result: tuple of sim_stats, sim_summary, step_rpt
    """

    n_paths = len(path_stats)

    # Calculate statistics across the entire simulation period
    sim_stats = pd.DataFrame(0, index=path_stats[0][0].index, columns=range(1, n_paths + 1))
    for i, res in enumerate(path_stats):
        sim_stats[i + 1] = res[0]

    cols = ['Avg', 'Std', 'Max', 'Min']
    funcs = [np.mean, np.std, np.max, np.min]
    sim_summary = pd.DataFrame(0, index=sim_stats.index, columns=cols)

    for col, func in zip(cols, funcs):
        sim_summary[col] = func(sim_stats, axis=1)

    # Calculate statistics for each period (to track average dynamics of harvesting over time)
    step_rpt = pd.DataFrame(0, index=path_stats[0][1].index, columns=path_stats[0][1].columns)
    cols = ['harvest', 'donate', 'div', 'interest', 'port_val']

    # If benchmark has been given add its average performance to the table
    if not np.isnan(sim_summary.loc['tracking', 'Avg']):
        cols += ['bmk_val']

    for res in path_stats:
        one_path_rpt = res[1].fillna(0)
        step_rpt[cols] += one_path_rpt[cols] / n_paths

    return sim_stats, sim_summary, step_rpt


def load_benchmark_data(data_dict, data_files) -> Optional[Path]:
    """
    Add benchmark data to the data_dict structure in place
    :param data_dict: context market data + simulation params
    :param data_files:  directory of data files and benchmark extension info
    :returns: bmk_file full path or None if program failed
    """
    params = data_dict['params']
    fixed_alloc = params.get('fixed_alloc')

    data_dict['bmk_val'] = None
    bmk_dir = Path(config.PATHS_DIR)
    bmk_code = data_files['bmk_code']

    if fixed_alloc is None:
        bmk_file = bmk_dir / f'elm_{bmk_code}_passive_tsst.csv'
    elif fixed_alloc == 0.75:
        bmk_file = bmk_dir / f'fix75_{bmk_code}_passive_tsst.csv'
    elif fixed_alloc == 1.0:
        bmk_file = bmk_dir / f'fix100_{bmk_code}_passive_tsst.csv'
    else:
        raise FileNotFoundError(f"Benchmark file not found")

    if bmk_file.exists():
        df_bmk = pd.read_csv(bmk_file, index_col=0)
        data_dict['bmk_paths'] = df_bmk
        df_bmk.columns = [int(x) for x in df_bmk.columns]  # Set type of columns names to int

        # Log benchmark used
        for func in [print, logging.info]:
            func(f"Benchmark file: {bmk_file}")

    else:
        print(f"Warning: benchmark file not found: {bmk_file}")

    return df_bmk, bmk_file


# ***********************************************************
# Entry point
# ***********************************************************
if __name__ == "__main__":
    # Simulation parameters
    n_paths = 25
    input_file = '../inputs/config_ac_multipath.xlsx'
    disp_method = DispMethod.LTFO
    fixed_alloc = 1.0  # 1.0, 0.75
    gen_shuffles = True  # Load shuffle info, rather than generate it
    use_benchmark = True
    results_dir = '../results/multipath/'
    bmk_code = 'r7_v16_p25'

    # Equity asset alloction parmeters
    alloc_params = {
        'base_weight': 0.75,
        'bandwidth': 0.25,
        'w_val': 1,
        'w_mom': 1
    }

    # Start Simulation
    timer_full_run = Timer(name="Full Run", text="Full run time: {seconds:.1f} sec")
    timer_full_run.start()

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = results_dir + 'sim_' + timestamp + '.log'

    # Silently remove prior log_files and set up logger
    with suppress(OSError):
        os.remove(log_file)

    logging.basicConfig(level=logging.INFO, filename=log_file,
                        format='%(message)s')

    # Load simulation parameters
    params = load_params_multipath(input_file, n_paths=n_paths, fixed_alloc=fixed_alloc,
                                   disp_method=disp_method)
    params['alloc_params'] = alloc_params
    params['fixed_alloc'] = fixed_alloc
    params['use_benchmark'] = use_benchmark

    # Load simulation data files
    data_dir = Path(config.DATA_DIR)
    data_files = {'px': data_dir / config.PX_PICKLE,
                  'tri': data_dir / config.TR_PICKLE,
                  'w': data_dir / config.W_PICKLE,
                  'bmk_code': bmk_code}

    # Load base data (common for all paths)
    base_dict = load_data(data_files, params)
    base_dict['disp_method'] = disp_method

    if use_benchmark:
        bmk_paths, bmk_file = load_benchmark_data(base_dict, data_files)

    # Generate paths shuffles and starting weights
    paths_dir = Path(config.PATHS_DIR)
    dt = params['dt']
    path_files = {'base_dict': os.path.join(paths_dir, f'base_data_dict_{dt}.pickle'),
                  'shuffles': os.path.join(paths_dir, f'path_shuffles_{dt}.csv'),
                  'weights': os.path.join(paths_dir, f'path_weights_{dt}.csv')}

    if gen_shuffles:
        # Generate return shuffles (for paths) and random weights
        path_info = init_path_info(base=base_dict, params=params, path_files=path_files, save=True)
    else:
        # Load shuffles and weights from the saved files
        path_info = init_path_info(base=base_dict, params=params, path_files=path_files, load=True)

    # Run simulation - Loop over paths
    n_steps, n_stocks = path_info['shuffles'].shape[0], path_info['weights'].shape[1]
    path_stats = []
    port_paths = pd.DataFrame(np.nan, index=range(n_steps + 1), columns=path_info['shuffles'].columns)

    for func in [print, logging.info]:
        func('Inputs:\n' + str(params) + '\n')
        func('\nConfig file: ' + input_file + '\n')
        func('\nPath Files:\n' + str(path_files) + '\n')
        func(f'#Paths: {n_paths}, #Steps: {n_steps}, #Stocks: {n_stocks}\n')
        func(f'Disp Method: {disp_method}\n')
        if use_benchmark:
            func(f'Benchmark file: {bmk_file}\n')

    for i in range(1, n_paths + 1):
        # ToDo: write the path_data_from_files function
        print(f"Running sim for path {i}")
        data_dict = gen_path_n_alloc_data(i, path_info, params, log_stats=True)
        one_path_summary, one_path_steps = run_sim_w_cash(data_dict, suffix=timestamp,
                                                          save_files=False)

        path_stats.append((one_path_summary, one_path_steps))
        port_paths[i] = one_path_steps['port_val']

    # Process simulation results
    sim_stats, sim_summary, steps_report = process_path_stats(path_stats)

    # Print out results
    print("\nSimuation results by path (annualized, %):")
    print(df_to_format(sim_stats * 100, formats={'_dflt': '{:.2f}'}))
    pd_to_csv(sim_stats, 'sim_stats', suffix=timestamp, dir_path=results_dir)

    print("\nSimulation summary (annualized, %):")
    print(df_to_format(sim_summary * 100, formats={'_dflt': '{:.2f}'}))
    pd_to_csv(sim_summary, 'sim_summary', suffix=timestamp, dir_path=results_dir)

    print("\nResults per period (%):")
    pd_to_csv(steps_report, 'steps_report', suffix=timestamp, dir_path=results_dir)

    # steps_report.drop(columns='hvst_potl') - drop some columns so that that the report fits the screen
    pd.options.display.min_rows = 30
    print(df_to_format(steps_report * 100,
                       formats={'div': '{:.2f}', 'donate': '{:.2f}', 'interest': '{:.2f}',
                                'port_val': '{:.2f}', 'bmk_val': '{:.2f}',
                                '_dflt': '{:.4f}'}))

    pd_to_csv(port_paths, 'port_paths', suffix=timestamp, dir_path=results_dir)

    # ToDo: Investigate - why is harvest = nan in the log file?
    # ToDo: Add a run with benchmark functionality - maybe next week since I don't have a lot of time left
    # ToDo - add timer for the entire simulation (use codetimig librarary)

    timer_full_run.stop()
    print("Done")
