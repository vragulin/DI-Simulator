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
from typing import Optional
from contextlib import suppress
from sim_one_path_mdata import load_params, load_data
from sim_multi_path_mdata import gen_data_dict_emba
from port_lots_class import DispMethod
from sim_account import run_sim_w_cash
from sim_account_mdata import load_params_n_range
from gen_equity_alloc_ts import norm_ma

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
        mom_win = int(1 * config.ANN_FACTOR / dt) -1
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

    # Add interest rate info

    # Get any other arrays needed for calculation

    return data_dict


# ***********************************************************
# Entry point
# ***********************************************************
if __name__ == "__main__":
    # Simulation parameters
    n_paths = 5
    input_file = '../inputs/config_ac_multipath.xlsx'
    disp_method = DispMethod.LTFO
    fixed_alloc = None # 1.0, 0.75
    gen_shuffles = True  # Load shuffle info, rather than generate it

    # Equity asset alloction parmeters
    alloc_params = {
        'base_weight': 0.75,
        'bandwidth': 0.25,
        'w_val': 1,
        'w_mom': 1
    }

    # Start Simulation
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = '../results/ac_w_cash/sim_' + timestamp + '.log'

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

    # Load simulation data files
    data_dir = Path(config.DATA_DIR)
    data_files = {'px': data_dir / config.PX_PICKLE,
                  'tri': data_dir / config.TR_PICKLE,
                  'w': data_dir / config.W_PICKLE}

    # Load base data (common for all paths)
    base_dict = load_data(data_files, params)

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

    logging.info('Inputs:\n' + str(params) + '\n')
    logging.info('\nConfig file: ' + input_file + '\n')
    logging.info('\nPath Files:\n' + str(path_files) + '\n')
    logging.info(f'#Paths: {n_paths}, #Steps: {n_steps}, #Stocks: {n_stocks}')

    for i in range(1, n_paths+1):
        # ToDo: write the path_data_from_files function
        data_dict = gen_path_n_alloc_data(i, path_info, params, log_stats = True)
        print(f"Running sim for path {i}")
        # sim_stats, step_report  = run_sim_w_cash(data_dict, suffix=timestamp)

    print("Done")
