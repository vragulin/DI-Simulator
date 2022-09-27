"""
    Run multi-path harvesting simulation.
    Use multiple paths
    by V. Ragulin, started 12-Aug-2022
"""
import copy
import datetime as dt
import logging
import os
import time
import numpy as np
from contextlib import suppress
from typing import Optional
import pandas as pd
import pickle
from pretty_print import df_to_format
import sim_one_path as sop
import index_math as im
import sim_one_path_mdata as sopm
from sim_multi_path import process_path_stats
from gen_sim_paths import calc_path_stats

pd.options.display.float_format = '{:,.4f}'.format


def gen_data_dict_emba(d_px: pd.DataFrame, w0: np.array,
                       base_dict: dict, params: dict) -> dict:
    """ Implement James' logic to implement his paths for now
        Later I can replace this with another path generator

    :param d_px: dataframe of price moves (n_steps+1, n_stocks).  First row = zeros
    :param w0: series of initial weights
    :param base_dict: base data dictionary, which we will modify.
                            the program does not modify it
    :param params: dictionary of simulation parameters
    :returns: data_dict with final arrays
    """

    vol_override = params['vol_override']
    return_override = params['ret_override']
    dt = params['dt']
    out_dict = copy.deepcopy(base_dict)
    n_steps, n_stocks = d_px.shape

    # Get returns approximately equal to override
    if vol_override > 0:
        vol = np.sqrt(252 / dt) * np.var(d_px.values @ w0.T) ** 0.5
        d_px *= vol_override / vol
        d_px = np.maximum(d_px, -1 + np.finfo(float).eps)  # Ensure no neg prices

    if return_override > 0:
        rand_return = (252 / dt) * (np.sum(w0 * np.product(1 + d_px.values[1:, :], axis=0)
                                           ) ** (1 / n_steps) - 1)
        d_px.iloc[1:, :] += (return_override - rand_return) * (dt / 252)
        d_px.iloc[0, :] = 0
        d_px = np.maximum(d_px, -1 + np.finfo(float).eps)  # Ensure no neg prices

    # Update the remaining data series
    px = (1 + d_px).cumprod()
    idx_vals = px.values @ w0.T

    idx_rets = idx_vals[1:] / idx_vals[:-1] - 1
    idx_return = (252 / dt) * (idx_vals[-1] ** (1 / n_steps) - 1)

    idx_vol = np.sqrt(252 / dt) * np.std(idx_rets)
    idx_vol_fix_w = np.sqrt(252 / dt) * np.var(d_px.to_numpy() @ w0.transpose()) ** 0.5

    d_tri = d_px + out_dict['div'].reset_index()
    tri = (1 + d_tri).cumprod()

    # Calculate weights over time
    w = im.index_weights_over_time(w0, px.values)

    # Pack results into the output dictionary
    out_dict['w'] = w
    out_dict['d_px'] = d_px
    out_dict['px'] = px
    out_dict['d_tri'] = d_tri
    out_dict['tri'] = tri

    # Vectorize
    sopm.vectorize_dict(out_dict, ['px', 'd_px', 'd_tri', 'div', 'w'])

    return out_dict


def init_path_info(files: dict) -> dict:
    """
    Read files needed to generate paths
    :param files: dictionary with file paths
    :returns: dict
    """

    with open(files['base_dict'], "rb") as handle:
        base_data_dict = pickle.load(handle)

    shuffles = pd.read_csv(files['shuffles'], index_col=0)
    weights = pd.read_csv(files['weights'], index_col=0)

    out_dict = {'base_dict': base_data_dict,
                'shuffles': shuffles,
                'weights': weights}

    return out_dict


def path_data_from_files(i_path: int, path_info: dict, params: dict,
                         log_stats: bool = False) -> dict:
    """ Generate path data from sim parameters and
        path info files.  Replicate James' logic as much as possible
        since I am trying to match his run (maybe later I move his adjustment
        formulas into a functionand make it modular)

    :param i_path: path number
    :param path_info: dictionary with file locations of where to find path data
    :param params: dictionary with other simualtion parameters
    :param log_stats: if true, print out (or log) moments of the resulting paths
    :return: data_dict
    """

    base = path_info['base_dict']
    shuffles = path_info['shuffles']
    weights = path_info['weights']
    dt = params['dt']

    # Get the frequency of the path files
    try:
        shuffle_step = base['shuffle_idx_step']
    except KeyError:
        shuffle_step = params['dt']

        # Check that path files frequency is the same as params['dt']
        assert dt == base['px'].index[1], \
            f"Frequency does not match. Params: {dt}, Files: {base['px'].index[1]}."

    # Generate weights and price moves from the data files
    shuffle_idx = (shuffles.values[i_path, :] / shuffle_step).astype(int)
    d_px_arr = base['d_px'].values[shuffle_idx, :]
    d_px = pd.DataFrame(d_px_arr, index=base['d_px'].index,
                        columns=base['d_px'].columns).reset_index(drop=True)

    w0 = weights.iloc[i_path].values

    # Generate the data_dict from the price changes and the starting weights
    data_dict = gen_data_dict_emba(d_px, w0, base, params)

    # Set other parameters needed to downstream logic
    data_dict['params'] = params
    data_dict['prices_from_pickle'] = True

    if log_stats:
        idx_ret, idx_vol, stk_vol = calc_path_stats(data_dict, weights=w0, verbose=True)

    return data_dict


# ***********************************************************
# Entry point
# ***********************************************************
if __name__ == "__main__":

    # Set up logging
    timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = 'results/sim_' + timestamp + '.log'

    # Silently remove prior log_files and set up logger
    with suppress(OSError):
        os.remove(log_file)

    logging.basicConfig(level=logging.ERROR, filename=log_file,
                        format='%(message)s')

    # Load input parameters and set up accumulator structures
    input_file = 'inputs/config_mkt_data_test.xlsx'
    params = sopm.load_params(input_file)

    # Files with info needed to generate paths
    # path_data_dir = 'data/paths'
    # p_files = {'base_dict': f"{path_data_dir}/base_data_dict_replace_60.pickle",
    #            'shuffles': f"{path_data_dir}/path_shuffles_replace_60.csv",
    #            'weights': f"{path_data_dir}/path_weights_replace_60.csv"}

    path_dir = 'data/paths'
    dt = 20
    p_files = {'base_dict': os.path.join(path_dir, f'base_data_dict_{dt}.pickle'),
               'shuffles': os.path.join(path_dir, f'path_shuffles_{dt}.csv'),
               'weights': os.path.join(path_dir, f'path_weights_{dt}.csv')}
    # 'base_dict': os.path.join(path_dir, 'base_dict.pickle'),
    # 'shuffles': os.path.join(path_dir, 'shuffle.pickle'),
    # 'weights': os.path.join(path_dir, 'weights.pickle')}

    path_info = init_path_info(p_files)

    n_paths, n_stocks = path_info['weights'].shape
    # n_steps = path_info['shuffles'][1] - 1
    path_stats = []

    # Run simulation
    # Save file parameters into the log files
    logging.error('Inputs:\n' + str(params) + '\n')
    logging.error('\nConfig file: ' + input_file + '\n')
    logging.error('\nPath Files:\n' + str(p_files) + '\n')
    logging.error(f'#Paths: {n_paths}, #Stocks: {n_stocks}')


    tic = time.perf_counter()  # Timer start

    for i in range(n_paths):
        print(f"Path #{i + 1}")
        logging.info(f"\nPath #{i + 1}:")

        data_dict = path_data_from_files(i, path_info, params, log_stats=True)

        one_path_summary, one_path_steps = sop.run_sim(data_dict, suffix=timestamp)
        path_stats.append((one_path_summary, one_path_steps))

    toc = time.perf_counter()  # Timer end

    # Process simulation results
    t_total = toc - tic
    t_path = t_total / n_paths
    print(f'Simulation took {t_total:0.4f} sec., {t_path:0.4f} sec. per path.\n')

    sim_summary, sim_stats, steps_report = process_path_stats(path_stats)

    print("\nSimulation results by path (annualized, %):")
    print(df_to_format(sim_stats * 100, formats={'_dflt': '{:.6f}'}))
    sop.pd_to_csv(sim_stats, 'sim_stats', suffix=timestamp)

    print("\nSimulation summary (annualized, %):")
    print(df_to_format(sim_summary * 100, formats={'_dflt': '{:.6f}'}))
    sop.pd_to_csv(sim_summary, 'sim_summary', suffix=timestamp)

    print("\nResults per period (%):")
    sop.pd_to_csv(steps_report, 'steps_report', suffix=timestamp)

    # steps_report.drop(columns='hvst_potl') - drop some columns so that that the report fits the screen
    pd.options.display.min_rows = 20
    # steps_to_print = df_to_format(steps_report * 100,
    #                               formats={'_dflt': '{:.2f}'},
    #                               multipliers={'port_val': 0.01, 'bmk_val': 0.01}
    #                               )
    # print(steps_to_print)

    print(df_to_format(steps_report * 100,
                       formats={'_dflt': '{:.6f}'},
                       multipliers={'port_val': 0.01, 'bmk_val': 0.01}
                       ))

    print("\nDone")
