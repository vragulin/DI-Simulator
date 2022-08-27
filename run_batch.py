"""
    Run a batch of multi-path simuations.
    by V. Ragulin, started 23-Aug-2022
"""
import datetime as dt
import logging
import os
import time
from contextlib import suppress
from typing import Optional
import pandas as pd
from itertools import product

pd.options.display.float_format = '{:,.4f}'.format

from pretty_print import df_to_format
import sim_one_path as sop
import sim_one_path_mdata as sopm
import sim_multi_path_mdata as smpm


def init_batch() -> pd.DataFrame:
    """ Define parameters for runs in the batch.
    :return: dataframe with param settings for each run
    """

    columns = ['dt', 'donate', 'ret_override', 'vol_scaling']
    dt = [60, 20, 5]
    donate = [0, 0.05, 0.1]
    vol = [0.5, 1.0, 1.5]
    ret = [0, 0.02, 0.05, 0.1]

    batch_data = list(product(dt, donate, ret, vol))
    df_sim = pd.DataFrame(batch_data, columns=columns).astype({'dt': int})
    df_sim.index.name = 'run_id'
    return df_sim


def gen_sim_params(base_params: dict, overrides: pd.Series) -> dict:
    """  Apply overrides to generate parameter dictionary for a simulation
    :param base_params: dictionary with base parameter values
    :param overrides: series with overrides
    :return: updated dictionary
    """
    new_params = base_params.copy()
    new_params.update(overrides.to_dict())
    # Make sure 'dt' is of the correct dtype (int)
    new_params['dt'] = int(new_params['dt'])
    new_params['harvest_freq'] = new_params['dt']
    return new_params


def already_processed(sim_name: str, dir_path: str = '/results/batch') -> bool:
    """ Check if the results for a scenario have already been saved """

    for file_code in ['sim_stats', 'sim_summary', 'steps_report']:
        if sim_name is not None:
            fname = dir_path + file_code + '_' + sim_name + ".csv"
        else:
            fname = dir_path + file_code + ".csv"

        if not os.path.exists(fname):
            return False

    return True


def run_one_param_set(params: dict, n_paths: int = 1, name: Optional[str] = None,
                      overwrite: bool = True, dir_path: str = '/results/batch') -> int:
    """ Run one multi-path simulation """

    # Check if this scenario has already been processed
    if (not overwrite) and already_processed(sim_name, dir_path):
        print(f"Simulation {name} already processed.")
        return -1

    # If we have not saved data for this simualtion already, run it now
    data_dict = sopm.load_data(params, randomize=params['randomize'])

    logging.info('Inputs:\n' + str(params) + '\n')
    path_stats = []

    # Run simulation
    tic = time.perf_counter()  # Timer start
    for i in range(n_paths):
        print(f"Path #{i + 1}")
        logging.info(f"\nPath #{i + 1}:")
        one_path_summary, one_path_steps = sop.run_sim(data_dict, suffix=sim_name,
                                                       dir_path=dir_path, save_files=False)
        path_stats.append((one_path_summary, one_path_steps))

        if i < n_paths - 1:  # if it's not the last iteration - update data
            data_dict = sopm.load_data(params, data_dict=data_dict, randomize=params['randomize'])

    toc = time.perf_counter()  # Timer end

    # Process simulation results
    t_total = toc - tic
    t_path = t_total / n_paths
    print(f'Simulation took {t_total:0.4f} sec., {t_path:0.4f} sec. per path.\n')

    sim_summary, sim_stats, steps_report = smpm.process_path_stats(path_stats)

    print("\nSimulation results by path (annualized, %):")
    print(df_to_format(sim_stats * 100, formats={'_dflt': '{:.2f}'}))
    sop.pd_to_csv(sim_stats, 'sim_stats', suffix=sim_name, dir_path=dir_path)

    print("\nSimulation summary (annualized, %):")
    print(df_to_format(sim_summary * 100, formats={'_dflt': '{:.2f}'}))
    sop.pd_to_csv(sim_summary, 'sim_summary', suffix=sim_name, dir_path=dir_path)

    print("\nResults per period (%):")
    sop.pd_to_csv(steps_report, 'steps_report', suffix=sim_name, dir_path=dir_path)

    # steps_report.drop(columns='hvst_potl') - drop some columns so that that the report fits the screen
    pd.options.display.min_rows = 20
    print(df_to_format(steps_report * 100,
                       formats={'_dflt': '{:.2f}'},
                       multipliers={'port_val': 0.01, 'bmk_val': 0.01}
                       ))
    return 0


# ***********************************************************
# Entry point
# ***********************************************************
if __name__ == "__main__":
    # Set up logging
    # timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    batch_name = 'prod10_thresh0'
    dir_path = 'results/batch/'
    log_file = dir_path + 'sim_' + batch_name + '.log'

    # Silently remove prior log_files and set up logger
    with suppress(OSError):
        os.remove(log_file)

    logging.basicConfig(level=logging.ERROR, filename=log_file,
                        format='%(message)s')

    # Load input parameters and set up accumulator structures
    input_file = 'inputs/config_mkt_data.xlsx'
    base_params = sopm.load_params(input_file)
    sim_batch = init_batch()
    sop.pd_to_csv(sim_batch, 'sim_batch', suffix=batch_name, dir_path=dir_path)
    n_paths = 10

    for idx, overrides in sim_batch.iterrows():

        if True: #int(idx) > 53:
            params = gen_sim_params(base_params, overrides)

            print(f"Running simulation: {idx}")
            logging.info(f"\nRuning simulation: {idx}\n")
            print(overrides)

            # Run simulation for a single parameter set
            sim_name = f"{batch_name}_{idx}"
            run_one_param_set(params, n_paths=n_paths, name=sim_name,
                              overwrite=True, dir_path=dir_path)

    print("\nDone")
