"""
    Run multi-path harvesting simulation.
    Use multiple paths
    by V. Ragulin, started 12-Aug-2022
"""
import datetime as dt
import logging
import os
import time
from contextlib import suppress

import numpy as np
import pandas as pd

pd.options.display.float_format = '{:,.4f}'.format

from pretty_print import df_to_format
import sim_one_path as sop
import sim_one_path_mdata as sopm
from sim_multi_path import process_path_stats

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
    input_file = 'inputs/config_mkt_data.xlsx'
    params = sopm.load_params(input_file)
    data_dict = sopm.load_data(params, randomize=params['randomize'])

    logging.info('Inputs:\n' + str(params) + '\n')
    n_paths = 5
    path_stats = []

    # Run simulation
    tic = time.perf_counter()  # Timer start
    for i in range(n_paths):
        print(f"Path #{i + 1}")
        logging.info(f"\nPath #{i + 1}:")
        one_path_summary, one_path_steps = sop.run_sim(data_dict, suffix=timestamp)
        path_stats.append((one_path_summary, one_path_steps))

        if i < n_paths - 1:  # if it's not the last iteration - update data
            data_dict = sopm.load_data(params, randomize=params['randomize'])

    toc = time.perf_counter()  # Timer end

    # Process simulation results
    t_total = toc - tic
    t_path = t_total / n_paths
    print(f'Simulation took {t_total:0.4f} sec., {t_path:0.4f} sec. per path.\n')

    sim_summary, sim_stats, steps_report = process_path_stats(path_stats)

    print("\nSimulation results by path (annualized, %):")
    print(df_to_format(sim_stats * 100, formats={'_dflt': '{:.2f}'}))
    sop.pd_to_csv(sim_stats, 'sim_stats', suffix=timestamp)

    print("\nSimulation summary (annualized, %):")
    print(df_to_format(sim_summary * 100, formats={'_dflt': '{:.2f}'}))
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
                       formats={'_dflt': '{:.2f}'},
                       multipliers={'port_val': 0.01, 'bmk_val': 0.01}
                       ))

    print("\nDone")
