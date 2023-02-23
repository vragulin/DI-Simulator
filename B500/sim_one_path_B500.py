"""
    Direct index / tax harvesting simulation.  One path simulation using market data from pickle filessim_one_path_mdata.py
    by V. Ragulin, started 15-Aug-2022
"""
import datetime as dt
import logging
import os
import time
import warnings
from contextlib import suppress

import pandas as pd

from pretty_print import df_to_format
from sim_one_path import run_sim
from load_data_B500 import load_params, load_data


# ***********************************************************
# Entry point
# ***********************************************************
if __name__ == "__main__":
    timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = '../results/sim_' + timestamp + '.log'

    # Silently remove prior log_files and set up logger
    with suppress(OSError):
        os.remove(log_file)

    logging.basicConfig(level=logging.INFO, filename=log_file,
                        format='%(message)s')
    # Run simulation
    tic = time.perf_counter()  # Timer start

    # Load simulation parameters
    input_file = '../inputs/config_B500_1path.xlsx'
    params = load_params(input_file)

    # Load simulation data files
    dir_path = r'C:\Users\vragu\OneDrive\Desktop\Proj\DirectIndexing\data\indices\B500'
    data_files = {'px': os.path.join(dir_path, 'B500_sim_px.pickle'),
                  'tri': os.path.join(dir_path, 'B500_sim_tri.pickle'),
                  'w': os.path.join(dir_path, 'B500_sim_w.pickle')}

    data_dict = load_data(data_files, params)

    sim_stats, step_report = run_sim(data_dict, suffix=timestamp)

    toc = time.perf_counter()  # Timer start
    print(f'Simulation took {(toc - tic):0.4f} sec.\n')

    print("Trade Summary (x100):")
    print(df_to_format(step_report * 100, formats={'_dflt': '{:.2f}'}))

    print("\nSimulation statistics:")
    print(df_to_format(pd.DataFrame(sim_stats * 100, columns=["annualized (%)"]), formats={'_dflt': '{:.6f}'}))

    print("\nDone")
