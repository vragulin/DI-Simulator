"""
Run a simulation with an optimizer.
by V. Ragulin.  Started 27-Aug-22
"""
import os
import datetime as dt
import logging
import pandas as pd
from contextlib import suppress

import sim_one_path as sop
from pretty_print import df_to_format

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
    input_file = 'inputs/test5.xlsx'
    inputs = sop.load_sim_settings_from_file(input_file, randomize=False)
    sim_stats, step_report = sop.run_sim(inputs, suffix=timestamp, log_file=log_file)

    print("\nTrade Summary (x100, %):")
    print(df_to_format(step_report * 100, formats={'_dflt': '{:.2f}'}))

    print("\nSimulation statistics (%):")
    print(df_to_format(pd.DataFrame(sim_stats * 100, columns=["annualized (%)"]), formats={'_dflt': '{:.2f}'}))
    print("\nDone")