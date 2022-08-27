""" Process output of a batch run
    Collect simulation statistics into a single dataframe
    by V. Ragulin, started 24-Aug-22
"""
import os
import numpy as np
import pandas as pd
import sim_one_path as sop
from typing import Optional


def process_sim_file(idx: int, dir_path: str, batch_name: str) -> Optional[pd.Series]:
    """ Process output of a single run in a batch """
    #  Check if file exists
    fname = f"sim_summary_{batch_name}_{int(idx)}.csv"
    fpath = os.path.join(dir_path, fname)

    #  Read file
    if os.path.exists(fpath):
        data = pd.read_csv(fpath, index_col=0)
    else:
        return None

    #  Pack data into a tuple
    return data['Avg']


# Entry Point
if __name__ == "__main__":

    # Directory paths and file formats
    dir_path = 'results/batch_thresh_0.02/'
    batch_name = 'prod10'

    # Read simulation settings file
    f_batch = os.path.join(dir_path,
                           f"sim_batch_{batch_name}.csv")

    b_params = pd.read_csv(f_batch, index_col='run_id')

    # Process individual sim files, save results into a dataframe
    cols = ['port_ret', 'index_ret', 'index_vol',
            'tracking_std', 'hvst_grs', 'hvst_net',
            'hvst_potl', 'hvst_grs/potl',
            'hvst_net/potl', 'hvst_n/trckng']

    b_res = pd.DataFrame(np.nan, columns=cols, index=b_params.index)
    for idx in b_params.index:

        sim_stats = process_sim_file(idx, dir_path, batch_name)
        if sim_stats is None:
            print(f"Results for sim={idx} not found.")
        else:
            b_res.loc[idx, :] = sim_stats

        # if int(idx) == 70:
        #     break

    batch_summ = pd.concat([b_params, b_res], axis=1)

    # Save summary dataframe
    print(batch_summ)
    sop.pd_to_csv(batch_summ, 'batch_summary', suffix=batch_name, dir_path=dir_path)

    print("Done")
