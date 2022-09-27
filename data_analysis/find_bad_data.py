""" Find bad data in the datafiles """
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

# Constants
# Data files
# Which dataset to use - clean (if True) or full (if False)
clean_flag = True
data_freq = 20

if clean_flag:
    WORKING_DIR = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp_clean/"
else:
    WORKING_DIR = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
PX_PICKLE = "idx_prices.pickle"
TR_PICKLE = "idx_t_rets.pickle"
W_PICKLE = "idx_daily_w.pickle"


def load_data() -> dict:
    # Load price file
    px0 = pd.read_pickle(pathlib.Path(WORKING_DIR) / PX_PICKLE
                         ).fillna(method='ffill').fillna(0)
    dates = px0.index
    dates_idx = range(0, len(px0.index), data_freq)
    px0.index = range(0, len(px0.index))
    px0 = px0.reindex(index=dates_idx)

    # Filter out stocks with full history
    full_indic: np.array = (px0 > 0).all().values

    px = px0.iloc[:, full_indic]
    d_px = np.log(px).diff().fillna(0)
    d_px_arr = d_px.to_numpy()
    n_steps, n_stocks = d_px_arr.shape - np.array((-1, 0))

    # Start of analysis - calculate autocovariance for all stocks for different horizons, and rank
    n_lags = 12
    ac = pd.DataFrame(0, index=range(n_lags), columns=px.columns)

    for lag in range(n_lags):
        for stk in range(n_stocks):
            # print(stk)
            # Calculate auto-covariance for each stock at lag l
            ac.iloc[lag, stk] = np.cov((d_px_arr[lag + 1:, stk], d_px_arr[:-lag - 1, stk]))[0, 1]

    # Sort stocks by avg autocorreltaion
    abs_ac_sorted = np.abs(ac).max(axis=0).sort_values()

    # Pack everything into a dictionary
    out_dict = {'px': px,
                'd_px': d_px,
                'dates': dates,
                'dates_idx': dates_idx,
                'acov': ac,
                'abs_acov_sorted': abs_ac_sorted}

    return out_dict


def plot(dfx, ax, name='name'):
    ax.plot(dfx)
    ax.grid()

    # Title
    ax.set_title(f'{name}', fontsize=20)


def plot_stocks(data_dict: dict, log: bool = False) -> None:
    """ Plot most suspicious stocks """
    nrows, ncols = 4, 3
    nstocks = nrows * ncols
    stk_list = data_dict['abs_acov_sorted'].index[-nrows * ncols:]
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 20))

    if log:
        px = np.log10(data_dict['px'])
    else:
        px = data_dict['px']

    px.index = data_dict['dates'][data_dict['dates_idx']]

    for i, name in enumerate(stk_list):
        r = (nstocks - i - 1) % nrows
        c = (nstocks - i - 1) // nrows
        plot(px[name], ax[r, c], name)

    plt.tight_layout()
    plt.show()


# Entry Point
if __name__ == "__main__":
    data_dict = load_data()

    plot_stocks(data_dict, log=False)
    plot_stocks(data_dict, log=True)

    print('Done')

    # plot_stocks(bad_stocks)
