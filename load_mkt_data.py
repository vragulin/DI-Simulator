""" Elm's code (James) to load data from pickle files
    Adapted by Vlad, 15-Aug-2022
"""
import os
import numpy as np
import pandas as pd
from typing import Optional
import index_math as im
import config

# Data location
working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
PX_PICKLE = "idx_prices.pickle"
TR_PICKLE = "idx_t_rets.pickle"
W_PICKLE = "idx_daily_w.pickle"


def load_mkt_data(data_freq: int, data_dict: Optional[dict] = None, replace: bool = False,
                  randomize: bool = False, return_override: Optional[float] = None,
                  vol_scaling: float = 1.0, fixed_weights: bool = True) -> dict:
    """
    :param data_freq:  number of trading days between rebalance points
    :param data_dict:  parameter dictionary
    :param replace: if True, sample with replacement when randomly selecting historical returns
    :param randomize: generate price history by randomly selecting historical returns (by days)
    :param return_override: adjust simulated index return to be equal to a specific number
    :param vol_scaling: rescale volatility of the series by a factor,

            keep mean geometric return same or equal to override below
    :param fixed_weights: if True assume fixed index weights for the stocks, otherwise, assume equal shares
                            (and weights evolve in line with stock prices)
    :return: dict of arrays with simulated or historical prices, returns, weights
    """

    if data_dict is None:
        px = pd.read_pickle(os.path.join(working_dir, PX_PICKLE)).fillna(method='ffill').fillna(0)
        px.index = range(0, len(px.index))
        px = px.reindex(index=range(0, len(px.index), data_freq))

        tri = pd.read_pickle(os.path.join(working_dir, TR_PICKLE)).fillna(method='ffill').fillna(0)
        tri.index = range(0, len(tri.index))
        tri = tri.reindex(index=range(0, len(tri.index), data_freq))

        w = pd.read_pickle(os.path.join(working_dir, W_PICKLE)).fillna(0)
        w.index = range(0, len(w.index))
        w = np.maximum(0, w.reindex(index=range(0, len(w.index), data_freq)))
        w /= (np.sum(w.to_numpy(), axis=1)[:, None] @ np.ones((1, len(w.columns))))

        out_dict = {'px': px, 'tri': tri, 'w': w}

        # Build returns
        out_dict['d_px'] = out_dict['px'].pct_change().fillna(0).replace(np.inf, 0)
        out_dict['d_tri'] = out_dict['tri'].pct_change().fillna(0).replace(np.inf, 0)

        # Check that we don't have negative dividends, this can happen if we have bad input data
        out_dict['d_tri'] = np.maximum(out_dict['d_tri'], out_dict['d_px'])

        out_dict['div'] = out_dict['d_tri'] - out_dict['d_px']

        # Keep track of whether volatility has been rescaled
        out_dict['vol_mult'] = 1.0
    else:
        out_dict = data_dict

    # Only use series with good data
    px = out_dict['px']
    n_steps = px.shape[0] - 1
    freq = config.ANN_FACTOR / data_freq

    # find stocks present for the entire sample and (for fixed share indices) ones that have not risen too much and pull out everything else
    max_rise = 1e10 if fixed_weights else config.max_rise  # only check max_rise for fixed shares indices
    full_indic = (
            (px > px.loc[0, :] * (1 - config.max_drop)) &
            (px < px.loc[0, :] * (1 + max_rise))
    ).all().values

    if randomize:
        d_px = out_dict['d_px'].iloc[:, full_indic].sample(frac=1, replace=replace).reset_index(drop=True)
    else:
        d_px = out_dict['d_px'].iloc[:, full_indic].reset_index(drop=True)
    d_px.iloc[0, :] = 0

    # Set random weights
    rand_weights = np.exp(np.random.normal(size=(1, d_px.shape[1])))
    rand_weights /= np.sum(rand_weights)

    # Rescale volatility of the series if needed
    if 'vol_mult' not in out_dict:
        out_dict['vol_mult'] = 1
    elif vol_scaling != out_dict['vol_mult']:
        # Row 0 of d_px is all zeros, so update everything from row 1
        d_px.values[1:, :] = im.rescale_frame_vol(d_px.values[1:, :],
                                                  vol_scaling / out_dict['vol_mult'])
        out_dict['vol_mult'] = vol_scaling

    # Get returns approximately equal to override
    if return_override is not None:
        if fixed_weights:
            rand_return = freq * (
                    np.product(1 + (rand_weights * d_px).sum(axis=1)
                               ) ** (1 / n_steps) - 1
            )
        else:  # fixed shares
            rand_return = freq * ((rand_weights @ np.product(1 + d_px, axis=0)
                                   ) ** (1 / n_steps) - 1
                                  )[0]  # Convert to scalar from an array

        d_px.iloc[1:, :] += (return_override - rand_return) / freq

    # Ensure that prices don't go below max_drop to avoid computational issues
    d_px = np.maximum(d_px, -config.max_drop)

    # Pack results into the output dictionary
    out_dict['d_px'] = d_px
    out_dict['px'] = px = (1 + d_px).cumprod()
    out_dict['div'] = out_dict['div'].iloc[:, full_indic].reset_index(drop=True)
    out_dict['d_tri'] = out_dict['d_px'] + out_dict['div']

    if fixed_weights:
        out_dict['w'] = np.ones((n_steps + 1, 1)) @ rand_weights
    else:  # fixed shares
        out_dict['w'] = im.index_weights_over_time(rand_weights, px.values)

    # Vectorize
    for k in ('d_px', 'd_tri', 'div', 'w'):
        if isinstance(out_dict[k], pd.DataFrame):
            out_dict[k + '_arr'] = out_dict[k].to_numpy()
        else:
            out_dict[k + '_arr'] = out_dict[k]

    # Capture other settings
    out_dict['fixed_weights'] = fixed_weights

    return out_dict


# Entry point
if __name__ == "__main__":
    # Build inputs
    inputs = {'dt': 60,
              'tau_div_start': 0.0,
              'tau_div_end': 0.0,
              'tau_st_start': 0.0,
              'tau_st_end': 0.0,
              'tau_lt_start': 0.0,
              'tau_lt_end': 0.0,
              'donate_start_pct': 0.00,
              'donate_end_pct': 0.00,
              'div_reinvest': False,
              'div_payout': True,
              'div_override': 0.00,
              'harvest': 'none',
              'harvest_thresh': -0.02,
              'harvest_freq': 60,
              'clock_reset': False,
              'rebal_freq': 60,
              'donate_freq': 240,
              'donate_thresh': 0.0,
              'terminal_donation': 0,
              'donate': False,
              'replace': False,
              # 'randomize': False,
              'randomize': True,
              'return_override': -1,
              'N_sim': 1,
              'savings_reinvest_rate': -1,
              'loss_offset_pct': 1,
              }

    data_dict = load_mkt_data(inputs['dt'], replace=inputs['replace'], randomize=inputs['randomize'],
                              return_override=inputs['return_override'])

    print('Done')
