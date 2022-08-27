"""
Useful functions to calculate index returns, weights, etc...
"""
import numpy as np
from typing import Optional, Union


def index_avg_return(shares: np.array, prices: np.array) -> float:
    """ Calculate average simple period return for an index
        This function assumes a static index with no rebalancing and fixed
        number of shares over time
        Axis 0 is time, axis 1 - stocks.

        Parameters:
        :param shares: starting number of shares
                       (or initial weights if all starting px as the same)
        :param prices: individual stock prices

        :return average simple return per period
    """

    n_periods = prices.shape[0] - 1
    idx_starting_val = (shares * prices[0, :, None]).sum()
    idx_final_val = (shares * prices[-1, :, None]).sum()

    avg_period_idx_ret = (idx_final_val / idx_starting_val) ** (1 / n_periods) - 1

    return avg_period_idx_ret


def index_weights_over_time(shares: np.array, prices: np.array) -> np.array:
    """ Calculate index weights over time for a static index w/o rebalancing
        Axis 0 is time, axis 1 - stocks.

    :param shares: number of shares
                   (or initial weights if all starting px as the same)
    :param prices: individual stock prices (or returns indices
                    if the first variable gives initial weights)
    :return: weights over time for each stock
    """
    try:
        stock_vals = prices * shares.T
    except ValueError:
        stock_vals = prices * shares

    idx_vals = stock_vals.sum(axis=1)

    return stock_vals / idx_vals.reshape(stock_vals.shape[0], -1)


def index_vals(shares: np.array, prices: np.array, norm=False) -> np.array:
    """ Calculate index weights over time for a static index w/o rebalancing
        Axis 0 is time, axis 1 - stocks.

    :param shares: number of shares
                   (or initial weights if all starting px as the same)
    :param prices: individual stock prices (or returns indices
                    if the first variable gives initial weights)
    :param norm: normalize starting index value to 1 (def True)
    :return: index values over time
    """
    idx_vals = prices @ shares
    if norm:
        return idx_vals / idx_vals[0]
    else:
        return idx_vals


def total_ret_index(shares: np.array, prices: np.array, idx_div: np.array,
                    idx_vals: Optional[np.array] = None) -> np.array:
    """ Calculate index total returns
    :param shares: starting number of shares
    :param prices: individual stock prices (or returns indices
                    if the first variable gives initial weights)
    :param idx_div:  index divs in points
    :param idx_vals: optional - can supply idx_vals to save calculation time
    :return: total return index, starting from 1
    """

    # Calculate idx_vals if not supplied
    if idx_vals is None:
        idx_vals = index_vals(shares, prices)

    # idx_div_y = idx_div / idx_vals
    idx_div_y = np.zeros(idx_vals.shape)
    idx_div_y[1:] = idx_div[1:] / idx_vals[:-1]
    idx_px_rets = idx_vals[1:] / idx_vals[:-1] - 1
    idx_tot_period_rets = idx_px_rets + idx_div_y[1:]
    idx_tri = np.ones(len(idx_vals))[:, None]
    idx_tri[1:, 0] = np.cumprod(idx_tot_period_rets + 1)

    return idx_tri


def rescale_ts_vol(d_px: np.array, vol_mult: float, new_g_mean: Optional[float] = None) -> np.array:
    """ Rescale volatility for a time series of price percentage changes
     :param d_px: series of price percent changes
     :param vol_mult: multipler by which we want to increase the volatility
     :param new_g_mean: target expected (geometric) return of the new series, if None keep it the same as the input
     :return np.array: new series of re-scaled percent changes
     """
    n_steps = len(d_px)

    if new_g_mean is None:
        new_g_mean = (1 + d_px).prod() ** (1 / n_steps) - 1

    # Scale up volatity (vol0 ** vol_mult is the same as multiplying log_vol0 by vol_mult).
    s1 = (1 + d_px) ** vol_mult

    # Adjust the growth rate to get the same cumulative return
    g_mean1 = s1.prod() ** (1 / n_steps) - 1
    adj_factor = (1 + new_g_mean) / (1 + g_mean1)

    return (s1 * adj_factor) - 1


def rescale_frame_vol(d_px: np.array, vol_mult: float,
                      new_g_mean: Optional[Union[np.array, float]] = None) -> np.array:
    """ Rescale volatility for a dataframe
    :param d_px: 2-d array of price changes, axis 0 is time,a axis1 is securities
    :param vol_mult: multipler by which we want to increase the volatility
    :param new_g_mean: target expected (geometric) return of the new series,
        can be float or array same size as axis1
        if None keep it the same as the input
    :return np.array: new series of re-scaled percent changes
    """

    n_steps, n_stocks = d_px.shape

    if new_g_mean is None:
        new_g_mean = (1 + d_px).prod(axis=0) ** (1 / n_steps) - 1

    # Scale up volatity (vol0 ** vol_mult is the same as multiplying log_vol0 by vol_mult).
    s1 = (1 + d_px) ** vol_mult

    # Adjust the growth rate to get the same cumulative return
    g_mean1 = s1.prod(axis=0) ** (1 / n_steps) - 1
    adj_factor = (1 + new_g_mean) / (1 + g_mean1)

    return (s1 * adj_factor) - 1
