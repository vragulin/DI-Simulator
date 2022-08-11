"""
Useful functions to calculate index returns, weights, etc...
"""
import numpy as np


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
    stock_vals = prices * shares.T
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


def total_ret_index(shares: np.array, prices: np.array, idx_div: np.array) -> np.array:
    """ Calculate index total returns
    :param shares: starting number of shares
    :param prices: individual stock prices (or returns indices
                    if the first variable gives initial weights)
    :param idx_div:  index divs in points
    :return: total return index, starting from 1
    """

    idx_vals = index_vals(shares, prices)
    idx_div_y = idx_div / idx_vals
    idx_px_rets = idx_vals[1:] / idx_vals[:-1] - 1
    idx_tot_period_rets = idx_px_rets + idx_div_y[1:]
    idx_tri = np.ones(idx_vals.shape)
    idx_tri[1:, 0] = np.cumprod(idx_tot_period_rets + 1)

    return idx_tri
