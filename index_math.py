"""
Useful functions to calculate index returns, weights, etc...
"""
import numpy as np
from typing import Optional, Union
from scipy.optimize import minimize
from numba import njit

# Throw errors instead of warnings
np.seterr('raise')

# @njit
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


# @njit
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


# @njit
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


# @njit
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


@njit
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


# @njit
def rescale_frame_vol(d_px: np.array, vol_mult: float = 1,
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


# @njit
def irr_obj(r: float, cf: np.array, dt: int,
            ann_factor: float = 252) -> float:
    t = np.linspace(start=0, stop=dt * len(cf), num=len(cf), endpoint=False) / ann_factor
    pv = float(np.sum(cf * np.exp(-r * t)))
    return abs(pv)


# @njit
def irr_solve(cf: np.array, dt: int, ann_factor: float = 252, guess: float = 0.09,
              bounds: tuple = (0.0, 0.2), freq: Optional[int] = None) -> float:
    """ Calculate IRR of an array of cash flows
    :param cf: array of cash flows
    :param dt: length of each step
    :param ann_factor: number of trading days per annum
    :param guess: intiial guess for r
    :param bounds: bounds tuple (lb, up)
    :param freq: frequency of the calculated IRR (default None, or continuous)
    :return: exponential IRR
    """
    x0 = np.asarray(guess)
    r = minimize(irr_obj, x0, args=(cf, dt, ann_factor), bounds=[bounds]).x

    if freq is None:
        return r[0]
    else:
        return (np.exp(r[0] / freq) - 1) * freq


# @njit
def index_liq_tax(idx_val: np.array, idx_div: np.array, idx_tri: Optional[np.array],
                  dt: int, ann_factor: float = 252, tax_lt: float = 0.28,
                  tax_st: float = 0.5, div_payout: bool = True) -> float:
    """ Calculate tax liability for liquidating an index portfolio at the end
    Assume index is based on a fixed basket, so can be treated as single security.
    :param idx_vals:    series of index values (no re-investment of divs)
    :param idx_divs:    serues if ubdex dividends
    :param idx_tri:     include re-invesstment of divs
    :param dt:          length of a time step in trading days
    :param ann_factor:  number of trading days per annum
    :param tax_lt:      long-term gains tax rate
    :param tax_st:      short-term gains tax rate
    :param div_payout:  whether index reinvests divs (default True)
    :return:  tax paid on full liquidation of the index at maturity
    """
    # Treat the index as a single security.  Assume that at each
    # rebalance date we open a new lot (initial purchase + div reinvestment)

    # Flatten input arrays so that we can handle both column and row vectors
    i_val = idx_val.ravel()
    i_div = idx_div.ravel()
    i_tri = idx_tri.ravel()

    # Calculate size of lots for 'reinvesting dividends' (if specified)
    n_points = len(i_val)
    lot_basis = i_val.copy()
    if div_payout:
        lot_basis[1:] = 0
    else:
        lot_basis[1:] = i_div[1:] * i_tri[:-1] / i_val[:-1]

    # Capital gain on each lot
    gain_prc = i_val[-1] / i_val - 1
    gain = lot_basis * gain_prc

    # Applicable tax rates
    lot_dates = np.arange(n_points) * dt
    lt_indic = (lot_dates[-1] - lot_dates) >= ann_factor
    tax_rates = tax_st + lt_indic * (tax_lt - tax_st)

    # Sum tax liability across all lots (pos umber = we pay tax)
    tax = tax_rates @ gain

    return tax


# def index_irr(dt: int, idx_start: float, idx_end: float, idx_div: np.array,
#               guess: float = 0.09, bounds: tuple = (0.0, 0.2), ann_factor: float=252) -> float:
def index_irr(dt: int, idx_start: Union[float, np.ndarray], idx_end: Union[float, np.ndarray],
              idx_div: np.array,
              guess: float = 0.09, bounds: tuple = (-0.5, 0.5),
              ann_factor: float = 252, freq: Optional[int] = None) -> float:
    """ Calculate IRR of an index
    :param dt: length of each time step in days
    :param idx_start: starting value of an index
    :param idx_end: final value of an index
    :param idx_div: vector of index dividends
    :param guess: intiial guess for r
    :param bounds: bounds tuple (lb, up)
    :param ann_factor: number of trading days per annum
    :param freq: frequency of the calculated IRR (default None, or continuous)
    :return: exponential annualized irr of the cash flows
    """

    # If start/end values were given as arrays, convert to float
    i_start = idx_start[0] if isinstance(idx_start, np.ndarray) else idx_start
    i_end = idx_end[0] if isinstance(idx_start, np.ndarray) else idx_end

    cf = idx_div.copy()
    cf[0] -= i_start
    cf[-1] += i_end

    irr = irr_solve(cf, dt, ann_factor=ann_factor, guess=guess, bounds=bounds, freq=freq)

    return irr


def stock_vol_avg(d_px: np.array, w0: np.array, dt: float) -> float:
    """ Calculate weighted average individual stock vol
    for a fixed-share index
    :param d_px: array of price % changes (n_steps+1, n_stocks)
    :param w0: array of initial weights (n_stocks), assume weights
                evolve with share prices
    :param dt: length of a time step in years
    :return: weighted average vol of stocks in the index
    """

    # Vols of individual stocks
    assert np.abs(dt) > np.finfo(float).eps, "Invalid time step, dt=0"

    stk_vols = np.log(1+d_px[1:, :]).std(axis=0) / np.sqrt(dt)
    vol_avg = stk_vols @ w0

    return vol_avg
