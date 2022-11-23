"""
Fast functions used to calculate tax liability on a multi-stock portfolio of lots
"""
from typing import Iterable

import pandas as pd
import numpy as np
from numba import njit


@njit
def intervaled_cumsum0(ar: np.ndarray, sizes: np.array) -> np.ndarray:
    """ Compute a cumsum across intervals (each interval starts from zero)
        Params:  array, sizes of intervals for cumsum()
        Return:  array with parial (intervaled) cumsum results
    """

    # Make a copy to be used as output array
    out = ar.copy()

    # Get cumumlative values of array
    arc = ar.cumsum()

    # Get cumsumed indices to be used to place differentiated values into
    # input array's copy
    idx = sizes.cumsum()

    # Place differentiated values that when cumumlatively summed later on would
    # give us the desired intervaled cumsum
    out[idx[0]] = ar[idx[0]] - arc[idx[0] - 1]
    out[idx[1:-1]] = ar[idx[1:-1]] - np.diff(arc[idx[:-1] - 1])
    return out.cumsum()


@njit
def intervaled_cumsum(ar: np.ndarray, idx: np.array) -> np.ndarray:
    """ Compute a cumsum across intervals (idx - starts of intervals)
        Params:  array, index of starting positions for cumsum()
        Return:  array with parial (intervaled) cumsum results
    """
    # Make a copy to be used as output array
    out = ar.copy()

    # Get cumumlative values of array
    arc = ar.cumsum()

    # Place differentiated values that when cumumlatively summed later on would
    # give us the desired intervaled cumsum
    out[idx[1]] = ar[idx[1]] - arc[idx[1] - 1]
    out[idx[2:]] = ar[idx[2:]] - np.diff(arc[idx[1:] - 1])
    return out.cumsum()


@njit
def intervaled_diff0(ar: np.ndarray, sizes: np.array) -> np.ndarray:
    """ Calculate differences by interval, first element of each interval is unchanged
        This is an inverse of intervaled_cumsum0 above
        Params:  array, sizes of intervals for cumsum()
        Return:  array with parial (intervaled) cumsum results

    """

    # Get cumsumed indices to be used to place correct values at the start of each interval
    idx = sizes.cumsum()

    # out = np.diff(ar, prepend=0)
    out = np.empty(ar.shape)
    out[1:] = np.diff(ar)
    out[0] = ar[0]
    out[idx[:-1]] = ar[idx[:-1]]
    return out


@njit
def intervaled_diff(ar: np.ndarray, idx: np.array) -> np.ndarray:
    """ Calculate differences by interval, first element of each interval is unchanged
        This is an inverse of intervaled_cumsum above
        Params:  array, index of starting positions for cumsum()
        Return:  array with parial (intervaled) cumsum results
    """

    # out = np.diff(ar, prepend=0)
    out = np.empty(ar.shape)
    out[1:] = np.diff(ar)

    out[idx] = ar[idx]
    return out


def block_start_index_pd(df: pd.DataFrame, col='Ticker') -> np.array:
    """ Calculate index of rows where new blocks of tickers start.
        This version uses pandas
        Assume that the key column is aleady sorted
        Input: df - dataframe, col - name of column to analyse
    """
    prev_non_equal = df[col].values != df[col].shift().values
    nz = np.nonzero(prev_non_equal)
    return np.array(nz).astype(int).ravel()


@njit
def block_start_index(arr: np.ndarray) -> np.array:
    """ Calculate index of rows where new blocks of tickers start
        This version only uses numpy
        Assume that the array is aleady sorted
        Input: arr
        Return: array of indices
    """
    prev_non_equal = arr[1:] != arr[:-1]
    nz, _ = np.nonzero(prev_non_equal)
    out = np.hstack([0, nz + 1])
    return out


@njit
def build_broadcast_index(sizes: np.array) -> np.array:
    """ Build an index for broadcasting a variable from a stock column to a lots column """
    nout = np.sum(sizes)
    out = np.zeros(nout, dtype=np.int32)
    counter = 0
    for i, s in enumerate(sizes):
        out[counter:counter + s] = i
        counter += s
    return out
