"""
Test module to validate index calculations with variable weights
V. Ragulin - started 2-Feb-2023
"""
import pytest as pt
import index_math as im
import numpy as np
import pandas as pd


@pt.fixture()
def stock_prices() -> pd.DataFrame:
    # Load stock prices
    data_file = '../scratch/Test_idx_with_variable_weights_data.xlsx'
    px = pd.read_excel(data_file, sheet_name='Prices', index_col=0, header=0)
    return px


@pt.fixture()
def var_weights() -> pd.DataFrame:
    # Load weights
    data_file = '../scratch/Test_idx_with_variable_weights_data.xlsx'
    w = pd.read_excel(data_file, sheet_name='Weights', index_col=0, header=0)
    return w


@pt.fixture
def exp_vals_rets() -> pd.DataFrame:
    exp_vals = np.array([1, 1.316666667, 2.340075758, 2.970336162, 2.427188978, 3.114892521])
    exp_rets = np.zeros(exp_vals.shape)
    exp_rets[1:] = exp_vals[1:] / exp_vals[:-1] - 1

    df = pd.DataFrame(exp_vals, columns=['idx_val'])
    df['idx_ret'] = exp_rets
    return df


def test_idx_vals_from_weights(var_weights, stock_prices, exp_vals_rets):
    # Load input data setup
    px = stock_prices
    w = var_weights
    df_exp = exp_vals_rets

    idx_vals, idx_rets = im.index_vals_from_weights(w.values, px.values)

    np.testing.assert_allclose(idx_vals, df_exp['idx_val'])
    np.testing.assert_allclose(idx_rets, df_exp['idx_ret'])


def test_idx_vals_from_weights_pd(var_weights, stock_prices, exp_vals_rets):
    # Load input data setup
    px = stock_prices
    w = var_weights
    df_exp = exp_vals_rets

    df = im.index_vals_from_weights_pd(w, px)

    np.testing.assert_allclose(df['idx_val'].values, df_exp['idx_val'].values)
    np.testing.assert_allclose(df['idx_ret'].values, df_exp['idx_ret'].values)
