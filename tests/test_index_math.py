""" Test index_math library """
import pytest as pt
from main import load_sim_settings_from_file
import index_math as im
import numpy as np
import config

@pt.fixture
def sim5():
    f = 'inputs/test5.xlsx'
    sim = load_sim_settings_from_file(f)
    return sim

def test_portfolio_has_correct_n_stocks(sim5):
    assert len(sim5['w_idx']) == 5


def test_sum_weights(sim5):
    assert pt.approx(sim5['w_idx'].values.sum()) == 1.0


def test_idx_avg_return_is_correct(sim5):
    shares = sim5['w_idx'].values
    prices = sim5['prices'].values
    px_norm = prices / prices[0, :]

    expected = 0.032925867
    actual = im.index_avg_return(shares, px_norm)

    assert pt.approx(actual, 1.0e-5) == expected


def test_idx_weights_over_time(sim5):
    shares = sim5['w_idx'].values
    prices = sim5['prices'].values
    px_norm = prices / prices[0, :]

    w = im.index_weights_over_time(shares, px_norm)

    # Check that add up to 1
    exp_sum = 1
    act_sum = w.sum(axis=1)
    # diff = actual - expected
    # assert pt.approx(np.std(diff), abs=1e-6) == 0
    np.testing.assert_allclose(exp_sum, act_sum, atol=1e-6)

    # Check first row
    exp_row1 = [0.3, 0.260157, 0.288277, 0.311762, 0.333952, 0.33168]
    act_row1 = w[:, 0]
    np.testing.assert_allclose(exp_row1, act_row1, atol=1e-6)


def test_idx_vals_no_norm(sim5):
    shares = sim5['w_idx'].values
    prices = sim5['prices'].values
    px_norm = prices / prices[0, :]

    expected = [1, 1.037833, 1.040667, 1.0585, 1.078, 1.175833]
    actual = im.index_vals(shares, px_norm)
    # diff = actual - expected
    # assert pt.approx(np.std(diff), abs=1e-6) == 0
    np.testing.assert_allclose(expected, actual, atol=1e-6)

def test_idx_vals_norm(sim5):
    shares = sim5['w_idx'].values
    prices = sim5['prices'].values

    expected = [1, 1.083019, 1.045283, 1.071698, 1.05283, 1.158491]
    actual = im.index_vals(shares, prices, norm=True)
    # diff = actual - expected
    # assert pt.approx(np.std(diff), abs=1e-6) == 0
    np.testing.assert_allclose(expected, actual, atol=1e-6)

def test_total_ret_index(sim5):
    shares = sim5['w_idx'].values
    prices = sim5['prices'].values
    px_norm = prices / prices[0, :]

    # Calc dividends array
    dt = sim5['params'].loc['dt', 'Value']
    w_start = sim5['w_idx'].values
    div_y_per_dt = sim5['stk_info']['Div Yield'] * dt / config.ANN_FACTOR
    div = px_norm * div_y_per_dt.values
    idx_div = div @ w_start

    expected = [1, 1.040397, 1.045709, 1.066229, 1.088329, 1.189665]
    actual = im.total_ret_index(shares, px_norm, idx_div)
    np.testing.assert_allclose(expected,actual,atol=1e-6)
