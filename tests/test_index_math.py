""" Test index_math library """
import pytest as pt
from sim_one_path import load_sim_settings_from_file
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


def test_idx_weights_shares_transposed(sim5):
    shares = sim5['w_idx'].values.T
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

    expected = np.array([1, 1.037833, 1.040667, 1.0585, 1.078, 1.175833]).reshape(-1, 1)
    actual = im.index_vals(shares, px_norm)
    # diff = actual - expected
    # assert pt.approx(np.std(diff), abs=1e-6) == 0
    np.testing.assert_allclose(expected, actual, atol=1e-6)


def test_idx_vals_norm(sim5):
    shares = sim5['w_idx'].values
    prices = sim5['prices'].values

    expected = np.array([1, 1.083019, 1.045283, 1.071698, 1.05283, 1.158491]).reshape(-1, 1)
    actual = im.index_vals(shares, prices, norm=True)
    # diff = actual - expected
    # assert pt.approx(np.std(diff), abs=1e-6) == 0
    np.testing.assert_allclose(expected, actual, atol=1e-6)


def test_total_ret_index(sim5):
    shares = sim5['w_idx'].values
    prices = sim5['prices'].values
    px_norm = prices / prices[0, :]

    # Calc dividends array
    dt = sim5['params'].loc['dt']
    w_start = sim5['w_idx'].values
    div_y_per_dt = sim5['stk_info']['Div Yield'] * dt / config.ANN_FACTOR
    div = np.zeros(prices.shape)
    div[1:] = px_norm[:-1] * div_y_per_dt.values
    idx_div = div @ w_start

    expected = np.array([1, 1.040208, 1.045715, 1.066119, 1.08841, 1.189697]).reshape(-1, 1)
    actual = im.total_ret_index(shares, px_norm, idx_div)
    np.testing.assert_allclose(expected, actual, atol=1e-6)


def test_total_ret_index_provide_idx_vals(sim5):
    shares = sim5['w_idx'].values
    prices = sim5['prices'].values
    px_norm = prices / prices[0, :]

    # Calc dividends array
    dt = sim5['params'].loc['dt']
    w_start = sim5['w_idx'].values
    div_y_per_dt = sim5['stk_info']['Div Yield'] * dt / config.ANN_FACTOR
    div = np.zeros(prices.shape)
    div[1:] = px_norm[:-1] * div_y_per_dt.values
    idx_div = div @ w_start

    idx_vals = im.index_vals(shares, px_norm)

    expected = np.array([1, 1.040208, 1.045715, 1.066119, 1.08841, 1.189697]).reshape(-1, 1)
    actual = im.total_ret_index(shares, px_norm, idx_div, idx_vals=idx_vals)
    np.testing.assert_allclose(expected, actual, atol=1e-6)


@pt.fixture
def d_px():
    px = np.array([1, 1.1, 0.9, 1.15, 0.95, 1.1])
    dp = px[1:] / px[:-1] - 1
    return dp


def test_rescale_ts_vol_no_override(d_px):
    vol_mult = 1.25
    expected = np.array(
        [0.121169, -0.225551, 0.352071, -0.216187, 0.195410]
    )
    actual = im.rescale_ts_vol(d_px, vol_mult=vol_mult)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_rescale_ts_vol_override(d_px):
    vol_mult = 1.25
    new_g_mean = -0.01
    expected = np.array(
        [0.089, -0.247772, 0.313277, -0.238677, 0.161110]
    )
    actual = im.rescale_ts_vol(d_px, vol_mult=vol_mult, new_g_mean=new_g_mean)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pt.fixture
def d_px_2d():
    px = np.array(
        [[1, 1.1, 0.9, 1.15, 0.95, 1.1],
         [1, 1, 1, 1, 1, 1],
         [1, 0.9, 1.1, 0.85, 1, 0.9]]
    ).T
    dp = px[1:, :] / px[:-1, :] - 1
    return dp


def test_rescale_frame_vol_no_override(d_px_2d):
    vol_mult = 1.25
    expected = np.array(
        [[0.121169, -0.225551, 0.352071, -0.216187, 0.19541],
         [0, 0, 0, 0, 0],
         [-0.118766, 0.29189, -0.271683, 0.231726, -0.118766]]
    ).T
    actual = im.rescale_frame_vol(d_px_2d, vol_mult=vol_mult)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_rescale_frame_vol_override_float(d_px_2d):
    vol_mult = 1.25
    new_g_mean = 0.01
    expected = np.array([
        [0.111, -0.232576, 0.339807, -0.223296, 0.184567],
        [0.01, 0.01, 0.01, 0.01, 0.01],
        [-0.091, 0.332596, -0.248734, 0.270536, -0.091]
    ]).T
    actual = im.rescale_frame_vol(d_px_2d, vol_mult=vol_mult, new_g_mean=0.01)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_rescale_frame_vol_override_array(d_px_2d):
    vol_mult = 1.25
    new_g_mean = np.array([0.01, -0.01, 0.02])
    expected = np.array([
        [0.111, -0.232576, 0.339807, -0.223296, 0.184567],
        [-0.01, -0.01, -0.01, -0.01, -0.01],
        [-0.082, 0.34579, -0.241296, 0.283116, -0.082]
    ]).T
    actual = im.rescale_frame_vol(d_px_2d, vol_mult=vol_mult, new_g_mean=new_g_mean)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pt.fixture
def cash_flow():
    cf = np.array(
        [-1, 0, 0.05, 0, -0.2, 0.2, 0.05, -0.09, 1.1]
    )
    return cf


def test_irr_obj(cash_flow):
    guess = 0.10571
    npv = im.irr_obj(guess, cash_flow, 30, ann_factor=240)
    exp_npv = 0.0
    assert pt.approx(npv, abs=1.0e-6) == exp_npv


def test_irr_solve_default_guess_bnds(cash_flow):
    act_irr = im.irr_solve(cash_flow, 30, 240)
    exp_irr = 0.10571
    assert pt.approx(exp_irr, abs=1.0e-6) == act_irr


def test_irr_solve_my_guess_bnds(cash_flow):
    guess = 0.01
    bounds = (-0.1, +0.5)
    act_irr = im.irr_solve(cash_flow, 30, 240, guess, bounds)
    exp_irr = 0.10571
    assert pt.approx(exp_irr, abs=1.0e-6) == act_irr


def test_irr_solve_ann_freq(cash_flow):
    guess = 0.01
    bounds = (-0.1, +0.5)
    act_irr = im.irr_solve(cash_flow, 30, 240, guess, bounds, 1)
    exp_irr = 0.111499
    assert pt.approx(exp_irr, abs=1.0e-6) == act_irr



def test_irr_solve_qtr_freq(cash_flow):
    guess = 0.01
    bounds = (-0.1, +0.5)
    act_irr = im.irr_solve(cash_flow, 30, 240, guess, bounds, 4)
    exp_irr = 0.107119
    assert pt.approx(exp_irr, abs=1.0e-6) == act_irr


def test_index_liq_tax_all_lt_gains():
    idx_val = np.array([1, 1.2, 1, 0.9, 1.2, 1.3])

    idx_div = np.zeros(idx_val.shape)
    idx_div[1:] = idx_val[:-1] * 0.05

    idx_tri = idx_val.copy()
    idx_tri[1:] = idx_tri[0] * ((idx_val[1:] + idx_div[1:]) / idx_val[:-1]).cumprod()

    tax = im.index_liq_tax(idx_val, idx_div, idx_tri, 240, 240, div_payout=False)
    exp_tax = 0.0985108

    assert pt.approx(exp_tax, abs=1.0e-6) == tax


def test_index_liq_tax_with_st_gains():
    idx_val = np.array([1, 1.039, 1.052, 1.078, 1.103, 1.153,
                        1.156, 1.22, 1.207])
    dt = 60
    idx_div = np.zeros(idx_val.shape)
    idx_div[1:] = idx_val[:-1] * 0.05 * dt / 240

    idx_tri = idx_val.copy()
    idx_tri[1:] = idx_tri[0] * ((idx_val[1:] + idx_div[1:]) / idx_val[:-1]).cumprod()

    tax = im.index_liq_tax(idx_val, idx_div, idx_tri, dt, 240, div_payout=False)
    exp_tax = 0.060482328

    assert pt.approx(exp_tax, abs=1.0e-6) == tax


def test_index_liq_tax_column_vectors():
    idx_val = np.array([1, 1.039, 1.052, 1.078, 1.103, 1.153,
                        1.156, 1.22, 1.207])
    dt = 60
    idx_div = np.zeros(idx_val.shape)
    idx_div[1:] = idx_val[:-1] * 0.05 * dt / 240

    idx_tri = idx_val.copy()
    idx_tri[1:] = idx_tri[0] * ((idx_val[1:] + idx_div[1:]) / idx_val[:-1]).cumprod()

    tax = im.index_liq_tax(idx_val.T, idx_div.T, idx_tri.T, dt, 240, div_payout=False)
    exp_tax = 0.060482328

    assert pt.approx(exp_tax, abs=1.0e-6) == tax


def test_index_liq_tax_all_lt_gains_payout():
    idx_val = np.array([1, 1.2, 1, 0.9, 1.2, 1.3])

    idx_div = np.zeros(idx_val.shape)
    idx_div[1:] = idx_val[:-1] * 0.05

    idx_tri = idx_val.copy()
    idx_tri[1:] = idx_tri[0] * ((idx_val[1:] + idx_div[1:]) / idx_val[:-1]).cumprod()

    tax = im.index_liq_tax(idx_val, idx_div, idx_tri, 240, 240, div_payout=True)
    exp_tax = 0.084

    assert pt.approx(exp_tax, abs=1.0e-6) == tax


def test_index_liq_tax_with_st_gains_payout():
    idx_val = np.array([1, 1.039, 1.1])
    dt = 60
    idx_div = np.zeros(idx_val.shape)
    idx_div[1:] = idx_val[:-1] * 0.05 * dt / 240

    idx_tri = idx_val.copy()
    idx_tri[1:] = idx_tri[0] * ((idx_val[1:] + idx_div[1:]) / idx_val[:-1]).cumprod()

    tax = im.index_liq_tax(idx_val, idx_div, idx_tri, dt, 240, div_payout=True)
    exp_tax = 0.05

    assert pt.approx(exp_tax, abs=1.0e-6) == tax


@pt.fixture
def irr_data():
    idx_div = [0, 0, 0.05, 0, 0.1, 0.02, 0.05, 0.01, 0.1]
    idx_start = 1.0
    idx_end = 0.9
    dt = 30
    ann_factor = 240

    irr_data = {'idx_div': idx_div, 'idx_start': idx_start,
                'idx_end': idx_end, 'dt': dt, 'ann_factor': ann_factor}
    return irr_data


def test_index_irr(irr_data):
    irr = im.index_irr(irr_data['dt'], irr_data['idx_start'],
                       irr_data['idx_end'], irr_data['idx_div'],
                       ann_factor=irr_data['ann_factor'])
    exp_irr = 0.228283792
    assert pt.approx(exp_irr, abs=1.0e-6) == irr


def test_index_irr(irr_data):
    irr = im.index_irr(irr_data['dt'], irr_data['idx_start'],
                       irr_data['idx_end'], irr_data['idx_div'],
                       ann_factor=irr_data['ann_factor'], freq=4)
    exp_irr = 0.234924
    assert pt.approx(exp_irr, abs=1.0e-6) == irr


@pt.fixture
def stock_avg_vol_data():
    d_px_arr = [[0, 0, 0],
                [0.1, 0.1, -0.1],
                [-0.1, 0, 0.2],
                [0.2, 0.1, -0.1],
                [-0.2, 0, 0.2]]
    d_px = np.array(d_px_arr)

    w = np.array([0.4, 0.35, 0.25])

    dt = 0.25
    return d_px, w, dt


def test_stk_avg_vol_zero_dt(stock_avg_vol_data):
    d_px, w, dt = stock_avg_vol_data
    with pt.raises(AssertionError):
        vol_avg = im.stock_vol_avg(d_px, w, 0)


def test_stk_avg_vol(stock_avg_vol_data):
    d_px, w, dt = stock_avg_vol_data
    vol_avg = im.stock_vol_avg(d_px, w, dt)
    exp_vol = 0.233387
    assert pt.approx(exp_vol, abs=1.0e-6) == vol_avg


