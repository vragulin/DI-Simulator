"""
Testing random market data generation
"""
import inspect
import re
import numpy as np

import pytest as pt

from load_mkt_data import load_mkt_data


def varname(p):
    """ Print variable name """
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


@pt.fixture
def inputs():
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
              'randomize': False,
              # 'randomize': True,
              'return_override': -1,
              'N_sim': 1,
              'savings_reinvest_rate': -1,
              'loss_offset_pct': 1,
              }
    return inputs


def test_non_random_correct_sizes(inputs):
    # Test without randomization
    data_dict = load_mkt_data(inputs['dt'], replace=inputs['replace'], randomize=inputs['randomize'],
                              return_override=inputs['return_override'])

    assert data_dict['px'].shape == (124, 1172)
    assert data_dict['tri'].shape == (124, 1172)


def test_random_correct_sizes(inputs):
    data_dict = load_mkt_data(inputs['dt'], replace=inputs['replace'], randomize=True,
                              return_override=inputs['return_override'])

    assert data_dict['d_px'].shape == (124, 548)
    assert data_dict['div'].shape == (124, 548)


def test_random_ret_override_match_target(inputs):
    return_override = 0.01
    data_dict = load_mkt_data(inputs['dt'], replace=inputs['replace'], randomize=True, return_override=return_override)

    w = data_dict['w']
    px = data_dict['px']
    d_px = data_dict['d_px']
    div = data_dict['div']
    d_tri = data_dict['d_tri']

    data_freq = inputs['dt']

    idx_ret = (240 / data_freq) * (np.sum(w[0, :] * \
        np.product(1 + d_px.to_numpy()[1:, :], axis=0)) ** \
        (1 / (d_px.shape[0])) - 1)

    assert pt.approx(idx_ret, abs=1e-3) == return_override


def test_random_ret_fixed_weights(inputs):
        data_dict = load_mkt_data(inputs['dt'], replace=inputs['replace'], randomize=True)
        w = data_dict['w']
        assert w.shape == (124, 548)
        np.testing.assert_equal(w[0, :], w[1, :])
        assert pt.approx(w[0, :].sum(), 1e-10) == 1


def test_random_ret_fixed_shares_raises_val_error(inputs):
    try:
        data_dict = load_mkt_data(inputs['dt'], replace=inputs['replace'], randomize=True, fixed_weights=False)
        assert False
    except ValueError:
        assert True


def test_random_correct_stdev(inputs):
    return_override = 0.10

    data_dict = load_mkt_data(inputs['dt'], replace=inputs['replace'], randomize=True, return_override=return_override)

    # Calculate average standard deviation of stocks in the sample
    avg_std = data_dict['d_px'].std()
    data_freq = inputs['dt']
    avg_std_ann = avg_std * np.sqrt(240/data_freq)

    assert pt.approx(avg_std_ann.mean(), abs=1e-2) == 0.367
