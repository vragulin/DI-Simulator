"""
Testing random market data generation
"""
import inspect
import re
import numpy as np
import os
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
    working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
    PX_PICKLE = "idx_prices.pickle"
    TR_PICKLE = "idx_t_rets.pickle"
    W_PICKLE = "idx_daily_w.pickle"

    data_files = {
        'px': os.path.join(working_dir, PX_PICKLE),
        'tri': os.path.join(working_dir, TR_PICKLE),
        'w': os.path.join(working_dir, W_PICKLE)
    }

    dt = 60

    filter_params = {
        'max': 100,
        'min': 0.001
    }

    return data_files, dt, filter_params


def test_correct_sizes_no_filter(inputs):
    data_files, dt, filter_params = inputs

    # Test without randomization
    data_dict = load_mkt_data(data_files, dt)

    assert data_dict['px'].shape == (124, 548)
    assert data_dict['tri'].shape == (124, 548)


def test_correct_size_filter(inputs):
    data_files, dt, filter_params = inputs

    # Test without randomization
    data_dict = load_mkt_data(data_files, dt, filter_params=filter_params)

    assert data_dict['px'].shape == (124, 494)
    assert data_dict['tri'].shape == (124, 494)
    assert data_dict['w'].shape == (124, 494)


def test_fixed_weights(inputs):
    data_files, dt, filter_params = inputs

    # Test without randomization
    data_dict = load_mkt_data(data_files, dt, filter_params=filter_params,
                              fixed_weights=True)
    w = data_dict['w'].values
    assert w.shape == (124, 494)
    np.testing.assert_equal(w[0, :], w[1, :])
    assert pt.approx(w[0, :].sum(), 1e-10) == 1


def test_rand_weights_set_rand_seed(inputs):
    data_files, dt, filter_params = inputs
    rand_seed = 7

    # Test without randomization
    data_dict = load_mkt_data(data_files, dt, filter_params=filter_params,
                              rand_seed=rand_seed)

    assert data_dict['px'].shape == (124, 494)
    assert data_dict['tri'].shape == (124, 494)
    assert data_dict['w'].shape == (124, 494)

    w = data_dict['w'].values
    assert pt.approx(w[0, :].sum(), 1e-10) == 1

    exp_3w_0 = np.array([0.007366,  0.000853,  0.001404])
    np.testing.assert_array_almost_equal(w[0, :3], exp_3w_0, decimal=6)

    d_px = data_dict['d_px'].values
    w1 = w[0, :] * (1 + d_px[1, :])
    w1 /= w1.sum()
    np.testing.assert_array_almost_equal(w[1, :3], w1[:3], decimal=6)

