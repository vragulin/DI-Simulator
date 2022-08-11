""" Test process_input_data function """
import pytest as pt
from main import load_sim_settings_from_file, process_input_data
import index_math as im
import numpy as np


@pt.fixture
def out5():
    f = 'inputs/test5.xlsx'
    sim = load_sim_settings_from_file(f)
    out_dict = process_input_data(sim)
    return out_dict


def test_normalized_prices(out5):
    px = out5['px'].values
    assert px.shape == (6, 5)
    np.testing.assert_allclose(px[:, 0], [1, 0.9, 1, 1.1, 1.2, 1.3])


def test_price_moves(out5):
    d_px = out5['d_px'].values
    assert d_px.shape == (6, 5)
    np.testing.assert_allclose(d_px[:, 1],
                       [0, 0.05, 0.047619, -0.136364, 0.052632, 0.05],
                       atol=1e-6)


def test_tri(out5):
    tri = out5['tri'].values
    assert tri.shape == (6, 5)
    np.testing.assert_allclose(tri[:, 2],
                       [1, 1.035833, 1.071837, 0.907042, 1.010092, 1.046287],
                       atol=1e-6)



def test_d_tri(out5):
    d_tri = out5['d_tri'].values
    assert d_tri.shape == (6, 5)
    np.testing.assert_allclose(d_tri[:, 2],
                       [0, 0.035833, 0.034758, -0.15375, 0.113611, 0.035833],
                       atol=1e-6)

def test_div(out5):
    div = out5['div'].values
    assert div.shape == (6, 5)
    np.testing.assert_allclose(div[:, 2],
                       [0.0025, 0.002583, 0.002667, 0.00225, 0.0025, 0.002583],
                       atol=1e-6)

def test_w(out5):
    w = out5['w']
    assert w.shape == (6, 5)
    np.testing.assert_allclose(w[:, 2],
                       [0.25, 0.248916, 0.256246, 0.212565, 0.231911, 0.219702],
                       atol=1e-6)

def test_idx_div(out5):
    idx_div = out5['idx_div'].values
    np.testing.assert_allclose(idx_div[:, 0],
                       [0.002375, 0.002661, 0.002472, 0.002633, 0.002485, 0.002771],
                       atol=1e-6)


def test_idx_vals(out5):
    idx_vals = out5['idx_vals']
    np.testing.assert_allclose(idx_vals[:, 0],
                   [1, 1.037833333, 1.040666667, 1.0585, 1.078, 1.175833333],
                   atol=1e-6)


def test_idx_tot_rets(out5):
    idx_tri = out5['idx_tri']
    np.testing.assert_allclose(idx_tri[:, 0],
                   [1, 1.040397, 1.045709, 1.066229, 1.088329, 1.189665],
                   atol=1e-6)


# =============================
# Tests with return override
# =============================
@pt.fixture
def out5o():
    f = 'inputs/test5.xlsx'
    sim = load_sim_settings_from_file(f)
    out_dict = process_input_data(sim, return_override=0.1)
    return out_dict

def test_px_w_override(out5o):
    px = out5o['px'].values
    assert px.shape == (6, 5)
    np.testing.assert_allclose(px[:, 1],
                       [1, 1.041188, 1.081593, 0.924572, 0.965086, 1.004836],
                       atol=1e-6)


def test_d_px_w_override(out5o):
    d_px = out5o['d_px'].values
    assert d_px.shape == (6, 5)
    np.testing.assert_allclose(d_px[:, 1],
                       [0, 0.041188, 0.038807, -0.145176, 0.043819, 0.041188],
                       atol=1e-6)


def test_tri_w_override(out5o):
    tri = out5o['tri'].values
    assert tri.shape == (6, 5)
    np.testing.assert_allclose(tri[:, 2],
                       [1, 1.027021, 1.053668, 0.882382, 0.974854, 1.001196],
                       atol=1e-6)


def test_d_tri_w_override(out5o):
    d_tri = out5o['d_tri'].values
    assert d_tri.shape == (6, 5)
    np.testing.assert_allclose(d_tri[:, 2],
                       [0, 0.027021, 0.025946, -0.162562, 0.104799, 0.027021],
                       atol=1e-6)


def test_weights_w_override(out5o):
    w = out5o['w']
    assert w.shape == (6, 5)
    np.testing.assert_allclose(w[:, 2],
                       [0.25, 0.248907, 0.256327, 0.212262, 0.231768, 0.219471],
                       atol=1e-6)


def test_idx_div_w_override(out5o):
    idx_div = out5o['idx_div'].values
    np.testing.assert_allclose(idx_div[:, 0],
                       [0.002375, 0.00264, 0.002429, 0.002565, 0.002399, 0.002653],
                       atol=1e-6)


def test_idx_vals_w_override(out5o):
    idx_vals = out5o['idx_vals']
    np.testing.assert_allclose(idx_vals[:, 0],
                   [1, 1.029021, 1.022659, 1.031115, 1.04094, 1.126222],
                   atol=1e-6)


def test_tot_rets_w_override(out5o):
    idx_tri = out5o['idx_tri']
    np.testing.assert_allclose(idx_tri[:, 0],
                   [1, 1.031587, 1.027658, 1.038713, 1.051004, 1.139586],
                   atol=1e-6)
