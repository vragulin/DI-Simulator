import pytest as pt
import numpy as np

import rescale_stock_paths as rsp


@pt.fixture
def idx_data():
    px = np.array(
        [[1, 1.2, 0.9, 0.8, 0.95, 1.3],
         [1, 1, 1, 1, 1, 1],
         [1, 1.05, 0.9, 1.4, 1, 1.2]]
    ).T
    dp = px[1:, :] / px[:-1, :] - 1
    w = [0.45, 0.3, 0.25]
    return dp, w


def test_lin_model():
    y = np.array([3, 5, 7])
    x = np.array([1, 2, 3])
    a, b = rsp.lin_model(y,x)

    assert pt.approx(a, 1.0e-5) == 1.0
    assert pt.approx(b, 1.0e-5) == 2.0


def test_rescale_no_overrides_ret_px(idx_data):
    dp0, w0 = idx_data
    d_px = np.array(dp0)
    w = np.array(w0)

    idx_ann_ret = 0.09
    idx_ann_vol = 0.15
    stk_res_vol_factor = 2.0
    ann_factor = 2.0

    px_adj, idx_vals_adj = rsp.rescale_stocks_to_match_idx(
                            d_px, w,
                            idx_ann_ret=idx_ann_ret,
                            idx_ann_vol=idx_ann_vol,
                            stk_res_vol_factor=stk_res_vol_factor,
                            ann_factor=ann_factor,
                            output_rets=False)

    exp_px = np.array([
                    [1, 1, 1],
                    [1.250793, 1.00763, 0.947953],
                    [0.915609, 1.076817, 0.928407],
                    [0.590728, 1.083886, 1.845532],
                    [0.943229, 1.240574, 0.965957],
                    [1.348528, 1.23816, 1.048509]])
    exp_idx_vals = np.array(
                    [1, 1.102134, 0.967171, 1.052376, 1.038115, 1.240413])
    np.testing.assert_allclose(exp_px, px_adj, atol=1e-6)
    np.testing.assert_allclose(exp_idx_vals, idx_vals_adj, atol=1e-6)


def test_rescale_no_overrides_ret_ret(idx_data):
    dp0, w0 = idx_data
    d_px = np.array(dp0)
    w = np.array(w0)

    idx_ann_ret = 0.09
    idx_ann_vol = 0.15
    stk_res_vol_factor = 2.0
    ann_factor = 2.0

    d_px_adj, idx_rets_adj = rsp.rescale_stocks_to_match_idx(
        d_px, w,
        idx_ann_ret=idx_ann_ret,
        idx_ann_vol=idx_ann_vol,
        stk_res_vol_factor=stk_res_vol_factor,
        ann_factor=ann_factor)

    exp_d_px = np.array([
                [0.250793, 0.00763, -0.052047],
                [-0.267977, 0.068663, -0.020619],
                [-0.354825, 0.006565, 0.987847],
                [0.596723, 0.144561, -0.476597],
                [0.429693, -0.001946, 0.085461]])

    exp_idx_rets = np.array(
                    [0.102134, -0.122456, 0.088098, -0.013552, 0.194871])

    np.testing.assert_allclose(exp_d_px, d_px_adj, atol=1e-6)
    np.testing.assert_allclose(exp_idx_rets, idx_rets_adj, atol=1e-6)


def test_rescale_no_resid_scaling(idx_data):
    dp0, w0 = idx_data
    d_px = np.array(dp0)
    w = np.array(w0)

    idx_ann_ret = 0.09
    idx_ann_vol = 0.15
    ann_factor = 2.0

    d_px_adj, idx_rets_adj = rsp.rescale_stocks_to_match_idx(
        d_px, w,
        idx_ann_ret=idx_ann_ret,
        idx_ann_vol=idx_ann_vol,
        ann_factor=ann_factor)

    exp_d_px = np.array([
        [0.199602, -0.000332, 0.049651],
        [-0.219764, 0.040314, -0.108302],
        [-0.109412, 0.001912, 0.558529],
        [0.210354, 0.019246, -0.271967],
        [0.348774, -0.014358, 0.182771]])

    exp_idx_rets = np.array(
                    [0.102134, -0.122456, 0.088098, -0.013552, 0.194871])

    np.testing.assert_allclose(exp_d_px, d_px_adj, atol=1e-6)
    np.testing.assert_allclose(exp_idx_rets, idx_rets_adj, atol=1e-6)


def test_rescale_no_vol_scaling(idx_data):
    dp0, w0 = idx_data
    d_px = np.array(dp0)
    w = np.array(w0)

    idx_ann_ret = 0.09
    stk_res_vol_factor = 2.0
    ann_factor = 2.0

    d_px_adj, idx_rets_adj = rsp.rescale_stocks_to_match_idx(
        d_px, w,
        idx_ann_ret=idx_ann_ret,
        stk_res_vol_factor=stk_res_vol_factor,
        ann_factor=ann_factor)

    exp_d_px = np.array([
                    [0.262696915, 0.017220172, -0.043025182],
                    [-0.289883282, 0.036682581, -0.049927037],
                    [-0.350142926, 0.013869018, 1.002271842],
                    [0.580957817, 0.133260717, -0.481764515],
                    [0.463837729, 0.021890797, 0.111384427]])

    exp_idx_rets = np.array(
                    [0.112623368,-0.148717061, 0.095993565,
                     -0.023291491, 0.223407569])

    np.testing.assert_allclose(exp_d_px, d_px_adj, atol=1e-6)
    np.testing.assert_allclose(exp_idx_rets, idx_rets_adj, atol=1e-6)


def test_rescale_no_level_scaling(idx_data):

    dp0, w0 = idx_data
    d_px = np.array(dp0)
    w = np.array(w0)

    idx_ann_vol = 0.15
    stk_res_vol_factor = 2.0
    ann_factor = 2.0

    d_px_adj, idx_rets_adj = rsp.rescale_stocks_to_match_idx(
        d_px, w,
        idx_ann_vol=idx_ann_vol,
        stk_res_vol_factor=stk_res_vol_factor,
        ann_factor=ann_factor)

    exp_d_px = np.array([
                    [0.239412, -0.001538, -0.060672],
                    [-0.274638,	0.058939, -0.02953],
                    [-0.360695,	-0.002594, 0.96976],
                    [0.582195, 0.134147, -0.481359],
                    [0.416684, -0.011026, 0.075584]])

    exp_idx_rets = np.array(
                    [0.092106115, -0.130440556, 0.078197304,
                     -0.022527252, 0.183999])

    np.testing.assert_allclose(exp_d_px, d_px_adj, atol=1e-6)
    np.testing.assert_allclose(exp_idx_rets, idx_rets_adj, atol=1e-6)


def test_calc_path_stats_w_defaults(idx_data):

    # Generate data
    dp0, w0 = idx_data
    d_px = np.array(dp0)
    w = np.array(w0)

    idx_ann_vol = 0.15
    stk_res_vol_factor = 2.0
    ann_factor = 2.0

    d_px_adj, idx_rets_adj = rsp.rescale_stocks_to_match_idx(
        d_px, w,
        idx_ann_vol=idx_ann_vol,
        stk_res_vol_factor=stk_res_vol_factor,
        ann_factor=ann_factor)

    # Calculate moments of the data
    idx_ret, idx_vol, resid_vol = rsp.calc_path_stats(1, d_px_adj, w,
                                        vol_fixed_w=False, ann_factor=ann_factor)
    assert pt.approx(idx_ret, abs=1e-6) == 0.069063
    assert pt.approx(idx_vol, abs=1e-6) == 0.15


def test_calc_path_stats_w_res_vol(idx_data):
    # Generate data
    dp0, w0 = idx_data
    d_px = np.array(dp0)
    w = np.array(w0)

    idx_ann_vol = 0.15
    stk_res_vol_factor = 2.0
    ann_factor = 2.0

    d_px_adj, idx_rets_adj = rsp.rescale_stocks_to_match_idx(
        d_px, w,
        idx_ann_vol=idx_ann_vol,
        stk_res_vol_factor=stk_res_vol_factor,
        ann_factor=ann_factor)

    # Calculate moments of the data
    idx_ret, idx_vol, resid_vol = rsp.calc_path_stats(1, d_px_adj, w,
                                        vol_fixed_w=False,
                                        calc_res_vol=True,
                                        ann_factor=ann_factor)

    assert pt.approx(idx_ret, abs=1e-6) == 0.069063
    assert pt.approx(idx_vol, abs=1e-6) == 0.15
    assert pt.approx(resid_vol, abs=1e-6) == 0.4016428


def test_calc_path_stats_vol_fixed_shr(idx_data):
    # Generate data
    dp0, w0 = idx_data
    d_px = np.array(dp0)
    w = np.array(w0)

    idx_ann_vol = 0.15
    stk_res_vol_factor = 2.0
    ann_factor = 2.0

    d_px_adj, idx_rets_adj = rsp.rescale_stocks_to_match_idx(
        d_px, w,
        idx_ann_vol=idx_ann_vol,
        stk_res_vol_factor=stk_res_vol_factor,
        ann_factor=ann_factor)

    # Calculate moments of the data
    idx_ret, idx_vol, resid_vol = rsp.calc_path_stats(1, d_px_adj, w,
                                        vol_fixed_w=True,
                                        calc_res_vol=True,
                                        ann_factor=ann_factor)

    assert pt.approx(idx_ret, abs=1e-6) == 0.069063
    assert pt.approx(idx_vol, abs=1e-6) == 0.153314

