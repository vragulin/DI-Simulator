"""
Testing gen_random
"""
import numpy as np
import pytest as pt
from gen_random import gen_rand_returns

def test_2_by_1_case():

    n_stocks = 2
    n_steps = 1

    output = gen_rand_returns(n_stocks=n_stocks, n_steps=n_steps, dt=0.25)
    expected = [[1, 1,],
                [1.11341979,1.51123906]]
    np.testing.assert_allclose(output, expected, atol=1e-6)

def test_mean_std_large_sample():
    """ Testing that large sample properties of the distribution as as expected
        (i.e. sufficiently close to theoretical values).
    """
    n_stocks = 1000
    n_steps = 1000

    output = gen_rand_returns(n_stocks=n_stocks, n_steps=n_steps, dt=0.25, cum=False)

    # Check mean
    mean_ret = np.log(1+output).mean()
    exp_mean_ret = 0.018247980300370517
    assert pt.approx(mean_ret, 1e-10) == exp_mean_ret

    # Check std
    idx_std = np.log((1+output).mean(axis=1)).std()
    exp_std = 0.07216096578902917
    assert pt.approx(idx_std, 1e-10) == exp_std

    # Check correl
    avg_correl = np.corrcoef(np.log(1+output), rowvar=False).mean()
    exp_corr = 0.38084183430452856
    assert pt.approx(avg_correl, 1e-10) == exp_corr





