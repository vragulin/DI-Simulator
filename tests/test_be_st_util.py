""" Test breakeven short-term tax utilization function
    V. Ragulin, started 30-Nov-22
"""

import numpy as np
import pytest
import pytest as pt

from be_st_util import be_st_util

@pt.fixture
def inputs() -> tuple:
    """
    Load input cash flows for the tests
    """

    n = 21

    # Pre-tax + non-taxable cash flows
    cf = np.zeros(n)
    cf[0] = -100.0
    cf[-1] = 387.0

    # Taxes
    st_gains = np.ones(n) * -2.0

    lt_gains = np.ones(n) * 3.0

    lt_gains[0] = st_gains[0] = 0.0
    lt_gains[-1] = cf[-1] + cf[0] - np.sum(st_gains) - np.sum(lt_gains[:-1])

    return cf, st_gains, lt_gains

def test_good_inputs(inputs):

    cf, st_gains, lt_gains = inputs
    tax_st, tax_lt = 0.5, 0.28
    bmk_irr_ann = 5.76243e-2
    bmk_irr = np.log(1+bmk_irr_ann)
    dt = 252

    bestu = be_st_util(cf, st_gains, lt_gains, tax_st, tax_lt, dt, bmk_irr)
    expected = 0.28145

    assert pytest.approx(bestu, rel=1e-3)


@pt.fixture
def inputs2() -> tuple:
    """
    Load input cash flows for the tests
    """

    n = 6

    # Pre-tax + non-taxable cash flows
    cf = np.zeros(n)
    cf[0] = -100.0
    cf[-1] = 107.0

    # Taxes
    st_gains = np.ones(n) * -1.0
    st_gains[0] = st_gains[-1] = 0

    lt_gains = -st_gains
    lt_gains[-1] = cf[-1] + cf[0] - np.sum(st_gains) - np.sum(lt_gains[:-1])

    return cf, st_gains, lt_gains


def test_quarterly_cf(inputs2):

    cf, st_gains, lt_gains = inputs2
    tax_st, tax_lt = 0.5, 0.28
    bmk_irr = 4.49102e-2
    dt = 60

    bestu = be_st_util(cf, st_gains, lt_gains, tax_st, tax_lt, dt, bmk_irr)
    expected = 0.5

    assert pytest.approx(bestu, rel=1e-3)
