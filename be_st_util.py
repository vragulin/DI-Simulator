"""
Calculate Break-even Short-Term Utilization, defined as follows:
"""

import numpy as np
from typing import Optional


# Throw errors instead of warnings
np.seterr('raise')


def be_st_util(cf: np.array, st_gains: np.array, lt_gains: np.array,
               tax_st: float, tax_lt: float, dt: int,
               bmk_irr: float, ann_factor: float = 252) -> float:
    """ Assume we have a "baseline" strategy without harvesting (but that may possibly include other non-trivial features like dynamic weights, lot disposition logic, incidental taxes on portfolio rebalancing and final liquidation).  This strategy has an after-tax IRR = r_baseline.
    Against this baseline we are evaluating a "harvest" strategy with IRR = r_harvest,, and define its tax_alpha = r_harvest - r_baseline
    For the harvest strategy, assume we can only offset x% of our short-term losses against short-term gains, and likewise pay tax on x% of ST gains (this part is not very material since most of our strategies would not have a lot of ST gains, but it allows to ignore carryforward and netting rules)
    We  remaining ST losses are offset against LT gains (and get the benefit at LT tax rate)
    LT Gains and Losses are also taxed at the LT tax rate
    With this setup, we define BESTLU as the x% percentage of ST Losses utilized such that the harvest strategy has zero tax alpha vs. the baseline.

    :param cf: array of net cash flows that are not taxed at either LT or ST rates
                (initial investment, final value, net_divs)
    :param st_gains: gains that are taxed at ST rate
    :param lt_gains: gains that are taxed at LT rate
    :param tax_st: short-term capital gains tax rate
    :param tax_lt: long-term captial gains tax rate
    :param dt: lenth of a time step in trading adys
    :param bmk_irr: exponential annualized irr of a benchmark
    :param ann_factor:  number of trading days per annum
    :return: breakeven short-term tax utilization rate (ideally between 0 and 1, but in degenerate
             cases can be outside that range. The user can decide what to do about it.
    """

    # Flatten input arrays so that we can handle both column and row vectors
    cf_v = cf.ravel()
    st_gains_v = st_gains.ravel()
    lt_gains_v = lt_gains.ravel()

    # Check that vectors are the same size:
    n = len(cf_v)
    if (n != len(st_gains_v)) or (n != len(lt_gains_v)):
        f"""Error: cf and st_gains arrays have different length:
         cf: {n}, st_gains: {len(st_gains_v)}, lt_gains: {len(lt_gains_v)}."""

    # Calculate discount factors
    t = np.linspace(start=0, stop=dt*n, num=n, endpoint=False) / ann_factor
    disc_factors = np.exp(-bmk_irr * t)

    # Calculate after-tax CF at full utilization
    cf_full = cf_v - st_gains_v * tax_st - lt_gains_v * tax_lt
    npv_full = np.sum(cf_full * disc_factors)

    # Calculate after-tax CF at 100% utilization
    cf_zero = cf_v - (st_gains_v + lt_gains_v) * tax_lt
    npv_zero = np.sum(cf_zero * disc_factors)

    # Solve for BESTU (the breakeven short-term tax utilization)
    bestu = - npv_zero / (npv_full - npv_zero)

    return bestu


