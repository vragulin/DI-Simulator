"""
     Generate random price paths for direct indexing simulations
     by V. Ragulin, started 12-Aug-2022
"""
import numpy as np
from numpy import random


def gen_rand_returns(n_stocks: int = 10, n_steps: int = 10, dt: float = 1, risk_free: float = 0.02,
                       erp: float = 0.06, sig_idx: float = 0.15, corr: float = 0.4,
                       dispersion: float = 0.2, cum: bool = True,
                       rng: np.random._generator.Generator = None) -> np.array:
    """

    :param n_stocks: - number of stocks
    :param n_steps:  - number of simulation steps
    :param dt: - length of 1 step (used to scale mu and sigma)
    :param risk_free: - risk-free rate per period
    :param erp: - equity risk premium
    :param sig_idx: - volatility of the beta factor per period
    :param corr: - implied correlation (used to calc vol of the residuals)
    :param dispersion: - stock beta and residual vols are drawn from uniform in the range of [1-disp - 1+disp]
    :param cum: - show cumulative return indices (if True), else show returns
    :param rng: - random number generator object
    :return: np array of period returns n_stocks x n_steps, or of cumulative returns (n_stocks x (n_steps+1))
                starting at 1
    """

    # Initialize random number generator if not given
    if rng is None:
        rng = np.random.default_rng(2022)

    ind_xs_rets = rng.normal(erp * dt, sig_idx * np.sqrt(dt), n_steps)

    # Residual random walk changes (dz)
    res_dz = rng.normal(0, 1, (n_steps, n_stocks))

    # Vector of betas, ensure that they average to 1
    assert 1.0 >= dispersion >= 0, "dispersion must be between 0 and 1"
    if dispersion > 0:
        betas = rng.uniform(1 - dispersion, 1 + dispersion, n_stocks)
        betas += (1 - np.mean(betas))
    else:
        betas = 1

    # Stock residual standard deviations
    assert 0 < corr <= 1, "corr must be in (0 and 1]"
    base_resid_vol = sig_idx * np.sqrt((1 - corr) / corr)

    if dispersion > 0:
        stock_vol_mults = rng.uniform(1 - dispersion, 1 + dispersion, n_stocks)
        stock_vol_mults += (1 - np.mean(stock_vol_mults))
    else:
        stock_vol_mults = 1

    stock_res_vols = base_resid_vol * stock_vol_mults

    # Log returns
    log_rets = risk_free * dt + ind_xs_rets[:, None] * betas[None, :] \
               + res_dz * stock_res_vols * np.sqrt(dt)

    # If requested, generate cumulative returns
    if cum:
        cum_log_rets = np.zeros((n_steps + 1, n_stocks))
        cum_log_rets[1:] = np.cumsum(log_rets, axis=0)
        return np.exp(cum_log_rets)
    else:
        return np.exp(log_rets) - 1
