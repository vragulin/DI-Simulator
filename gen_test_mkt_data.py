"""
Generate test market data in pickle format for testing
by V. Ragulin, statrted 6-Sep-2022
"""

import numpy as np
import pandas as pd
import datetime
import os
from typing import Optional
from pathlib import Path
from gen_random import gen_rand_returns


def gen_trading_dates(until: datetime, n: int) -> list:
    """ Return a list of n trading days ending on end_date
        I know it's ugly, but don't have time to learn dateutil.
        :param until: final date
        :param n: number of business days
        :return: list of days in the datetime.date format
    """
    count = 0
    dates = []
    current = until
    while count < n:
        if current.weekday() < 5:
            dates.append(current)
            count += 1
        current -= datetime.timedelta(days=1)

    return dates


def gen_test_mkt_data(files: dict, tgt_px_ret: float = 0.07, n_stocks: int = 10,
                      n_steps: int = 2400, dt: float = 1 / 252,
                      risk_free: float = 0.0, erp: float = 0.05,
                      sig_idx: float = 0.15, corr: float = 0.4,
                      dispersion: float = 0.18, div_yield: float = 0.02,
                      dates: Optional[list] = None) -> None:
    """ Generate and save pickle files to be used by the optimizer
    :param files: dictionary with paths of output files
    :param tgt_px_ret: target price return (excluding div) for the index
    :param n_stocks: number of stocks
    :param n_steps: number of time steps (i.e. #points = n_steps+1)
    :param dt: length of each step in years
    :param risk_free: risk-free rate (annual)
    :param erp: equity risk premium
    :param sig_idx: volatility (annual) of the sytematic component (~index vol)
    :params corr: pairwise correlation of stocks
    :param dispersion: - stock beta and residual vols are drawn from uniform in the range of [1-disp - 1+disp]
    :param div_yield: annual div yield, assume paid every period
    :param dates:  dates to use as index, if None - use numbers
    """

    # Define parameters
    prices0 = gen_rand_returns(n_stocks=n_stocks, n_steps=n_steps, dt=dt,
                               risk_free=risk_free, erp=erp, sig_idx=sig_idx,
                               corr=corr, dispersion=dispersion)

    w0 = np.ones(n_stocks) * (1 / n_stocks)
    idx_vals0 = prices0 @ w0

    idx_step_growth = (idx_vals0[-1] / idx_vals0[0]) ** (1 / n_steps)
    tgt_step_growth = (1 + tgt_px_ret) ** dt
    adj_factor = tgt_step_growth / idx_step_growth
    adj_vector = adj_factor ** np.arange(n_steps + 1)

    prices = prices0 * adj_vector.reshape(-1, 1)
    idx_vals = idx_vals0 * adj_vector

    # Print stats: average stock vol, index vol to make sure numbers are sensible
    rets = np.log(prices[1:, :] / prices[:-1, :])
    vol_stk = np.std(rets) * np.sqrt(1 / dt)

    idx_rets = np.log(idx_vals[1:] / idx_vals[:-1])
    vol_idx = np.std(idx_rets) * np.sqrt(1 / dt)

    idx_cagr = (idx_vals[-1] / idx_vals[0]) ** (1 / (n_steps * dt)) - 1
    print(f"Index Ret = {idx_cagr * 100:.2f}")
    print(f"Avg Stock Vol = {vol_stk * 100:.2f}, Idx Vol = {vol_idx * 100:.2f}")

    weights = prices * w0[None, :] / idx_vals[:, None]

    # Total return - for a start assume equal div uniformly paid
    t_rets = np.exp(rets) + (1 + div_yield) ** dt - 2
    tri = np.ones(prices.shape)
    tri[1:, :] = (1 + t_rets).cumprod(axis=0)

    idx_tri = np.ones(idx_vals.shape)
    idx_t_ret = np.exp(idx_rets) + (1 + div_yield) ** dt - 2
    idx_tri[1:] = (1 + idx_t_ret).cumprod(axis=0)

    tri_cagr = (idx_tri[-1] / idx_tri[0]) ** (1 / (n_steps * dt)) - 1
    print(f"Index Total Ret = {tri_cagr * 100:.2f}")

    # Define dates to be used as index
    if dates is None:
        idx = range(n_steps + 1)
    else:
        idx = dates

    # Pack outputs into dataframes
    n_pad = int(np.log10(n_stocks-1)) + 1
    tickers = sorted([f"S{str(int(i)).zfill(n_pad)}" for i in range(n_stocks)])
    df_prices = pd.DataFrame(prices, index=idx, columns=tickers)
    df_tri = pd.DataFrame(tri, index=idx, columns=tickers)
    df_weights = pd.DataFrame(weights, index=idx, columns=tickers)

    # Save into pickle files
    df_prices.to_pickle(files['prices'])
    df_tri.to_pickle(files['t_rets'])
    df_weights.to_pickle(files['daily_w'])

    return None


# Entry Point
if __name__ == "__main__":
    tgt_px_ret = 0.07
    n_stocks = 500
    n_steps = 1260
    dt = 1 / 252
    sig_idx = 0.16
    corr = 0.4
    dispersion = 0.18
    div_yield = 0.02
    end_date = datetime.datetime(2022, 9, 1)

    # File locations
    code = "5y_"
    dir_path = Path(f'data/test_data_{code}{n_stocks}')
    dir_path.mkdir(parents=True, exist_ok=True)

    # files = {'prices': os.path.join(dir_path, 'prices.pickle'),
    #          't_rets': os.path.join(dir_path, 't_rets.pickle'),
    #          'daily_w': os.path.join(dir_path, 'daily_w.pickle')}

    files = {'prices': dir_path / 'prices.pickle',
             't_rets': dir_path / 't_rets.pickle',
             'daily_w': dir_path / 'daily_w.pickle'}

    # Trading dates
    t_dates = sorted(gen_trading_dates(until=end_date, n=n_steps + 1))

    # Generate test market data
    gen_test_mkt_data(files=files, tgt_px_ret=tgt_px_ret, n_stocks=n_stocks, n_steps=n_steps,
                      dt=dt, sig_idx=sig_idx, corr=corr, dispersion=dispersion,
                      div_yield=div_yield, dates=t_dates)
    print('Done')
