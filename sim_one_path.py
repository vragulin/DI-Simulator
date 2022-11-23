"""
    Direct index / tax harvesting simulation.  Based on James's embaTestbed2022.py.
    by V. Ragulin, started 3-Aug-2022
"""
import datetime as dt
import logging
import os
import time
import warnings
from contextlib import suppress
from typing import Union

import dateutil.relativedelta as du
import numpy as np
import pandas as pd
import pytest
import scipy.optimize as sco

import config
import index_math as im
from gen_random import gen_rand_returns
from load_mkt_data import vectorize_dict
from obj_function import obj_func_w_lots, fetch_obj_params, tax_lots
from port_lots_class import PortLots
from pretty_print import df_to_format
from simulator_class import Simulator
from heur_3step import heur_3step_rebal

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:,.4f}'.format


def load_sim_settings_from_file(file: str, randomize=False) -> dict:
    """ Load simulation settings from an excel file """

    w_idx = pd.read_excel(file, sheet_name='idx_info', index_col='Ticker')
    w_idx.sort_index(inplace=True)

    stk_info = pd.read_excel(file, sheet_name='stock_info', index_col='Ticker')
    stk_info.sort_index(inplace=True)

    # Disable warnings related to data validation and read the parameters
    warnings.simplefilter(action='ignore', category=UserWarning)
    params = pd.read_excel(file, sheet_name='params', index_col='Name')

    # Ensure correct data types
    params.loc['donate', 'Value'] = float(params.loc['donate', 'Value'])
    params.loc['final_liquidation', 'Value'] = int(params.loc['final_liquidation', 'Value'])

    if not randomize:
        prices = pd.read_excel(file, sheet_name='prices', index_col='Ticker')
        prices = prices.sort_index().T
        n_steps = len(prices) - 1
        rng = None
    else:
        n_steps = int(params.loc['n_steps', 'Value'])
        # Initialize random number generator
        prices = None
        rng = np.random.default_rng(2022)

    # Array of rebalance dates (so that we finish arond today)
    rebal_dates = sorted([config.t_last + du.relativedelta(months=-3 * x) for x in range(n_steps + 1)])

    # Pack into a dictionary
    input_dict = {'w_idx': w_idx, 'stk_info': stk_info, 'prices': prices,
                  'dates': rebal_dates, 'params': params['Value'],
                  'randomize': randomize, 'rng': rng}

    return input_dict


def process_input_data_from_pickle(inputs: dict) -> dict:
    """ Process inputs into a format consistent with simulation
        Adjust returns of individual stocks to match target index (price) return
    """

    tickers = list(inputs['d_px'].columns)
    params = inputs['params']
    freq = int(config.ANN_FACTOR / params['dt'])

    # Prices and changes - normalized to start from 1.0
    d_px = inputs['d_px']
    d_tri = inputs['d_tri']
    w_arr = inputs['w_arr']

    px = (1 + d_px).cumprod()
    tri = (1 + d_tri).cumprod()
    if params['div_override'] < 0:
        div = d_tri - d_px
    else:
        div = d_px * 0
        div[(div.index % freq == 0) & (div.index > 0)] = params['div_override']

    if inputs['params']['benchmark_type'] == 'fixed_weights':
        idx_vals = (1 + (d_px * w_arr).sum(axis=1)).cumprod()
        idx_tri = (1 + (d_tri * w_arr).sum(axis=1)).cumprod()
        idx_div = (div * w_arr).sum(axis=1)
    else:  # Fixed share index - basket does not change
        idx_vals = im.index_vals(w_arr[0], px.values)
        idx_div = np.zeros(idx_vals.shape)
        idx_div[1:] = (div.values[1:] * px.values[:-1]) @ w_arr[0]

        idx_tri = im.total_ret_index(w_arr[0], px.values, idx_div,
                                     idx_vals=idx_vals)

    # Calculate portfolio review dates
    dates = (pd.to_datetime(list(inputs['dates'])).date)[inputs['dates_idx']]

    # Pack results into an output structure
    out_dict = {'tickers': tickers, 'px': px, 'd_px': d_px, 'tri': tri, 'd_tri': d_tri,
                'div': div.values, 'w': inputs['w'], 'idx_div': idx_div,
                'idx_vals': idx_vals, 'idx_tri': idx_tri,
                'dates': dates, 'tax': config.tax,
                'params': params, 'trx_cost': config.trx_cost}

    vectorize_dict(out_dict, ['px', 'd_px', 'tri', 'd_tri', 'div', 'w'])

    return out_dict


def process_input_data_from_xl(inputs: dict, return_override: float = None) -> dict:
    """ Process inputs into a format consistent with simulation
        Adjust returns of individual stocks to match target index (price) return
    """

    tickers = list(inputs['stk_info'].index)
    params = inputs['params']

    # Generate random price paths
    if inputs['randomize']:
        n_stocks = len(tickers)
        n_steps = int(params['n_steps'])
        dt_in_years = params['dt'] / config.ANN_FACTOR
        risk_free = params['risk_free']
        erp = params['eq_risk_prem']
        sig_idx = params['index_vol']
        stock_corr = params['stock_corr']
        dispersion = params['dispersion']  # set dispersion in stock betas and residual vols (same for both)
        rng = inputs['rng']

        px_array = gen_rand_returns(n_stocks=n_stocks, n_steps=n_steps, dt=dt_in_years, risk_free=risk_free,
                                    erp=erp, sig_idx=sig_idx, corr=stock_corr, dispersion=dispersion, rng=rng)

        prices = pd.DataFrame(px_array, index=range(n_steps + 1), columns=tickers)
    else:
        prices = inputs['prices']

    d_px = prices.pct_change().fillna(0).replace(np.Inf, 0)
    px = (1 + d_px).cumprod()

    dt = params['dt']
    w_start = inputs['w_idx'].values

    # If specified, get price returns approximately equal to the overrride
    if return_override is not None:
        idx_avg_ret = im.index_avg_return(w_start, px.values)
        d_px.iloc[1:, :] += (1 + return_override) ** (dt / config.ANN_FACTOR) - 1 - idx_avg_ret
        px = (1 + d_px).cumprod()

    # Calc dividend per period and convert into an array
    div_y_per_dt = inputs['stk_info']['Div Yield'] * dt / config.ANN_FACTOR
    div = np.zeros(px.shape)
    # div[1:] = px[:-1] * div_y_per_dt
    # idx_div = div @ w_start
    div[1:] = div_y_per_dt
    idx_div = np.zeros((px.shape[0], 1))
    idx_div[1:] = (div[1:] * px[:-1]) @ w_start

    # Calc total return
    d_tri = d_px + div_y_per_dt
    d_tri.iloc[0, :] = 0
    tri = (1 + d_tri).cumprod()

    w = im.index_weights_over_time(w_start, px.values)
    idx_vals = im.index_vals(w_start, px.values)
    idx_tri = im.total_ret_index(w_start, px.values, idx_div * (1 - config.tax['div']))

    # Pack results into an output structure
    out_dict = {'tickers': tickers, 'px': px, 'd_px': d_px, 'tri': tri, 'd_tri': d_tri,
                'div': div, 'w': w, 'idx_div': idx_div,
                'idx_vals': idx_vals, 'idx_tri': idx_tri,
                'dates': inputs['dates'], 'tax': config.tax,
                'params': inputs['params'], 'trx_cost': config.trx_cost}

    vectorize_dict(out_dict, ['px', 'd_px', 'tri', 'd_tri', 'div', 'w'])

    return out_dict


def init_optimizer(sim_data: dict) -> dict:
    """ Calculate fields needed for the optimizer """

    # Covariance matrix
    sim_data['cov_matrix'] = sim_data['d_px'].cov()

    return sim_data


def init_sim_hist(port: PortLots) -> list:
    """ Initialize the structure to accumulate simulation results """
    # TODO - move this into PortLots class, and likewise other similar functions

    # Trades sub-dictionary
    opt = {
        'opt_trades': port.df_stocks['shares'],
        'harvest': 0,
        'potl_harvest': 0,
        'harv_ratio': np.nan,
        'port_val': port.port_value
    }

    # Rebalance sub-dictionary
    rebal = {
        'tax': 0,
        'trx_cost': 0,
        'net_buy': 0
    }

    sim_hist = [{'opt': opt, 'div': 0, 'interest': 0,
                 'rebal': rebal, 'donate': 0, 'clock_reset_tax': 0}]

    return sim_hist


def log_port_report(port: PortLots, t: int, lots: bool = True, log=True) -> None:
    """ Print report.  If log=True send to logger, otherwise print
    """
    if log:
        out_func = logging.info
    else:
        out_func = print

    out_func(f"t = {t}")
    out_func(port.sim_report())

    if lots:
        out_func("\nLots details")
        out_func(port.df_lots[['ticker', 'start_date', 'shares', 'basis', 'price', '%_gain']])


def port_divs_during_period(port: PortLots, t: int, sim_data: dict) -> float:
    """ Calculate portfolio divs received during the period (assume that all stocks are ex-div on the last day """

    params = sim_data['params']
    n_stocks = sim_data['div'].shape[1]

    if True:  # params['div_override'] < 0:
        stock_divs = sim_data['div'][t, :]
        prices = sim_data['px'].values[t - 1, :]
    else:
        if (params['dt'] * t) % config.ANN_FACTOR == 0:
            stock_divs = params['div_override'] * np.ones(n_stocks)
        else:
            stock_divs = np.zeros(n_stocks)

        # Replicate JH code - define div yield as % of p(t), not p(t-1)
        prices = sim_data['px'].values[t, :]

    shares = port.df_stocks['shares'].values
    port_divs = (shares * prices) @ stock_divs

    return port_divs


def guess_trade_signs(guess: pd.Series, port: PortLots, t: float, sim_data: dict) -> pd.DataFrame:
    """ Guess trade signs for the optimization """

    # Start with something very simple
    # TODO - improve the logic.  Only need to assign signs for loss lots.
    #   Others can be kept as free variables

    df_lots = port.df_lots

    trade_signs = pd.DataFrame()
    trade_signs['guess'] = guess

    # Nominate 'to_sell' the same stocks as in the initial guess.
    trade_signs['sign'] = np.minimum(np.sign(trade_signs), 0)

    # Other stocks with loss lots nominate as 'to_buy'.  Leave remainign as unrestricted
    trade_signs['with_loss_lots'] = df_lots.groupby(by='ticker')['%_gain'].apply(lambda x: (x < 0).any())
    trade_signs['sign'] = np.where(trade_signs['with_loss_lots'] & (trade_signs['sign'] != -1),
                                   1, trade_signs['sign'])

    return trade_signs


def gen_bounds(guess: pd.Series, port: PortLots, t: int, sim_data: dict, max_harvest: float) -> tuple:
    """ Generate optimization bounds and an initial guess based on
        guessing the trade signs for the losses
    """

    df = port.df_stocks
    max_w_actv = sim_data['params']['max_active_wgt']

    # Set bounds on the basis max active weight and long-only constraint
    df['ub'] = (df['w_tgt'] + max_w_actv - df['w']) * port.port_value / df['price']

    # Instead of the lower 'active risk' bond, only apply a non-negativity constraint (i.e. no shorts)
    df['lb'] = -df['shares']

    # Check if there are 'no-buy / no-sell' limitations
    for b in ['lb', 'ub']:
        df[b] = np.where(df['no_sells'], df[b].apply(lambda x: max(x, 0)), df[b])
        df[b] = np.where(df['no_buys'], df[b].apply(lambda x: min(x, 0)), df[b])

    # Guess directions of trades and impose restrictions to make the problem convex
    trade_signs = guess_trade_signs(guess, port, t, sim_data)
    # print("\nGuessed Trade Signs:")
    # print(trade_signs.T)

    # Update bounds
    for b in ['lb', 'ub']:
        df[b] = np.where(trade_signs['sign'] > 0, df[b].apply(lambda x: max(x, 0)), df[b])
        df[b] = np.where(trade_signs['sign'] < 0, df[b].apply(lambda x: min(x, 0)), df[b])

    # Save trade_signs analysis
    port.trade_signs = trade_signs
    # print("\nGuessed signs:\n", trade_signs['sign'])

    # Set bounds for variables
    bounds = tuple(
        (df.loc[ticker, 'lb'], df.loc[ticker, 'ub']) for ticker in df.index
    )

    return bounds


def opt_rebalance(port: PortLots, t: int, sim_data: dict, max_harvest: float = 0.5,
                  log_file: str = None) -> dict:
    """ Harvesting with an optimizer.  Generate a starting point with a heuristic rebalance
    """
    # print('Running opt_rebalance')

    # Run the heursitic optimizer to generate the initial guess and a list of 'to_buy' and 'to_sell' stocks
    res_first_step = heuristic_rebalance(port, t, sim_data, max_harvest)

    # ----------------------------------------------------
    # Improve the initial guess through the optimization
    # ----------------------------------------------------
    # Collect parameters for the objective function
    inp = fetch_obj_params(port, t, sim_data, max_harvest)

    # Set up bounds and an initial guess
    guess0 = res_first_step['opt_trades']

    # Make sure we don't have even small neg positions because of floating point precision issues
    # This causes the optimizer to blow up sometimes
    guess = np.maximum(-port.df_stocks['shares'], guess0)

    # Check that we don't have shorts, and that rounding does not move the guess too much
    assert (port.df_stocks['shares'] + guess).min() >= 0, \
        f"Negative positions: {(port.df_stocks['shares'] + guess).min()}"

    try:

        assert np.abs(guess - guess0).max() < config.tol, "Error: move in guess over limit"
    except AssertionError:
        print("Something wrong here")

    bounds = gen_bounds(guess, port, t, sim_data, max_harvest)

    # Constraints
    # @njit
    def new_excess_cash(trades: np.array, price: np.array, trx_cost: float,
                        port_cash: float, port_value: float, min_w_cash: float) -> float:
        """ Calculate excess cash weight after the rebalance using numpy (no pandas)
        """
        trx_costs = price * np.abs(trades) * trx_cost
        new_cash = port_cash - price @ trades

        w_excess_cash = new_cash / (port_value - trx_costs.sum()) - min_w_cash
        return w_excess_cash + np.finfo(float).eps  # Give some flex for rounding error

    # Active beta of the new portfolio
    # @njit
    def new_active_beta(trades: np.array, shares: np.array, price: np.array, beta: np.array,
                        w_tgt: np.array, trx_cost: float, port_val: float) -> float:
        """ Calculate active beta (i.e. actual - benchmark) after the rebalance
            Fast numpy version
        """
        new_shrs = shares + trades
        trx_costs = np.abs(trades) * price * trx_cost
        new_port_val = port_val - trx_costs.sum()
        new_w = new_shrs * price / new_port_val
        new_w_actv = new_w - w_tgt

        new_beta_actv = new_w_actv @ beta
        return new_beta_actv

    # print("Checking if the guess is feasible and calc its obj_func:")
    # xcash = new_excess_cash(guess.values, inp["price"], inp["trx_cost"], inp["cash"], inp["port_val"],
    #                         inp["min_w_cash"])
    # # print(f"Excess cash = {xcash * 100:.3f}%")
    #
    # guessbeta = new_active_beta(guess.values, inp["shares"], inp["price"], inp["beta"],
    #                             inp["w_tgt"], inp["trx_cost"], inp["port_val"])
    # # print(f"Active beta = {guessbeta * 100:.3f}%")
    #
    obj_guess = obj_func_w_lots(guess.values,
                                inp['shares'], inp['price'], inp['w_tgt'], inp['xret'],
                                inp['trx_cost'], inp['port_val'], inp['alpha_horizon'], inp['cov_matrix'],
                                inp['max_sell'], inp['tax_per_shr'], inp['cum_shrs'],
                                inp['tkr_broadcast_idx'], inp['tkr_block_idx'],
                                inp['crra'])

    # print(f"Obj Func(guess) = {obj_guess:,.4f}")

    constraints = (
        # Minimum excess cash post-rebalance
        {'type': 'ineq', 'fun': new_excess_cash,
         'args': (inp['price'],
                  inp['trx_cost'],
                  inp['cash'],
                  inp['port_val'],
                  inp['min_w_cash'])},
        # Active beta post-rebalance
        {'type': 'eq', 'fun': new_active_beta,
         'args': (inp['shares'],
                  inp['price'],
                  inp['beta'],
                  inp['w_tgt'],
                  inp['trx_cost'],
                  inp['port_val'])})

    # Extra arguments for the objective function (other than x)
    args = (inp['shares'], inp['price'], inp['w_tgt'], inp['xret'],
            inp['trx_cost'], inp['port_val'], inp['alpha_horizon'], inp['cov_matrix'],
            inp['max_sell'], inp['tax_per_shr'], inp['cum_shrs'],
            inp['tkr_broadcast_idx'], inp['tkr_block_idx'],
            inp['crra'])

    # Callback to print results
    ret_sim = Simulator(obj_func_w_lots, log_file)

    # Run the optimization
    # noinspection PyTypeChecker
    max_adj_ret = sco.minimize(
        # Objective function
        # fun = risk_adj_after_tax_ret,
        fun=ret_sim.simulate,
        # Additional parameters
        args=args,
        # Initial guess
        x0=guess,
        method='SLSQP',
        bounds=bounds,
        # tol=1e-8,
        constraints=constraints,
        callback=ret_sim.callback,
        options={'maxiter': config.maxiter, 'disp': False}
    )

    # Close log file handle
    if log_file is not None:
        ret_sim.f_handle.close()

    # Unpack the output structure and prepare the output
    # Pack output into a dictionary
    # Note that the harvest field does not take into account the tax we paid on rebalancing trades
    # that is taken from the port.rebal_sim() method
    df_stocks = port.df_stocks
    df_lots = port.df_lots

    res = res_first_step.copy()

    # If optimizer did not improve, fall back to the heuristic guess
    if max_adj_ret['fun'] <= obj_guess:
        opt_trades = max_adj_ret['x']
    else:
        if (max_adj_ret['fun'] - obj_guess) > 1e-5:
            print("Warning: optimizer is worse than heuristic, using heuristic. "
                  f"(Opt: {max_adj_ret['fun']:.6f}, Guess: {obj_guess:.6f})")
        opt_trades = guess.values

    df_stocks['trade'] = opt_trades
    harvest = -tax_lots(opt_trades, inp['max_sell'], inp['tax_per_shr'], inp['cum_shrs'],
                        inp['tkr_broadcast_idx'], inp['tkr_block_idx'])

    res['opt_trades'] = df_stocks['trade']
    res['harvest'] = harvest

    res['harv_ratio'] = res['harvest'] / res['potl_harvest'] if res['potl_harvest'] > 0 else np.nan
    return res


def heuristic_rebalance(port: PortLots, t: int, sim_data: dict, max_harvest: float = 0.5) -> dict:
    """ Heuristic tax-harvesting
    """

    # We harvest lots by whether they are over/underweight and tax per dollar sold
    # Add columns with active weight for each stock

    # Process lots to speed up calculations
    t_date = sim_data['dates'][t]
    params = sim_data['params']
    port.process_lots(sim_data['tax'], t_date)

    df_lots = port.df_lots
    df_stocks = port.df_stocks

    # Add info on over/underweight stocks
    df_lots['w_actv'] = df_lots['ticker'].apply(lambda x: df_stocks.loc[x, 'w_actv'])
    # TODO - explore if this can be done quicker with numpy smart indexing via fast_lot_math

    df_lots['overweight'] = df_lots['w_actv'] > 0
    df_lots['mv'] = df_lots['price'] * df_lots['shares']

    # Calculate tax per dollar and total harvest potential
    harvest_thresh = params['harvest_thresh']
    df_lots['tax_per_$'] = df_lots['tax per shr'] / df_lots['price']
    df_lots['below_thresh'] = df_lots['%_gain'] <= harvest_thresh
    df_lots['potl_harv_val'] = np.maximum(-df_lots['tax_per_$'] * df_lots['mv'] * df_lots['below_thresh'], 0)

    # Identify a set of stocks from which we will be harvesting
    # Select stocks with the highest harvesting potential up to max_harvest
    df_stocks['potl_harv_val'] = df_lots.groupby(by='ticker')['potl_harv_val'].sum()
    # TODO - explore if this can be done quicker with numpy smart indexing via fast_lot_math
    df_stocks.sort_values('potl_harv_val', ascending=False, inplace=True)

    df_stocks['cum_mv_for_harv'] = df_stocks['mkt_vals'].cumsum(axis=0)

    max_harvest_val = max_harvest * port.port_value
    df_stocks['to_sell'] = (df_stocks['cum_mv_for_harv'] <= max_harvest_val) & (df_stocks['potl_harv_val'] > 0)
    df_lots['stk_to_sell'] = df_lots['ticker'].apply(lambda x: df_stocks.loc[x, 'to_sell'])
    # TODO - explore if this can be done quicker with numpy smart indexing via fast_lot_math
    df_lots['trade'] = -df_lots['shares'] * df_lots['stk_to_sell'] * (df_lots['potl_harv_val'] > 0)
    df_lots['tax'] = df_lots['stk_to_sell'] * df_lots['potl_harv_val']

    df_stocks['trade'] = df_lots.groupby(by='ticker')['trade'].sum()

    # Also sell stocks with w_actv > max_active_weight where we have long-term capital gains
    if ('max_active_wgt' in params) and (df_stocks['w_actv'] > params['max_active_wgt']).any():
        # Identify long-term lots for stocks where we have excess overweights
        df_lots['for_cutting_pos'] = (df_lots['w_actv'] > params['max_active_wgt']) \
                                     & (df_lots['long_term']) \
                                     & (~df_lots['stk_to_sell'])

        if df_lots['for_cutting_pos'].sum() > 0:
            df_lots['shrs_for_cutting'] = df_lots['shares'] * df_lots['for_cutting_pos']
            df_stocks['shrs_for_cutting'] = df_lots.groupby(by='ticker')['shrs_for_cutting'].sum()

            # Determine for each stock the number of shares we need to sell to get within the overweight band
            df_stocks['out_actv_band%'] = np.maximum(df_stocks['w_actv'] - params['max_active_wgt'], 0)
            df_stocks['out_actv_band_shrs'] = df_stocks['out_actv_band%'] * port.port_value / df_stocks['price']
            df_stocks['shrs_for_cut_w_cap'] = np.minimum(df_stocks['out_actv_band_shrs'], df_stocks['shrs_for_cutting'])

            # Add to the trade list
            df_stocks['to_sell'] = df_stocks['to_sell'] | (df_stocks['shrs_for_cut_w_cap'] > 0)
            df_stocks['trade'] -= df_stocks['shrs_for_cut_w_cap']

    # Make sure we don't sell more than what we have (due to machine precision issues etc.)
    df_stocks['trade'] = np.maximum(-df_stocks['shares'], df_stocks['trade'])

    # Now calculate the buys to offset beta impact of harvest sales and re-invest excess cash
    # For now assume that beta of all stocks is 1

    # mv_harvest = -df_lots['trade'] @ df_lots['price']
    mv_harvest = -df_stocks['trade'] @ df_stocks['price']
    tot_mv_to_buy = mv_harvest + port.cash - port.cash_w_tgt * port.port_value

    # Define priority of purchases, starting with underweight stocks
    df_stocks.sort_values(['to_sell', 'w_actv'], ascending=[True, True], inplace=True)

    # Buy underweights positions up to market value of the stocks that we have sold
    df_stocks['cum_buy_mv'] = (-df_stocks['w_actv']
                               * (1 - df_stocks['to_sell'])
                               * (df_stocks['w_actv'] < 0)
                               ).cumsum() * port.port_value

    # Check that we don't buy more than what we have sold
    df_stocks['cum_buy_mv'] = np.minimum(df_stocks['cum_buy_mv'], tot_mv_to_buy)
    df_stocks['buy_mv'] = df_stocks['cum_buy_mv'].diff()
    df_stocks.loc[df_stocks.index[0], 'buy_mv'] = df_stocks.loc[df_stocks.index[0], 'cum_buy_mv']  # fill first element

    # If we have not bought enough, spread the shortfall across all remaining 'to buy' stocks
    # TODO - make sure that stocks with zero positions are included into 'to buy' stocks
    shortfall = tot_mv_to_buy - df_stocks['buy_mv'].sum()
    if shortfall > 0:
        n_buys = (1 - df_stocks['to_sell']).sum()
        mv_remaining_per_stock = shortfall / n_buys
        df_stocks['buy_mv'] += np.where(df_stocks['to_sell'], 0, mv_remaining_per_stock)

    df_stocks['trade'] += df_stocks['buy_mv'] / df_stocks['price']

    # Clean up - sort stocks in alphabetical order
    df_stocks.sort_index(inplace=True)

    # Pack output into a dictionary
    # Note that the harvest field does not take into account the tax we paid on rebalancing trades
    # that is taken from the port.rebal_sim() method
    res = {'opt_trades': df_stocks['trade'], 'potl_harvest': df_stocks['potl_harv_val'].sum(),
           'harvest': df_lots['tax'].sum()}

    res['harv_ratio'] = res['harvest'] / res['potl_harvest'] if res['potl_harvest'] > 0 else np.nan
    res['port_val'] = port.port_value

    return res


def update_donate_old(port: PortLots, donate_thresh: float, donate_pct: float) -> float:
    """ Update portfolio for donations
        Assumes that port has been updated with market data
        Old version - only donate full lots, so the amount is not exact.
        If we have large lots, on some paths we are not donating anything - hence this
        has been replaced with a better approach that donates the exact amount.
    """
    df_lots = port.df_lots
    df_lots['mv'] = df_lots['shares'] * df_lots['price']
    donate_amount = port.port_value
    while (donate_amount / port.port_value) >= donate_pct:
        df_lots['donate_indic'] = df_lots['%_gain'] >= donate_thresh
        donate_amount = np.sum(df_lots['donate_indic'] * df_lots['mv'])
        donate_thresh += 0.02
    df_lots['basis'] = np.where(df_lots['donate_indic'], df_lots['price'], df_lots['basis'])
    # df_lots['start_date'] = np.where(df_lots['donate_indic'], port.t_date, df_lots['start_date'])
    df_lots['start_date'] = df_lots['start_date'].where(~df_lots['donate_indic'], port.t_date)
    # TODO - log what we have donated in a better way
    return donate_amount


def update_donate(port: PortLots, donate_thresh: float, donate_pct: float) -> float:
    """ Update portfolio for donations
        Assumes that port has been updated with market data
        New version - donate the correct amount.
    """
    df_lots = port.df_lots
    df_lots['mv'] = df_lots['shares'] * df_lots['price']

    # Sort lots in the order of %_gain
    df_lots.sort_values(by='%_gain', ascending=False, inplace=True)
    df_lots['cum_mv'] = (df_lots['mv'] * (df_lots['%_gain'] >= donate_thresh)).cumsum()
    donate_amount = port.port_value * donate_pct
    df_lots['cum_donated'] = np.minimum(df_lots['cum_mv'], donate_amount)
    df_lots['donated'] = df_lots['cum_donated'].diff()
    df_lots.loc[df_lots.index[0], 'donated'] = df_lots.loc[df_lots.index[0], 'cum_donated']

    # If there is a partially donated lot (there will be at most one), split it into donated
    # and remaining part
    partial_iloc = np.where(
        (df_lots['donated'] > 0) & (np.abs(df_lots['donated'] - df_lots['mv']) > np.finfo(float).eps))
    if (len(partial_iloc) > 0) and (len(partial_iloc[0] > 0)):
        if config.DEBUG_MODE:
            assert len(partial_iloc) == 1, f"Too many partial lots: {len(partial_iloc)}"

        # Split the partially donated lot into 2
        # New lot (shares remaining after donations)
        partial_idx = df_lots.index[partial_iloc][0]
        new_idx = max(df_lots.index) + 1
        new_lot = pd.DataFrame(df_lots.loc[partial_idx, :]).T
        new_lot.index = [new_idx]
        new_lot.loc[new_idx, 'mv'] = df_lots.loc[partial_idx, 'mv'] - df_lots.loc[partial_idx, 'donated']
        new_lot.loc[new_idx, 'shares'] = new_lot.loc[new_idx, 'mv'] / new_lot.loc[new_idx, 'price']
        new_lot.loc[new_idx, 'donated'] = 0

        # Old lot - shares for donation
        df_lots.loc[partial_idx, 'mv'] = df_lots.loc[partial_idx, 'donated']
        df_lots.loc[partial_idx, 'shares'] = df_lots.loc[partial_idx, 'mv'] / df_lots.loc[partial_idx, 'price']

        # Append new lot to the dataframe
        # df_lots = pd.concat([df_lots, new_lot])
        df_lots.loc[new_idx, :] = new_lot.loc[new_idx, :]

    # Update basis and start date on the donated lots
    df_lots['donate_indic'] = df_lots['donated'] > 0
    df_lots['basis'] = np.where(df_lots['donate_indic'], df_lots['price'], df_lots['basis'])
    # df_lots['start_date'] = np.where(df_lots['donate_indic'], port.t_date, df_lots['start_date'])
    df_lots['start_date'] = df_lots['start_date'].where(~df_lots['donate_indic'], port.t_date)

    # Check that we donated correct amount
    total_donated = df_lots['donated'].sum()
    if config.DEBUG_MODE:
        assert pytest.approx(total_donated, rel=config.tol) \
               == np.minimum(donate_amount, np.max(df_lots['cum_mv'])), \
            f"Donate amount doea not match: target={donate_amount}, actual={total_donated}"

    return total_donated


def gen_sim_report(sim_hist: list, sim_data: dict) -> tuple:
    """ Print simulationr results """

    params = sim_data['params']
    algorithm = params['algorithm']
    div_payout = params.get('div_payout', True)

    freq = int(config.ANN_FACTOR / params['dt'])
    n_steps = len(sim_hist) - 1

    rpt_cols = ['hvst_grs', 'hvst_net', 'hvst_potl', 'ratio_grs', 'ratio_net',
                'donate', 'div', 'tracking', 'port_val', 'bmk_val']
    df_report = pd.DataFrame(np.nan, columns=rpt_cols, index=range(n_steps + 1))
    df_report.index.name = 'step'

    for i in range(n_steps + 1):
        df_report.loc[i, 'hvst_grs'] = sim_hist[i]['opt']['harvest']

        if algorithm == 'inst_replace':
            df_report.loc[i, 'hvst_net'] = df_report.loc[i, 'hvst_grs']
        else:
            df_report.loc[i, 'hvst_net'] = -sim_hist[i]['rebal']['tax']
        df_report.loc[i, 'hvst_net'] -= sim_hist[i]['clock_reset_tax']

        df_report.loc[i, 'hvst_potl'] = sim_hist[i]['opt']['potl_harvest']
        df_report.loc[i, 'port_val'] = sim_hist[i]['opt']['port_val']
        df_report.loc[i, 'donate'] = sim_hist[i]['donate']
        df_report.loc[i, 'div'] = sim_hist[i]['div']

    # Rescale as % of portfolio value
    for col in ['hvst_grs', 'hvst_net', 'hvst_potl', 'donate', 'div']:
        df_report[col] /= df_report['port_val']

    # Compute ratios on an annual rolling basis for smoothness
    df_report['ratio_grs'] = df_report['hvst_grs'].rolling(freq).sum() / df_report['hvst_potl'].rolling(freq).sum()
    df_report['ratio_net'] = df_report['hvst_net'].rolling(freq).sum() / df_report['hvst_potl'].rolling(freq).sum()
    if div_payout:
        df_report['bmk_val'] = sim_data['idx_vals'] * df_report.loc[0, 'port_val']
    else:
        df_report['bmk_val'] = sim_data['idx_tri'] * df_report.loc[0, 'port_val']
    df_report['tracking'] = (df_report['port_val'] / df_report['bmk_val']).pct_change()

    # Generate other statistics that don't go into the df_report table
    # Liquidation taxes if applicable
    sim_stats = pd.Series(dtype='float64')
    if 'fin_liq_tax' in sim_hist[-1]:
        assert params['benchmark_type'] == "fixed_shares", \
            "Final liquidation only implemented for fixed shares indices."

        idx_liq_tax = im.index_liq_tax(
            sim_data['idx_vals'], sim_data['idx_div'], sim_data['idx_tri'],
            params['dt'], ann_factor=config.ANN_FACTOR,
            tax_lt=sim_data['tax']['lt'], tax_st=sim_data['tax']['st'],
            div_payout=div_payout) * params['final_liquidation']

        port_liq_tax = sim_hist[-1]['fin_liq_tax']
    else:
        idx_liq_tax = port_liq_tax = 0

    # IRR metrics
    years = n_steps / freq
    if not div_payout:
        # sim_stats['idx_irr'] = (((sim_data['idx_tri'][-1] - idx_liq_tax) /
        #                          sim_data['idx_tri'][0]) ** (1 / n_steps) - 1)[0] * freq
        sim_stats['idx_irr'] = np.log(((sim_data['idx_tri'][-1] - idx_liq_tax) /
                                       sim_data['idx_tri'][0]))[0] * freq / n_steps
    else:
        sim_stats['idx_irr'] = im.index_irr(
            params['dt'], sim_data['idx_vals'][0],
            sim_data['idx_vals'][-1] - idx_liq_tax,
            sim_data['idx_div'].ravel() * (1 - config.tax['div']),
            # ann_factor=config.ANN_FACTOR, freq=freq)
            ann_factor=config.ANN_FACTOR, freq=None)

        # Calculate portfolio after-tax IRR
    cf = df_report['hvst_net'].values.copy()
    cf += df_report['div'].values
    cf *= df_report['port_val'].values
    cf[0] -= df_report.loc[0, 'port_val']
    cf[-1] += df_report.loc[n_steps, 'port_val'] - port_liq_tax
    sim_stats['port_irr'] = im.irr_solve(cf, params['dt'], ann_factor=config.ANN_FACTOR,
                                         # bounds=(-0.05, 0.2), freq=freq)
                                         bounds=(-0.5, 0.5), freq=None)

    # Adjust to match EMBA calculation with 252 days for discounting
    sim_stats['idx_irr'] *= config.EMBA_IRR_Factor
    sim_stats['port_irr'] *= config.EMBA_IRR_Factor

    # Calculate aggregate statistics
    hvst_grs = df_report['hvst_grs'].sum()
    hvst_net = df_report['hvst_net'].sum()
    hvst_potl = df_report['hvst_potl'].sum()

    if hvst_potl > 0:
        ratio_grs = hvst_grs / hvst_potl
        ratio_net = hvst_net / hvst_potl
    else:
        ratio_grs = ratio_net = np.nan

    tracking_std = df_report['tracking'].dropna().std()
    if tracking_std >= config.tol:
        hvst_n_2_trk_ann = (hvst_net / years) / (tracking_std * np.sqrt(freq))
    else:
        hvst_n_2_trk_ann = np.nan

    # idx_ann_ret = ((df_report.iloc[-1]['bmk_val'] / df_report.iloc[0]['bmk_val']) ** (1 / n_steps) - 1) * freq
    # port_ann_ret = ((df_report.iloc[-1]['port_val'] / df_report.iloc[0]['port_val']) ** (1 / n_steps) - 1) * freq
    # sim_stats['port_pretax_ret'] = port_ann_ret
    # sim_stats['index_pretax_ret'] = idx_ann_ret

    # Pack results into a datframe:
    sim_stats['index_vol'] = df_report['bmk_val'].pct_change().std() * np.sqrt(freq)
    sim_stats['stk_vol_wavg'] = im.stock_vol_avg(sim_data['d_px'].values,
                                                 sim_data['w_arr'][0],
                                                 params['dt'] / config.LT_TAX_DAYS)
    sim_stats['tracking_std'] = tracking_std * np.sqrt(freq)
    sim_stats['hvst_grs'] = hvst_grs / years
    sim_stats['hvst_net'] = hvst_net / years
    sim_stats['hvst_potl'] = hvst_potl / years
    sim_stats['hvst_grs/potl'] = ratio_grs
    sim_stats['hvst_net/potl'] = ratio_net
    sim_stats['hvst_n/trckng'] = hvst_n_2_trk_ann
    sim_stats['tax_alpha'] = sim_stats['port_irr'] - sim_stats['idx_irr']

    if tracking_std >= config.tol:
        sim_stats['tax_sr'] = sim_stats['tax_alpha'] / sim_stats['tracking_std']
    else:
        sim_stats['tax_sr'] = np.nan

    sim_stats['idx_liq_tax%'] = idx_liq_tax / df_report.iloc[-1]['bmk_val']
    sim_stats['port_liq_tax%'] = port_liq_tax / df_report.iloc[-1]['port_val']

    # Re-arrange sim_stats fields in logical order
    new_order = ['idx_irr', 'port_irr', 'tax_alpha', 'tax_sr', 'index_vol',
                 'stk_vol_wavg', 'tracking_std', 'hvst_grs', 'hvst_net',
                 'hvst_potl', 'hvst_grs/potl', 'hvst_net/potl', 'hvst_n/trckng',
                 'idx_liq_tax%', 'port_liq_tax%']

    sim_stats = sim_stats.reindex(index=new_order)
    return df_report, sim_stats


def trade_history(sim_hist: list, sim_data: dict, shares_flag: bool = False) -> pd.DataFrame:
    """
    Generate a matrix of trade history over the simulation
    :param sim_hist: dictionary with simulation results
    :param sim_data: input data
    :param shares_flag: boolean - if True show trades in shares, else in market value (default)
    :return: pandas dataframe t x stocks with trades at each point in each stock
    """
    n_times = len(sim_hist)
    tickers = sim_hist[0]['opt']['opt_trades'].index
    df_trades = pd.DataFrame(0, index=range(n_times), columns=tickers)

    for i in range(n_times):
        if sim_data['params']['algorithm'] != 'inst_replace':
            df_trades.loc[i, :] = sim_hist[i]['opt']['opt_trades']
        elif i == 0:
            df_trades.loc[i, :] = sim_hist[i]['opt']['opt_trades']
        else:
            # If we are instantly replacing these are not 'real trades', but we report them anyway
            df_trades.loc[i, :] = sim_hist[i]['opt']['harvest_trades']

        if not shares_flag:
            df_trades.loc[i, :] *= sim_data['px'].loc[i, :]

    return df_trades


def pd_to_csv(df: Union[pd.DataFrame, pd.Series], file_code: str, suffix: str = None,
              dir_path: str = 'results/'):
    """ Write dataframe to a file, perform all checks """

    if suffix is not None:
        fname = dir_path + file_code + '_' + suffix + ".csv"
    else:
        fname = dir_path + file_code + ".csv"

    # Silently remove existing file with the same name
    with suppress(OSError):
        os.remove(fname)

    # Write
    df.to_csv(fname)


def run_sim(inputs: dict, suffix=None, dir_path: str = 'results/',
            verbose: bool = True, save_files: bool = True, log_file: str = None) -> tuple:
    # For timing
    timer_on = True
    if timer_on:
        t_in_opt = 0
        t_in_rebal = 0

    # Load simulation parameters from file
    params = inputs['params']

    try:
        n_steps = int(params['n_steps'])
    except (KeyError, TypeError):
        try:
            n_steps = len(inputs['d_px']) - 1
        except KeyError:
            n_steps = len(inputs['prices']) - 1

    # Process simulation params
    if params['ret_override_flag']:
        return_override = params['ret_override']
    else:
        return_override = None

    if 'donate' not in params:
        params['donate'] = 0

    # Depending on prices source, process the data differently
    # later, it may be a good idea to merge these functions into one.
    if inputs.get('prices_from_pickle', False):
        sim_data = process_input_data_from_pickle(inputs)
    else:
        sim_data = process_input_data_from_xl(inputs, return_override=return_override)

    algorithm = params.get('algorithm', 'heuristic')
    if algorithm == "optimizer":
        sim_data = init_optimizer(sim_data)

    # Set up the initial portfolio matching the index at t=0
    port = PortLots.init_portfolio_from_dict(sim_data)
    port.update_sim_data(sim_data=sim_data, t=0)

    log_port_report(port, 0)

    # Loop over periods, rebalance / harvest at each period, keep track of P&L
    # Simulation parameter - maximum amount of stocks that we harvest at each step
    max_harvest = params['max_harvest']

    # Initialize a list to keep simulation info
    sim_hist = init_sim_hist(port)
    for t in range(1, n_steps + 1):

        if verbose:
            if t % config.VERBOSE_FREQ == 0:
                print(f"Step = {t}", end=",\t")
                print(f"# lots: {len(port.df_lots)}", end=",\t")
                print(f"# stocks: {len(port.df_stocks)}")

        # Take account of the dividends that we have received during the period
        # Assume that all divs during the period happen on the last day (i.e. stocks are ex-div)
        div = port_divs_during_period(port, t, sim_data) * (1 - config.tax['div'])
        if not params.get('div_payout', True):
            port.cash += div
            div = 0

        # Revalue portfolio
        port.update_sim_data(sim_data=sim_data, t=t)
        logging.info("\nBefore rebalance:")
        log_port_report(port, t)

        if timer_on: tic = time.perf_counter()
        if algorithm == 'inst_replace':
            opt_res = port.harvest_inst_replace(t, sim_data)
        elif algorithm == 'optimizer':
            opt_res = opt_rebalance(port, t, sim_data, max_harvest, log_file=log_file)
        elif algorithm == 'heur_3step':
            opt_res = heur_3step_rebal(port, t, sim_data, max_harvest)
        else:
            opt_res = heuristic_rebalance(port, t, sim_data, max_harvest)

        if timer_on:
            toc = time.perf_counter()
            t_in_opt += toc - tic

        logging.info("Trades:")
        logging.info(opt_res['opt_trades'])
        if algorithm == 'inst_replace':
            logging.info("\nHarvest:")
            logging.info(opt_res['harvest_trades'])
        logging.info(f"\nHarvest={opt_res['harvest']:.4f}, "
                     f"Potl Harvest={opt_res['potl_harvest']:.4f}, "
                     f"Ratio={opt_res['harv_ratio']:.3f}")

        # Execute the rebalance (for now don't worry about donations
        if timer_on: tic = time.perf_counter()
        rebal_res = port.rebal_sim(trades=opt_res['opt_trades'], sim_data=sim_data, t=t)
        if timer_on:
            toc = time.perf_counter()
            t_in_rebal += toc - tic

        # Donate
        if params['donate'] and (t * params['dt'] % params['donate_freq'] == 0):
            donate_amount = update_donate(port, params['donate_thresh'], params['donate'])
        else:
            donate_amount = 0

        # Clock reset
        if params['clock_reset']:
            # Specify cut-off externally to match emba.
            # lt_cutoff = None
            # if config.USE_EMBA_LT_CUT_OFF:
            #     steps_per_year = int(252 / params['dt'])
            #     if t > steps_per_year:
            #         lt_cutoff = sim_data['dates'][t-steps_per_year-1]
            # clock_reset_tax = port.reset_clock(reset_thresh=params['reset_thresh'], lt_cutoff=lt_cutoff)
            clock_reset_tax = port.reset_clock(reset_thresh=params['reset_thresh'])
        else:
            clock_reset_tax = 0

        sim_hist.append({'opt': opt_res, 'div': div, 'rebal': rebal_res, 'donate': donate_amount,
                         'clock_reset_tax': clock_reset_tax})

        # print(f"Harvest: {opt_res['harvest']}, Reset: {clock_reset_tax}")

    # Liauidation tax at the end
    if params['final_liquidation'] > 0:
        port.update_sim_data(sim_data=sim_data, t=n_steps)
        port.process_lots(sim_data['tax'], port.t_date)
        sim_hist[-1]['fin_liq_tax'] = port.liquid_tax(liq_pct=params['final_liquidation'])

    logging.info("\nAfter final rebalance:")
    log_port_report(port, n_steps)

    logging.info("\nSimulation results:")
    df_report, sim_stats = gen_sim_report(sim_hist, sim_data)
    logging.info(df_report)

    # Generate and save trade history
    df_trades = trade_history(sim_hist, sim_data, shares_flag=False)
    if save_files:
        pd_to_csv(df_trades, "trades", suffix=suffix, dir_path=dir_path)

    logging.info("\nSimulation statistics (annualized):")
    logging.info(sim_stats.to_string())
    if save_files:
        pd_to_csv(sim_stats, "stats", suffix=suffix, dir_path=dir_path)

    if timer_on:
        print(f"Timer: in opt: {t_in_opt:.2f}, in rebal: {t_in_rebal:.2f} sec")

    return sim_stats, df_report


# ***********************************************************
# Entry point
# ***********************************************************
if __name__ == "__main__":
    timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = 'results/sim_' + timestamp + '.log'

    # Silently remove prior log_files and set up logger
    with suppress(OSError):
        os.remove(log_file)

    logging.basicConfig(level=logging.INFO, filename=log_file,
                        format='%(message)s')
    # Run simulation
    input_file = 'inputs/test5.xlsx'
    inputs = load_sim_settings_from_file(input_file, randomize=False)

    # tic = time.perf_counter()

    sim_stats, step_report = run_sim(inputs, suffix=timestamp)
    # toc = time.perf_counter()
    print("\nTrade Summary (x100, %):")
    # print(f"Simulation time = {toc-tic} seconds.")
    print(df_to_format(step_report * 100, formats={'_dflt': '{:.2f}'}))

    print("\nSimulation statistics (%):")
    print(df_to_format(pd.DataFrame(sim_stats * 100, columns=["annualized (%)"]), formats={'_dflt': '{:.2f}'}))
    print("\nDone")
