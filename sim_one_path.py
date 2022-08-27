"""
    Direct index / tax harvesting simulation.  Based on James's embaTestbed2022.py.
    by V. Ragulin, started 3-Aug-2022
"""
import os
from contextlib import suppress
import numpy as np
import datetime as dt
import dateutil.relativedelta as du
import logging
from typing import Union
import pandas as pd

pd.options.display.float_format = '{:,.4f}'.format

import config
import index_math as im
from port_lots_class import PortLots
from gen_random import gen_rand_returns
from pretty_print import df_to_format


# Directory with on-period optimizer codebase
# sys.path.append('../DirectIndexing/src/optimizer/opt_with_lots')

def load_sim_settings_from_file(file: str, randomize=False) -> dict:
    """ Load simulation settings from an excel file """

    w_idx = pd.read_excel(file, sheet_name='idx_info', index_col='Ticker')
    w_idx.sort_index(inplace=True)

    stk_info = pd.read_excel(file, sheet_name='stock_info', index_col='Ticker')
    stk_info.sort_index(inplace=True)

    params = pd.read_excel(file, sheet_name='params', index_col='Name')

    if not randomize:
        prices = pd.read_excel(file, sheet_name='prices', index_col='Ticker')
        prices = prices.sort_index().T
        n_steps = len(prices.T) - 1
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


def process_input_data_from_pickle(inputs: dict, return_override: float = None) -> dict:
    """ Process inputs into a format consistent with simulation
        Adjust returns of individual stocks to match target index (price) return
    """

    tickers = list(inputs['d_px'].columns)
    n_steps = len(inputs['d_px'].index)
    params = inputs['params']

    # Prices and changes - normalized to start from 1.0
    d_px = inputs['d_px']
    d_tri = inputs['d_tri']
    w = inputs['w_idx']

    px = (1 + d_px).cumprod()
    tri = (1 + d_tri).cumprod()
    div = d_tri - d_px

    if inputs['mkt_data']['fixed_weights']:
        idx_vals = (1 + (d_px * w).sum(axis=1)).cumprod()
        idx_tri = (1 + (d_tri * w).sum(axis=1)).cumprod()
        idx_div = (div * w).sum(axis=1)
    else:
        w_start = w.values[0, :]
        idx_vals = im.index_vals(w_start, px.values)
        idx_div = np.zeros(idx_vals.shape)
        idx_div[1:] = (div.values[1:] * px.values[:-1]) @ w_start
        idx_tri = im.total_ret_index(w_start, px.values, idx_div,
                                     idx_vals=idx_vals)

    # Pack results into an output structure
    out_dict = {'tickers': tickers, 'px': px, 'd_px': d_px, 'tri': tri, 'd_tri': d_tri,
                'div': div, 'w': w.values, 'idx_div': idx_div,
                'idx_vals': idx_vals, 'idx_tri': idx_tri,
                'dates': inputs['dates'], 'tax': config.tax,
                'params': inputs['params'], 'trx_cost': config.trx_cost}

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
    div = px * div_y_per_dt
    idx_div = div @ w_start

    # Calculate donation percent per period (not using since we donate annually)
    # inputs['params']['donate_pct_per_period'] = inputs['params']['donate_pct'] * dt / config.ANN_FACTOR

    # Calc total return
    d_tri = d_px + div_y_per_dt
    d_tri.iloc[0, :] = 0
    tri = (1 + d_tri).cumprod()

    w = im.index_weights_over_time(w_start, px.values)
    idx_vals = im.index_vals(w_start, px.values)
    idx_tri = im.total_ret_index(w_start, px.values, idx_div.values)

    # Pack results into an output structure
    out_dict = {'tickers': tickers, 'px': px, 'd_px': d_px, 'tri': tri, 'd_tri': d_tri,
                'div': div, 'w': w, 'idx_div': idx_div,
                'idx_vals': idx_vals, 'idx_tri': idx_tri,
                'dates': inputs['dates'], 'tax': config.tax,
                'params': inputs['params'], 'trx_cost': config.trx_cost}

    return out_dict


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

    sim_hist = [{'opt': opt, 'rebal': rebal, 'donate': 0, 'clock_reset_tax': 0}]

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


def port_divs_during_period(port: PortLots, t: int, sim_data: dict) -> None:
    """ Calculate portfolio divs received during the period (assume that all stocks are ex-div on the last day """

    stock_divs = sim_data['div'].values[t, :]
    shares = port.df_stocks['shares'].values
    prices = sim_data['px'].values[t - 1, :]
    port_divs = (shares * prices) @ stock_divs

    return port_divs


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
    if ('max_actv_wgt' in params) and (df_stocks['w_actv'] > params['max_actv_wgt']).any():
        # Identify long-term lots for stocks where we have excess overweights
        df_lots['for_cutting_pos'] = (df_lots['w_actv'] > params['max_actv_wgt']) \
                                & (df_lots['long_term']) \
                                & (~df_lots['stk_to_sell'])

        if df_lots['for_cutting_pos'].sum() > 0:
            df_lots['shrs_for_cutting'] = df_lots['shares'] * df_lots['for_cutting_pos']
            df_stocks['shrs_for_cutting'] = df_lots.groupby(by='ticker')['shrs_for_cutting'].sum()

            # Determine for each stock the number of shares we need to sell to get within the overweight band
            df_stocks['out_actv_band%'] = np.maximum(df_stocks['w_actv'] - params['max_actv_wgt'], 0)
            df_stocks['out_actv_band_shrs'] = df_stocks['out_actv_band%'] * port.port_value / df_stocks['price']
            df_stocks['shrs_for_cut_w_cap'] = np.minimum(df_stocks['out_actv_band_shrs'], df_stocks['shrs_for_cutting'])

            # Add to the trade list
            df_stocks['to_sell'] = df_stocks['to_sell'] | (df_stocks['shrs_for_cut_w_cap'] > 0)
            df_stocks['trade'] -= df_stocks['shrs_for_cut_w_cap']

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


def update_donate(port: PortLots, donate_thresh: float, donate_pct: float) -> float:
    """ Update portfolio for donations
        Assumes that port has been updated with market data
    """
    df_lots = port.df_lots
    df_lots['mv'] = df_lots['shares'] * df_lots['price']
    donate_amount = port.port_value
    while (donate_amount / port.port_value) >= donate_pct:
        df_lots['donate_indic'] = df_lots['%_gain'] >= donate_thresh
        donate_amount = np.sum(df_lots['donate_indic'] * df_lots['mv'])
        donate_thresh += 0.02
    df_lots['basis'] = np.where(df_lots['donate_indic'], df_lots['price'], df_lots['basis'])
    df_lots['start_date'] = np.where(df_lots['donate_indic'], port.t_date, df_lots['start_date'])
    # TODO - log what we have donated in a better way
    return donate_amount


def gen_sim_report(sim_hist: list, sim_data: dict) -> pd.DataFrame:
    """ Print simulationr results """
    rpt_cols = ['hvst_grs', 'hvst_net', 'hvst_potl', 'ratio_grs', 'ratio_net',
                'donate', 'tracking', 'port_val', 'bmk_val']
    df_report = pd.DataFrame(np.nan, columns=rpt_cols, index=range(len(sim_hist)))
    df_report.index.name = 'step'

    for i in range(len(df_report)):
        df_report.loc[i, 'hvst_grs'] = sim_hist[i]['opt']['harvest']
        df_report.loc[i, 'hvst_net'] = -sim_hist[i]['rebal']['tax'] - sim_hist[i]['clock_reset_tax']
        df_report.loc[i, 'hvst_potl'] = sim_hist[i]['opt']['potl_harvest']
        # df_report.loc[i, 'net buy'] = sim_hist[i]['rebal']['net_buy']
        df_report.loc[i, 'port_val'] = sim_hist[i]['opt']['port_val']
        df_report.loc[i, 'donate'] = sim_hist[i]['donate']

    # Rescale as % of portfolio value
    for col in ['hvst_grs', 'hvst_net', 'hvst_potl', 'donate']:
        df_report[col] /= df_report['port_val']

    # Compute ratios on an annual rolling basis for smoothness
    freq = int(config.ANN_FACTOR / sim_data['params']['dt'])
    df_report['ratio_grs'] = df_report['hvst_grs'].rolling(freq).sum() / df_report['hvst_potl'].rolling(freq).sum()
    df_report['ratio_net'] = df_report['hvst_net'].rolling(freq).sum() / df_report['hvst_potl'].rolling(freq).sum()
    df_report['bmk_val'] = sim_data['idx_tri'] * df_report.loc[0, 'port_val']
    df_report['tracking'] = (df_report['port_val'] / df_report['bmk_val']).pct_change()

    return df_report


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
        df_trades.loc[i, :] = sim_hist[i]['opt']['opt_trades']

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
            verbose: bool = True, save_files: bool = True) -> tuple:

    # Load simulation parameters from file
    params = inputs['params']
    n_steps = len(inputs['d_px']) - 1

    # Process simulation params
    if params['ret_override_flag']:
        return_override = params['ret_override']
    else:
        return_override = None

    if params['donate_pct'] <= 0:
        params['donate'] = False

    # Depending on prices source, process the data differntly
    # later, it may be a good idea to merge these functions into one.
    if inputs.get('prices_from_pickle', False):
        sim_data = process_input_data_from_pickle(inputs, return_override=return_override)
    else:
        sim_data = process_input_data_from_xl(inputs, return_override=return_override)

    # Set up the initial portfolio matching t
    # he index at t=0
    port = PortLots.init_portfolio_from_dict(sim_data)
    port.update_sim_data(sim_data=sim_data, t=0)

    log_port_report(port, 0)

    # Loop over periods, rebalance / harvest at each period, keep track of P&L
    # Simulation parameter - maximum amount of stocks that we harvest at each step
    max_harvest = params['max_harvest']

    # Initialize the structure to keep simulation info
    sim_hist = init_sim_hist(port)
    for t in range(1, n_steps + 1):

        if verbose:
            if t % 25 == 0: print(f"Step = {t}")

        # Take account of the dividends that we have received during the period
        # Assume that all divs during the period happen on the last day (i.e. stocks are ex-div)
        port.cash += port_divs_during_period(port, t, sim_data)

        # Revalue portfolio
        port.update_sim_data(sim_data=sim_data, t=t)
        logging.info("\nBefore rebalance:")
        log_port_report(port, t)

        # Work out the optimal rebalance (for now use heuristic)
        opt_res = heuristic_rebalance(port, t, sim_data, max_harvest)
        logging.info("Trades:")
        logging.info(opt_res['opt_trades'])
        logging.info(f"\nHarvest={opt_res['harvest']:.4f}, "
                     f"Potl Harvest={opt_res['potl_harvest']:.4f}, "
                     f"Ratio={opt_res['harv_ratio']:.3f}")

        # Execute the rebalance (for now don't worry about donations
        rebal_res = port.rebal_sim(trades=opt_res['opt_trades'], sim_data=sim_data, t=t)

        # Donate
        if params['donate'] and (t * params['dt'] % params['donate_freq'] == 0):
            donate_amount = update_donate(port, params['donate_thresh'], params['donate_pct'])
        else:
            donate_amount = 0

        # Clock reset
        if params['clock_reset']:
            clock_reset_tax = port.reset_clock(reset_thresh=params['reset_thresh'])
        else:
            clock_reset_tax = 0

        sim_hist.append({'opt': opt_res, 'rebal': rebal_res, 'donate': donate_amount,
                         'clock_reset_tax': clock_reset_tax})

    logging.info("\nAfter final rebalance:")
    log_port_report(port, t)

    logging.info("\nSimulation results:")
    df_report = gen_sim_report(sim_hist, sim_data)
    logging.info(df_report)

    # Generate and save trade history
    df_trades = trade_history(sim_hist, sim_data, shares_flag=False)
    if save_files:
        pd_to_csv(df_trades, "trades", suffix=suffix, dir_path=dir_path)

    # Print total statistics
    hvst_grs = df_report['hvst_grs'].sum()
    hvst_net = df_report['hvst_net'].sum()
    hvst_potl = df_report['hvst_potl'].sum()
    # TODO - think of a more sensible measure of harvest / pot'l harvest, like an IRR or PME
    #     or at least scale it by portfolio value and represent in bp

    ratio_grs = hvst_grs / hvst_potl
    ratio_net = hvst_net / hvst_potl
    freq = config.ANN_FACTOR / params['dt']
    years = n_steps / freq
    tracking_std = df_report['tracking'].dropna().std()
    hvst_n_2_trk_ann = (hvst_net / years) / (tracking_std * np.sqrt(freq))
    idx_ann_ret = ((df_report.iloc[-1]['bmk_val'] / df_report.iloc[0]['bmk_val']) ** (1 / n_steps) - 1) * freq
    port_ann_ret = ((df_report.iloc[-1]['port_val'] / df_report.iloc[0]['port_val']) ** (1 / n_steps) - 1) * freq
    # tracking_cum = df_report.iloc[-1]['port_val'] / df_report.iloc[-1]['bmk_val'] - 1

    # Pack results into a datframe:
    sim_stats = pd.Series(dtype='float64')
    sim_stats['port_ret'] = port_ann_ret
    sim_stats['index_ret'] = idx_ann_ret
    sim_stats['index_vol'] = df_report['bmk_val'].pct_change().std() * np.sqrt(freq)
    sim_stats['tracking_std'] = tracking_std * np.sqrt(freq)
    sim_stats['hvst_grs'] = hvst_grs / years
    sim_stats['hvst_net'] = hvst_net / years
    sim_stats['hvst_potl'] = hvst_potl / years
    sim_stats['hvst_grs/potl'] = ratio_grs
    sim_stats['hvst_net/potl'] = ratio_net
    sim_stats['hvst_n/trckng'] = hvst_n_2_trk_ann

    logging.info("\nSimulation statistics (annualized):")
    logging.info(sim_stats.to_string())
    if save_files:
        pd_to_csv(sim_stats, "stats", suffix=suffix, dir_path=dir_path)

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
    input_file = 'inputs/test100.xlsx'
    inputs = load_sim_settings_from_file(input_file, randomize=True)
    sim_stats, step_report = run_sim(inputs, suffix=timestamp)

    print("\nTrade Summary (x100, %):")
    print(df_to_format(step_report * 100, formats={'_dflt': '{:.2f}'}))

    print("\nSimulation statistics (%):")
    print(df_to_format(pd.DataFrame(sim_stats * 100, columns=["annualized (%)"]), formats={'_dflt': '{:.2f}'}))
    print("\nDone")
