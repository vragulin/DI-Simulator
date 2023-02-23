""" Simulate an account that has both index and cash positions.
    As a start, assume the index basket (in terms of shares) does not change, so it's equivalent to trading an ETF)

    Started V. Ragulin - 4-Nov-2022
"""
import copy
import datetime
import os
import logging
import warnings
import pandas as pd
import config
import numpy as np
import index_math as im

from typing import Optional, Tuple
from codetiming import Timer
from contextlib import suppress
from load_mkt_data import vectorize_dict
from port_lots_class import PortLots, DispMethod
from sim_one_path import log_port_report, init_sim_hist, trade_history, \
    pd_to_csv, init_optimizer, update_donate
from pretty_print import df_to_format
from heur_w_cash import heuristic_w_cash

warnings.filterwarnings("ignore")


def load_sim_settings(file: str, randomize=False) -> dict:
    from sim_one_path import load_sim_settings_from_file

    # Load portfolio data
    input_dict = load_sim_settings_from_file(file, randomize)

    # Also, load cash data
    cash_info = pd.read_excel(file, sheet_name='cash', index_col='Date')
    input_dict['cash_info'] = cash_info.T

    return input_dict


def process_input_data(inputs: dict, input_from_xl: bool = False) -> dict:
    """ Process inputs into a dictionary format that can be used for a simulation
        :param inputs: input dictionary
        :param input_from_xl: flag idicating whether the data was read form an Excel or a series of pickle files
        :return: simulation data dictionary ('sim_data')
    """
    params = inputs['params']
    freq = int(config.ANN_FACTOR / params['dt'])

    if input_from_xl:
        tickers = list(inputs['stk_info'].index)
        prices = inputs['prices']
        dates = inputs['dates']
    else:
        tickers = list(inputs['px'].columns)
        prices = inputs['px']
        dates = inputs['dates'][list(inputs['dates_idx'])]

    d_px = prices.pct_change().fillna(0).replace(np.Inf, 0)
    px = (1 + d_px).cumprod()

    dt = params['dt']

    # Calc dividend per period and convert into an array
    if input_from_xl:
        div_y_per_dt = inputs['stk_info']['Div Yield'] * dt / config.ANN_FACTOR
        div = np.zeros(px.shape)
        div[1:] = div_y_per_dt
    else:
        div = inputs['div_arr']
        if params['div_override'] >= 0:
            div *= 0.0
            div[(px.index % freq == 0) & (px.index > 0)] = params['div_override']

    bmk_type = inputs['params']['benchmark_type']
    if bmk_type == 'fixed_shares':
        if input_from_xl:
            w_start = inputs['w_idx'].values
        else:
            w_start = inputs['w_arr'][0, None].T

        idx_div = np.zeros((px.shape[0], 1))
        idx_div[1:] = (div[1:] * px[:-1]) @ w_start

        w = im.index_weights_over_time(w_start, px.values)
        idx_vals = im.index_vals(w_start, px.values)
        idx_tri = im.total_ret_index(w_start, px.values, idx_div * (1 - config.tax['div']))

    elif bmk_type in ['var_weights', 'fixed_weights']:
        # Calculate index prices and price returns
        w = inputs['w']
        idx_rets = (d_px * w.shift(1)).sum(axis=1)
        idx_rets[0] = 0
        idx_vals = (1 + idx_rets).cumprod()

        # idx_vals1, idx_rets1 = im.index_vals_from_weights(inputs['w'].values, px.values)
        # np.testing.assert_allclose(idx_vals.values, idx_vals1)
        # np.testing.assert_allclose(idx_vals.values, idx_vals1)

        # Calculate index dividends (in % and price points)
        idx_div_yld = (div * w.shift(1)).sum(axis=1)
        idx_div_yld[0] = 0
        idx_div = idx_div_yld * idx_vals.shift(1)

        # Calculate index Total return (i.e. including dividends)
        idx_tot_rets = idx_rets + idx_div_yld
        idx_tri = (1 + idx_tot_rets).cumprod()
    else:
        raise NotImplementedError(f"Invalid benchmark type: {inputs['params']['benchmark_type']}")

    # Calc total return
    d_tri = d_px + div
    tri = (1 + d_tri).cumprod()

    # Adjust returns to match target if needed
    if params['ret_override_flag']:
        n_steps = len(px) - 1
        idx_period_ret = (idx_vals.ravel()[-1] / idx_vals.ravel()[0]) ** (1 / n_steps) - 1
        tgt_period_ret = (1 + params['ret_override']) ** (dt / config.ANN_FACTOR / config.EMBA_IRR_Factor) - 1
        adj_factor = (1 + tgt_period_ret) / (1 + idx_period_ret) - 1

        d_px.iloc[1:] = (1 + d_px.iloc[1:]) * (1 + adj_factor) - 1
        px = (1 + d_px).cumprod()

        # Calc total return for the stocks
        d_tri = d_px + div
        tri = (1 + d_tri).cumprod()

        # Recalculate index stats and other series to match target return
        if bmk_type == 'fixed_shares':
            idx_div = np.zeros((px.shape[0], 1))
            idx_div[1:] = (div[1:] * px[:-1]) @ w_start

            w = im.index_weights_over_time(w_start, px.values)
            idx_vals = im.index_vals(w_start, px.values)
            idx_tri = im.total_ret_index(w_start, px.values, idx_div * (1 - config.tax['div']))

        elif bmk_type in ['var_weights', 'fixed_weights']:
            # Calculate index prices and price returns
            idx_rets = (d_px * inputs['w'].shift(1)).sum(axis=1)
            idx_rets[0] = 0
            idx_vals = (1 + idx_rets).cumprod()

            # Calculate index dividends (in % and price points)
            idx_div_yld = (div * inputs['w'].shift(1)).sum(axis=1)
            idx_div_yld[0] = 0
            idx_div = idx_div_yld * idx_vals.shift(1)

            # Calculate index Total return (i.e. including dividends)
            idx_tot_rets = idx_rets + idx_div_yld
            idx_tri = (1 + idx_tot_rets).cumprod()

        else:
            raise NotImplementedError(f"Invalid benchmark type: {inputs['params']['benchmark_type']}")

    # Unpack cash info.  Different logic depening on whether the data is from excel or pickle
    if input_from_xl:
        cash = inputs['cash_info']
        eq_alloc = cash[['weight']]
        int_rate = cash[['ann int rate']]
    else:
        eq_alloc = inputs['eq_alloc']
        int_rate = inputs['int_rate']

    # Adjust weights for the equity/cash allocation target
    # w_adj = w * eq_alloc.values[:, None]
    w_adj = w * eq_alloc.values

    # Calculate period and cumulative returns on the cash account
    cash_ret = int_rate * dt / config.ANN_FACTOR
    cash_tri = (1 + cash_ret.shift().fillna(0)).cumprod()

    # Pack results into an output structure
    out_dict = {'tickers': tickers, 'px': px, 'd_px': d_px, 'tri': tri, 'd_tri': d_tri,
                'eq_alloc': eq_alloc, 'int_rate': int_rate,
                'cash_ret': cash_ret, 'cash_tri': cash_tri,
                'div': div, 'w': w_adj, 'idx_div': idx_div,
                'idx_vals': idx_vals, 'idx_tri': idx_tri,
                'bmk_vals': inputs['bmk_val'], 'dates': dates, 'tax': config.tax,
                'params': inputs['params'], 'trx_cost': config.trx_cost}

    vectorize_dict(out_dict, ['px', 'd_px', 'eq_alloc', 'int_rate', 'cash_ret', 'cash_tri',
                              'tri', 'd_tri', 'div', 'w'])

    return out_dict


def merge_rebal_harvest(opt_res: dict, rebal_res: dict, hvst_res: dict, algorithm: str = 'inst_replace') -> None:
    """  Update rebalance results inplace for the effects of the harvest (in place)
        :param opt_res: output of the
        rebalance algo
        :param rebal_res: output of the algo
        :param algorithm: algorithm used for the harvest
    """

    if algorithm == 'inst_replace':
        # Upate optimizer statistics
        for field in ['potl_harvest', 'harvest', 'harvest_trades', 'harv_ratio']:
            opt_res[field] = hvst_res[field]

        # Update tax and transacions costs statistics
        rebal_res['tax'] -= hvst_res['harvest']
        rebal_res['trx_cost'] += hvst_res['trx_cost']

    else:
        raise NotImplementedError(f"{algorithm} not implemented")

    return None


@Timer(name="One path sim", text="One path simulation time: {seconds:.1f} sec")
def run_sim_w_cash(inputs: dict, suffix: Optional[str], dir_path: str = '../results/ac_w_cash/',
                   verbose: bool = True, save_files: bool = True, log_file: str = None) -> tuple:
    # Load simulation parameters from file
    params = inputs['params']
    disp_method = inputs.get('disp_method', DispMethod.LTFO)
    max_harvest = inputs['params'].get('max_harvest', config.MAX_HVST_DFLT)

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

    # Create the sim_data structure
    input_from_xl = not inputs.get('prices_from_pickle', False)
    sim_data = process_input_data(inputs, input_from_xl=input_from_xl)

    algorithm = params.get('algorithm', 'heuristic')
    if algorithm == "optimizer":
        sim_data = init_optimizer(sim_data)

    # Set up the initial portfolio matching the index at t=0
    port = PortLots.init_portfolio_from_dict(sim_data)
    port.update_sim_data(sim_data=sim_data, t=0)

    # log_port_report(port, 0)

    # Loop over periods, rebalance / harvest at each period, keep track of P&L
    # Initialize a list to keep simulation info
    sim_hist = init_sim_hist(port)

    if config.TRACKING_RPT:
        df_actv_w = inputs['px'] * 0

    t_date = sim_data['dates'][0]

    for t in range(1, n_steps + 1):

        if t % config.VERBOSE_FREQ == 0:
            print(f"Step = {t}", end=",\t")
            print(f"# lots: {len(port.df_lots)}", end=",\t")
            print(f"# stocks: {len(port.df_stocks)}")

        # Take account of the dividends that we have received during the period
        # Assume that all divs during the period happen on the last day (i.e. stocks are ex-div)
        div, interest = port.cf_during_period(t, sim_data)
        if not params.get('div_payout', True):
            port.cash += (div + interest)
            div, interest = 0, 0

        # Revalue portfolio
        port.update_sim_data(sim_data=sim_data, t=t)
        # logging.info("\nBefore rebalance:")
        # log_port_report(port, t)

        t_date = sim_data['dates'][t]

        # Execute the rebalance depending on the chosen algo
        if algorithm in ['passive', 'inst_replace']:

            # Recalculate and sort the lots array in the right way
            port.process_lots(sim_data['tax'], t_date, method=inputs.get('disp_method'), sim_data=sim_data)

            # Calculate the rebalance
            opt_res = {
                'opt_trades': -port.df_stocks['w_actv'] * port.port_value / port.df_stocks['price'],
                'harvest': np.nan,
                'potl_harvest': np.nan,
                'harv_ratio': np.nan,
                'port_val': port.port_value
            }

            # logging.info("Trades:")
            # logging.info(opt_res['opt_trades'])

            # Execute the rebalance (for now don't worry about donations
            rebal_res = port.rebal_sim(opt_res['opt_trades'], sim_data, t=t,
                                       method=disp_method)

            # logging.info("\nRebal Trades:")
            # logging.info(opt_res['opt_trades'])

            if algorithm == 'inst_replace':
                # Keeping portfolio weights constant harvest the remaining lots
                hvst_res = port.harvest_inst_replace(t, sim_data)
                merge_rebal_harvest(opt_res=opt_res, rebal_res=rebal_res, hvst_res=hvst_res)

                # logging.info("\nHarvest Trades:")
                # logging.info(opt_res['harvest_trades'])

        elif algorithm == 'heuristic':
            # ToDo understand why it gives harvest = nan
            opt_res = heuristic_w_cash(port, t, sim_data, max_harvest)
            rebal_res = port.rebal_sim(trades=opt_res['opt_trades'], sim_data=sim_data, t=t)

        else:
            raise NotImplementedError(f"Algo {algorithm} not implemented")

        # logging.info(f"\nHarvest={opt_res['harvest']:.4f}, "
        #              f"Potl Harvest={opt_res['potl_harvest']:.4f}, "
        #              f"Ratio={opt_res['harv_ratio']:.3f}")
        # logging.info(f"\nTax={rebal_res['tax']:.4f}, "
        #              f"Net Buy={rebal_res['net_buy']:.4f}")

        # Donate
        if params['donate'] and (t * params['dt'] % params['donate_freq'] == 0):
            donate_amount = update_donate(port, params['donate_thresh'], params['donate'])
        else:
            donate_amount = 0

        # Clock reset
        if params['clock_reset']:
            clock_reset_tax = port.reset_clock(reset_thresh=params['reset_thresh'])
        else:
            clock_reset_tax = 0

        # ToDo: to reduce tracking check if I can reduce overweight holdings that were
        #  just reset or donated to reduce tracking
        #  Write another algo to reduce tracking at zero tax after a donate or close reset exercise.
        #  Think if I can make the donate and/or reset strategy more optimal
        sim_hist.append({'opt': opt_res, 'div': div, 'interest': interest,
                         'rebal': rebal_res, 'donate': donate_amount,
                         'clock_reset_tax': clock_reset_tax})

        if config.TRACKING_RPT:
            df_actv_w.iloc[t, :] = port.df_stocks['w_actv']

    # Liauidation tax at the end
    if params['final_liquidation'] > 0:
        port.update_sim_data(sim_data=sim_data, t=n_steps)
        port.process_lots(sim_data['tax'], t_date, method=inputs.get('disp_method'), sim_data=sim_data)
        sim_hist[-1]['fin_liq_tax'] = port.liquid_tax(liq_pct=params['final_liquidation'])

    logging.info("\nAfter final rebalance:")
    log_port_report(port, n_steps)

    logging.info("\nSimulation results:")
    sim_report, sim_stats = gen_sim_report(sim_hist, sim_data)
    logging.info(sim_report.to_string())
    if save_files:
        pd_to_csv(sim_report, "steps", suffix=suffix, dir_path=dir_path)

    # Generate and save trade history
    sim_trades = trade_history(sim_hist, sim_data, shares_flag=True)
    if save_files:
        pd_to_csv(sim_trades, "trades", suffix=suffix, dir_path=dir_path)

    logging.info("\nSimulation statistics (annualized):")
    logging.info(sim_stats.to_string())
    if save_files:
        pd_to_csv(sim_stats, "stats", suffix=suffix, dir_path=dir_path)

    # Generate a tracking analysis report
    if config.TRACKING_RPT:
        _, top_tracking = tracking_analysis(sim_report, df_actv_w, sim_data)
        if save_files:
            for code in ['all']: # sim_tracking.keys(): <- use this to log longs and shorts
                pd_to_csv(top_tracking[code], f"tracking_{code}", suffix=suffix,
                          dir_path=dir_path)
            # ToDo - also print top tracking by period and show calendar dates, not indices
    else:
        top_tracking = None


    return sim_stats, sim_report, top_tracking


def gen_sim_report(sim_hist: list, sim_data: dict) -> tuple:
    """ Print simulationr results """

    params = sim_data['params']
    # div_payout = params.get('div_payout', True)

    freq = int(config.ANN_FACTOR / params['dt'])
    n_steps = len(sim_hist) - 1

    rpt_cols = ['harvest', 'donate', 'div', 'interest', 'port_val']
    df_report = pd.DataFrame(np.nan, columns=rpt_cols, index=range(n_steps + 1))
    df_report.index.name = 'step'

    for i in range(n_steps + 1):
        df_report.loc[i, 'harvest'] = -sim_hist[i]['rebal']['tax'] \
                                      - sim_hist[i]['clock_reset_tax']
        df_report.loc[i, 'donate'] = sim_hist[i]['donate']
        df_report.loc[i, 'div'] = sim_hist[i]['div']
        df_report.loc[i, 'interest'] = sim_hist[i]['interest']
        df_report.loc[i, 'port_val'] = sim_hist[i]['opt']['port_val']

    # Rescale as % of portfolio value
    for col in ['harvest', 'donate', 'div', 'interest']:
        df_report[col] /= df_report['port_val']

    # Generate other statistics that don't go into the df_report table
    # Liquidation taxes if applicable
    sim_stats = pd.Series(dtype='float64')
    if 'fin_liq_tax' in sim_hist[-1]:
        port_liq_tax = sim_hist[-1]['fin_liq_tax']
    else:
        port_liq_tax = 0

    # IRR metrics
    years = n_steps / freq

    # Calculate portfolio after-tax IRR
    cf = df_report['harvest'].values.copy()
    cf += (df_report['div'].values + df_report['interest'].values)
    cf *= df_report['port_val'].values
    cf[0] -= df_report.loc[0, 'port_val']
    cf[-1] += df_report.loc[n_steps, 'port_val'] - port_liq_tax
    sim_stats['port_irr'] = im.irr_solve(cf, params['dt'], ann_factor=config.ANN_FACTOR,
                                         # bounds=(-0.05, 0.2), freq=freq)
                                         bounds=(-0.5, 0.5), freq=None)

    # Adjust to match EMBA calculation with 252 days for discounting
    sim_stats['port_irr'] *= config.EMBA_IRR_Factor

    # Calculate aggregate statistics
    sim_stats['harvest%'] = df_report['harvest'].sum() / years

    # Pack results into a datframe:
    sim_stats['liq_tax%'] = port_liq_tax / df_report.iloc[-1]['port_val']

    # Calculate tracking
    if sim_data['bmk_vals'] is not None:
        if len(sim_data['bmk_vals']) == len(df_report):
            df_report['bmk_val'] = sim_data['bmk_vals']
            df_report['tracking'] = np.log(df_report['port_val'] / df_report['bmk_val']).diff()
            sim_stats['tracking'] = df_report['tracking'].std() * np.sqrt(freq)
        else:
            print(f"\nWarning benchmark and sim length don't match!\n"
                  f"bmk length = {len(sim_data['bmk_vals'])}, "
                  f"sim_length = {len(df_report)}.\n"
                  f"Ignoring benchmark.\n")

    # Re-arrange sim_stats fields in logical order
    sim_stats = sim_stats.reindex(index=['port_irr', 'harvest%', 'liq_tax%', 'tracking'])

    return df_report, sim_stats


def tracking_analysis(sim_report: pd.DataFrame, df_actv_w: pd.DataFrame,
                      sim_data: dict, use_bmk: bool = False, verbose=True) -> Tuple[dict, dict]:
    """ Generate a dictionary of several tracking reports
    :param sim_report:  table of harvest, div, tracking, port_val vs. time
    :param df_actv_w: dataframe with active weights at each step
    :param sim_data: dict with input data for the simulation
    :param use_bmk: if True calc tracking vs. bmk, else vs. portfolio (the differenceis the rebalancing frequency)
    :param verbose: if True, print out reports
    :return: dataframe with tracking analysis by stock
    """

    # Calculate a matrix of residual returns by stock (assume beta=1)
    # Tracking is calculated vs. a 'perfectly replicated portfolio' with the same
    # frequency as the simulation, not the actual index, which is rebalanced daily

    # Calculate residual returns
    if use_bmk:
        raise NotImplementedError(f'Tracking vs. bmk not implemented')
    else:
        idx_rets = sim_data['idx_vals'].pct_change().fillna(0)
    res_rets = sim_data['d_px'] - idx_rets[:, None]

    # Tracking by stock
    df_trk = (res_rets * df_actv_w.shift(1)).fillna(0)
    trk_by_stk = pd.DataFrame(df_trk.sum(axis=0), columns=['cum tracking'])

    full_track_dict = {'all': trk_by_stk}

    # To the same table calc separate for longs and shorts
    for code, sgn in zip(['long', 'short'], [1, -1]):
        df_trk_1side = df_trk.where(df_actv_w * sgn > 0, 0)
        trk_by_stk_1side = pd.DataFrame(df_trk_1side.sum(axis=0),
                                        columns=['cum tracking ' + code])
        full_track_dict[code] = trk_by_stk_1side

    # Calc top results
    n_top = 5
    top_idx = list(range(n_top)) + [-n_top + x for x in range(n_top)]
    top_dict = {}

    for code in full_track_dict.keys():
        col_name = 'cum tracking' + (" " + code if code != 'all' else "")
        trk_one_case = (full_track_dict[code]).sort_values(by=[col_name])
        df_top = trk_one_case.iloc[top_idx, :] * 100
        top_dict[code] = df_top
        logging.info(f"\nTop {n_top * 2} tracking stocks - {code}:")
        logging.info(df_top.to_string())
        if verbose:
            print(f"\nTop {n_top * 2} tracking stocks - {code}:")
            print(df_to_format(df_top, formats={'_dflt': '{:.2f}'}))

    # Print top tracking by period
    trk_ts = pd.DataFrame(df_trk.sum(axis=1).values, index=sim_data['dates'], columns=['tracking'])
    trk_ts = trk_ts.sort_values(by=['tracking'])
    top_trk_steps = trk_ts.iloc[top_idx] * 100
    logging.info(f"\nTop {n_top * 2} tracking periods:")
    logging.info(top_trk_steps.to_string())
    if verbose:
        print(f"\nTop {n_top * 2} tracking periods:")
        print(df_to_format(top_trk_steps, formats={'_dflt': '{:.2f}'}))

    # Tracking by date
    return full_track_dict, top_dict


# ***********************************************************
# Entry point
# ***********************************************************
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = '../results/ac_w_cash/sim_' + timestamp + '.log'

    # Silently remove prior log_files and set up logger
    with suppress(OSError):
        os.remove(log_file)

    logging.basicConfig(level=logging.INFO, filename=log_file,
                        format='%(message)s')

    # Load data
    input_file = '../inputs/test_idx_cash5.xlsx'
    inputs = load_sim_settings(input_file, randomize=False)
    inputs['disp_method'] = DispMethod.TSST_ETF  # Set lot disposition method
    print(f"Disp Method = {inputs['disp_method'].name}")

    # Start the simulation
    sim_stats, step_report, tracking = run_sim_w_cash(inputs, suffix=timestamp)

    # Print results: step report + simulation statistics
    print("\nTrade Summary (x100, %):")
    # print(f"Simulation time = {toc-tic} seconds.")
    print(df_to_format(step_report * 100, formats={'_dflt': '{:.2f}'}))

    print("\nSimulation statistics (%):")
    print(df_to_format(pd.DataFrame(sim_stats * 100, columns=["annualized (%)"]), formats={'_dflt': '{:.2f}'}))

    if config.TRACKING_RPT:
        print('\nTracking Analysis %')
        print(tracking)

    print("\nDone")
