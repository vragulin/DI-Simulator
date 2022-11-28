"""
    Heuristic harvesting rebalance, with a cash target
    by V. Ragulin, started 30-Oct-2022
"""

import numpy as np
import pandas as pd
from port_lots_class import PortLots
import config


def heuristic_w_cash(port: PortLots, t: int, sim_data: dict, max_harvest: float = 0.5, log=False) -> dict:
    """ Heuristic tax-harvesting for an account that has cash as well as stocks
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

    # Calculate tax per dollar sold and total harvest potential
    harvest_thresh = params['harvest_thresh']
    df_lots['tax_per_$'] = df_lots['tax per shr'] / df_lots['price']

    # Set initial sell threshold so that we only include losses > harvest_threshold
    sell_thresh = harvest_thresh * sim_data['tax']['st']

    # Max harvest - an exogenous constraint to make sure we have a reasonable selection of stocks for buying
    max_harvest_val = port.port_value * (1 - (1 - max_harvest) * (1 - port.cash_w_tgt)) - port.cash

    # Min harvest - what we need to sell in order to reach te cash target
    min_harvest_val = port.port_value * port.cash_w_tgt - port.cash

    # Number of new lots that we will be adding at each loop iteration
    thresh_step = max(int(config.threshold_step_prc * len(df_lots)), 1)
    iter_count = 0

    # Loop until we meet all constraints as we make the threshold gradually more inclusive
    while True:
        df_lots['below_thresh'] = df_lots['tax_per_$'] <= sell_thresh
        df_lots['potl_harv_val'] = -df_lots['tax_per_$'] * df_lots['mv'] * df_lots['below_thresh']

        # Identify a set of stocks from which we will be harvesting
        # Select stocks with the highest harvesting potential up to max_harvest
        df_stocks['num_harv_lots'] = df_lots.groupby(by='ticker')['below_thresh'].sum()
        df_stocks['potl_harv_val'] = df_lots.groupby(by='ticker')['potl_harv_val'].sum()

        # Only consider stocks that we are allowed to sell
        df_stocks['num_harv_lots'] *= (1 - df_stocks['no_sells'])
        df_stocks['potl_harv_val'] *= (1 - df_stocks['no_sells'])

        # On the first iteration, harvest in order of potL_harv_val (same as uncontrained case),
        # if that does not work, switch metric to pot_harv_to_mv
        if iter_count == 0:
            sort_col = 'potl_harv_val'
        else:
            df_stocks['potl_harv_to_mv'] = df_stocks['potl_harv_val'] / df_stocks['mkt_vals']
            sort_col = 'potl_harv_to_mv'

        df_stocks.sort_values(sort_col, ascending=False, inplace=True)

        df_stocks['cum_mv_for_harv'] = np.where(df_stocks['no_sells'], 0, df_stocks['mkt_vals']).cumsum(axis=0)

        # Only check for max_harvest_val if we are not using all available lots:
        if not df_lots['below_thresh'].all():
            df_stocks['to_sell'] = (df_stocks['cum_mv_for_harv'] <= max_harvest_val) \
                                   & (df_stocks['num_harv_lots'] > 0)
        else:
            df_stocks['to_sell'] = df_stocks['num_harv_lots'] > 0

        # From the 'to_sell' stocks identify specific lots
        df_lots['stk_to_sell'] = df_lots['ticker'].apply(lambda x: df_stocks.loc[x, 'to_sell'])
        df_lots['trade'] = -df_lots['shares'] * df_lots['stk_to_sell'] * df_lots['below_thresh']
        df_lots['tax'] = df_lots['stk_to_sell'] * df_lots['potl_harv_val']

        # Make sure we don't sell too much
        if not df_lots['below_thresh'].all():
            df_stocks['trade'] = df_lots.groupby(by='ticker')['trade'].sum()
        else:
            # If all lots are <= threshold, cap the total sold amount
            df_stocks['cum_mv_for_harv'] = np.minimum(df_stocks['cum_mv_for_harv'], min_harvest_val)
            df_stocks['trade_mv'] = df_stocks['cum_mv_for_harv'].diff()
            df_stocks.loc[df_stocks.index[0], 'trade_mv'] = df_stocks['cum_mv_for_harv'].iloc[0]
            df_stocks['trade'] = -df_stocks['trade_mv'] / df_stocks['price']

        # Also sell stocks with w_actv > max_active_weight where we have long-term capital gains
        if ('max_active_wgt' in params) and (df_stocks['w_actv'] > params['max_active_wgt']).any():
            # Identify long-term lots for stocks where we have excess overweights
            df_lots['for_cutting_pos'] = (df_lots['w_actv'] > params['max_active_wgt']) \
                                         & (df_lots['long_term'])\
                                         & (~df_lots['stk_to_sell'])\
                                         & (~df_lots['no_sells'])

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
        if log: print(df_stocks['trade'])

        # Check if this selling is raising sufficient cash to meet the target
        mv_harvest = -df_stocks['trade'] @ df_stocks['price']
        if mv_harvest >= min_harvest_val:
            break
        else:
            # make the harvest threhold more inclusive, and try again -- adjust the threshold to add threshold_step
            # new lots
            excluded_lots_tax = df_lots[~df_lots['below_thresh']]['tax_per_$']
            if len(excluded_lots_tax) == 0:
                break
            else:
                sell_thresh = sorted(excluded_lots_tax)[min(thresh_step, len(excluded_lots_tax))-1]
                if log:
                    print(f"iter {iter_count}, # excluded lots = {len(excluded_lots_tax)}, new sell thresh = {sell_thresh:.5f}, "
                          f"harvest = {df_stocks['trade'] @ df_stocks['price']:.0f}")

        iter_count += 1

    # Now calculate the buys to offset beta impact of harvest sales and re-invest excess cash
    # For now assume that beta of all stocks is 1
    # mv_harvest = -df_stocks['trade'] @ df_stocks['price']
    tot_mv_to_buy = mv_harvest + port.cash - port.cash_w_tgt * port.port_value

    if tot_mv_to_buy > 0:
        # Invest the cash into the remaining 'to_buy' stocks
        # Define priority of purchases, starting with underweight stocks
        df_stocks.sort_values(['to_sell', 'w_actv'], ascending=[True, True], inplace=True)

        # Check if we have stocks left 'to buy'
        if not (df_stocks['to_sell'] | df_stocks['no_buys']).all():
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
        else:
            # Special case if all stocks have been classified as "to_sell".  It's a rare case that can only happen
            # When portfolio is very concentrated.  In this case just reduce all sales proportionally.
            df_stocks['trade'] *= (mv_harvest - tot_mv_to_buy) / mv_harvest

    elif tot_mv_to_buy < -config.tol:  # 0:
        raise ValueError("Unable to meet both constraints. Probably there are too many 'no sell' stocks.")

    # Clean up - sort stocks in alphabetical order
    df_stocks.sort_index(inplace=True)

    # Pack output into a dictionary
    # Note that the harvest field does not take into account the tax we paid on rebalancing trades
    # that is taken from the port.rebal_sim() method
    # noinspection PyDictCreation
    res = {'opt_trades': df_stocks['trade'],
           'potl_harvest': df_stocks['potl_harv_val'].sum(),
           'harvest': np.nan  # This algo does not calculate this, it will be done after the rebalance
           }

    res['harv_ratio'] = np.nan
    res['port_val'] = port.port_value

    return res
