"""
    3-step harvesting simulation.
    by V. Ragulin, started 22-Oct-2022
"""
import numpy as np
import pandas as pd
from port_lots_class import PortLots
import config


def heur_3step_rebal(port: PortLots, t: int, sim_data: dict, max_harvest: float = 0.5) -> dict:
    """ Heuristic tax-harvesting using Vic's 3-step method
    """

    # We harvest lots by whether they are over/underweight and tax per dollar sold
    # Add columns with active weight for each stock

    # print(f"t = {t}")
    # Process lots to speed up calculations
    t_date = sim_data['dates'][t]
    params = sim_data['params']
    port.process_lots(sim_data['tax'], t_date)

    # Split lots that became too large (it matters since we don't havest partial lots)
    max_lot_size = port.port_value * max_harvest * config.max_lot_prc
    port.split_large_lots(max_lot_size)
    port.process_lots(sim_data['tax'], t_date)

    # Start of the rebalance algo
    df_lots = port.df_lots
    df_stocks = port.df_stocks

    # Add info on over/underweight stocks
    df_lots['w_actv'] = df_lots['ticker'].apply(lambda x: df_stocks.loc[x, 'w_actv'])
    df_lots['overweight'] = df_lots['w_actv'] > 0
    df_lots['mv'] = df_lots['price'] * df_lots['shares']

    # Calculate tax per dollar and total harvest potential
    harvest_thresh = params['harvest_thresh']
    df_lots['tax_per_$'] = df_lots['tax per shr'] / df_lots['price']
    df_lots['below_thresh'] = df_lots['%_gain'] <= harvest_thresh
    df_lots['potl_hvst_val'] = np.maximum(-df_lots['tax_per_$'] * df_lots['mv'] * df_lots['below_thresh'], 0)

    # Identify a set of stocks from which we will be harvesting
    # Calculate post-harvest active weights to see which harvests will be the riskiest
    # TODO - consider taking a portfolio approach, and prioritize harvests with the largest portfolio impact
    df_lots['mv_to_hvst'] = np.where(df_lots['potl_hvst_val'] > 0, df_lots['mv'], 0)
    df_stocks['mv_to_hvst'] = df_lots.groupby(by='ticker')['mv_to_hvst'].sum()
    df_stocks['w_actv_post_hvst'] = np.where(df_stocks['mv_to_hvst'] > 0,
                                             df_stocks['w_actv'] - df_stocks['mv_to_hvst'] / port.port_value,
                                             0)

    # Identify stocks for the 3-step harvest:  Pick stocks with the lasgest post-harvest underweight
    df_stocks.sort_values('w_actv_post_hvst', inplace=True)
    potl_hvst = df_stocks['mv_to_hvst'].sum()
    df_stocks['cum_mv_to_hvst'] = df_stocks['mv_to_hvst'].cumsum()
    if potl_hvst > 0:
        df_stocks['3step'] = df_stocks['cum_mv_to_hvst'] <= config.prc_3step_hvst * potl_hvst

        # Check if for any 3-step candidates buying in step1 is riskier than just full harvesting or over limit
        df_stocks['w_actv_step1'] = np.where(df_stocks['3step'],
                                             df_stocks['w_actv']
                                             + df_stocks['mv_to_hvst'] * config.prc_3step_buy / port.port_value,
                                             0)

        df_stocks['3step'] = np.where(
            (np.abs(df_stocks['w_actv_post_hvst']) < np.abs(df_stocks['w_actv_step1'])) |
            (df_stocks['w_actv_post_hvst'] > params['max_active_wgt']),
            False, df_stocks['3step'])
    else:
        df_stocks['3step'] = False

    # Check that there are enough remaining harvesting candidates to offset the buys and
    # stay below the harvesting limit, otherwise, drop some 3step stocks
    while True:
        mv_buy_3step = (df_stocks['3step'] * df_stocks['mv_to_hvst'] * config.prc_3step_buy).sum()

        # No move to the actual harvesting.  Interesting that we don't really need to have any memory of prior steps.
        # Select stocks with the highest harvesting potential up to max_harvest
        df_stocks['potl_hvst_val'] = df_lots.groupby(by='ticker')['potl_hvst_val'].sum()
        full_potl_hvst_val = df_stocks['potl_hvst_val'].sum()  # Includes stocks that we chose to harvest via 3-step
        df_stocks['potl_hvst_val'] *= (1 - df_stocks['3step'])

        df_stocks.sort_values('potl_hvst_val', ascending=False, inplace=True)
        df_stocks['cum_mv_to_hvst'] = df_stocks['mkt_vals'].cumsum(axis=0)

        max_harvest_val = max_harvest * port.port_value + mv_buy_3step
        df_stocks['to_sell'] = (df_stocks['cum_mv_to_hvst'] <= max_harvest_val) & (df_stocks['potl_hvst_val'] > 0)
        df_lots['stk_to_sell'] = df_lots['ticker'].apply(lambda x: df_stocks.loc[x, 'to_sell'])
        df_lots['trade'] = -df_lots['shares'] * df_lots['stk_to_sell'] * (df_lots['potl_hvst_val'] > 0)
        df_lots['tax'] = df_lots['stk_to_sell'] * df_lots['potl_hvst_val']

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
                df_stocks['shrs_for_cut_w_cap'] = np.minimum(df_stocks['out_actv_band_shrs'],
                                                             df_stocks['shrs_for_cutting'])

                # Add to the trade list
                df_stocks['to_sell'] = df_stocks['to_sell'] | (df_stocks['shrs_for_cut_w_cap'] > 0)
                df_stocks['trade'] -= df_stocks['shrs_for_cut_w_cap']

        # Make sure we don't sell more than what we have (due to machine precision issues etc.)
        df_stocks['trade'] = np.maximum(-df_stocks['shares'], df_stocks['trade'])

        # Now calculate the buys to offset beta impact of harvest sales and re-invest excess cash
        # For now assume that beta of all stocks is 1

        # Buys driven by the first step of the 3 step harvest
        df_stocks['trade'] = np.where(df_stocks['3step'],
                                      df_stocks['mv_to_hvst'] / df_stocks['price'] * config.prc_3step_buy,
                                      df_stocks['trade'])

        mv_harvest = -df_stocks['trade'] @ df_stocks['price']

        if mv_harvest < 0:
            # Drop one of 3step candidates with the highest index
            idx_3step_max = np.max(np.where(df_stocks['3step']))
            df_stocks['3step'].iloc[idx_3step_max] = False

            # # If we dropped everything, exit the loop <- don't exit the loop, recalculate all the trades
            # if df_stocks['3step'].sum() == 0:
            #     break
        else:
            break

    tot_mv_to_buy = mv_harvest + port.cash - port.cash_w_tgt * port.port_value

    # Define priority of purchases, starting with underweight stocks
    df_stocks.sort_values(['3step', 'to_sell', 'w_actv'], ascending=[True, True, True], inplace=True)

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
        n_buys = (~(df_stocks['to_sell'] | df_stocks['3step'])).sum()
        mv_remaining_per_stock = shortfall / n_buys
        df_stocks['buy_mv'] += np.where(df_stocks['to_sell'] | df_stocks['3step'], 0, mv_remaining_per_stock)

    df_stocks['trade'] += df_stocks['buy_mv'] / df_stocks['price']

    # Clean up - sort stocks in alphabetical order
    df_stocks.sort_index(inplace=True)

    # Pack output into a dictionary
    # Note that the harvest field does not take into account the tax we paid on rebalancing trades
    # that is taken from the port.rebal_sim() method
    res = {'opt_trades': df_stocks['trade'],
           'potl_harvest': full_potl_hvst_val,
           'harvest': df_lots['tax'].sum()}

    res['harv_ratio'] = res['harvest'] / res['potl_harvest'] if res['potl_harvest'] > 0 else np.nan
    res['port_val'] = port.port_value

    return res
