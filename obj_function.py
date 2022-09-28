"""
Created on Fri JuL 13  2022

Objective function used for the Direct Indexing simulation
Based on the Ang's paper, incorporates tax, trx_costs, risk_aversion and alpha
@author: vragulin
"""

import numpy as np

import fast_lot_math as flm
from port_lots_class import PortLots

INFINITY = 9.999e15

def fetch_obj_params(port: PortLots, t: int, sim_data: dict, max_harvest: float) -> dict:
    """ Extract parameters used in the objective function from structures in the correct
        np array format that can then be passed to obj fucntion and constraints
    :param port:
    :param t:
    :param sim_data:
    :param max_harvest:
    :return: dictionary of np arrays and constants of the right types
    """
    zeros_float_col = np.zeros(len(port.df_stocks.index), dtype=float)
    ones_float_col = np.ones(zeros_float_col.shape)

    data_dict = {'shares': port.df_stocks['shares'].values.astype(float),
                 'price': port.df_stocks['price'].values.astype(float),
                 'w_tgt': port.df_stocks['w_tgt'].values.astype(float),
                 'xret': zeros_float_col,
                 'beta': ones_float_col,
                 'port_val': port.port_value,
                 'cash': port.cash,
                 'max_sell': port.lot_stats['max_sell'],
                 'tax_per_shr': port.lot_stats['tax_per_shr'],
                 'cum_shrs': port.lot_stats['cum_shrs'],
                 'tkr_broadcast_idx': port.lot_stats['tkr_broadcast_idx'],
                 'tkr_block_idx': port.lot_stats['tkr_block_idx'],
                 'trx_cost': 0,
                 'cov_matrix': sim_data['cov_matrix'].values.astype(float),
                 'crra': sim_data['params']['CRRA'],
                 'alpha_horizon': 0,
                 'min_w_cash': 0}

    return data_dict


#@njit
def tax_lots(trades: np.array, max_sell: np.array, tax_per_shr: np.array, cum_shrs: np.array,
             tkr_broadcast_idx: np.array, tkr_block_idx: np.array) -> float:
    """ Calculate tax liability for a trade
    """

    if (trades >= 0).all():
        return 0

    # If for at least one stock we are trying to sell more than allowed
    if (-trades > max_sell + np.finfo(float).eps).any():
        return INFINITY

    # Broadcast shares across all lots
    shrs_for_lots = trades[tkr_broadcast_idx]

    # Calc shares sold from each lot
    cum_sold = -np.maximum(shrs_for_lots, -cum_shrs)
    sold = flm.intervaled_diff(cum_sold, tkr_block_idx)

    # No tax on purchases
    sold_only = np.maximum(0, sold)

    total_tax = sold_only @ tax_per_shr

    return total_tax


#@njit
def obj_func_w_lots(trades: np.array, shares: np.array, price: np.array, w_tgt: np.array, xret: np.array,
                    trx_cost: float, port_val: float, alpha_horizon: float, cov_matrix: np.array,
                    max_sell: np.array, tax_per_shr: np.array, cum_shrs: np.array,
                    tkr_broadcast_idx: np.array, tkr_block_idx: np.array,
                    crra, risk_adj=True) -> float:
    """ Objective function for the optimization

        Parameters:
            trades   - iterable (list) of trades that matches tickers
            (args)   - other arguments describing the state of the portfolio, all np arrays

        Return:
            objective function that takes into account risk, return and tax impact
    """

    # Calculate post-rebalance portfolio metrics
    new_shrs = shares + trades

    # Transactions costs
    trx_costs = np.abs(trades) * price * trx_cost
    tot_trx_cost = trx_costs.sum()
    # print(f'Trans. costs = {tot_trx_cost}')

    # New weights and active weights
    new_port_val = port_val - tot_trx_cost
    new_w = new_shrs * price / new_port_val
    new_w_actv = new_w - w_tgt

    # Expected Return to the objective function (all in $)
    exp_pnl = xret @ new_w_actv * new_port_val * alpha_horizon
    # print(f'\nCalled numpy version:\nExp PnL = {exp_pnl}')

    # Risk-adjustment
    ann_factor = 252
    if risk_adj:
        risk_cost = 0.5 * crra * new_port_val * ann_factor * new_w_actv.T @ cov_matrix @ new_w_actv
    else:
        risk_cost = 0
    # print(f'Risk Cost = {risk_cost}')

    # Calculate tax liability
    tax = tax_lots(trades, max_sell, tax_per_shr, cum_shrs, tkr_broadcast_idx, tkr_block_idx)

    # Objective function (Risk-Adj After Tax Returns)
    ra_at_ret = exp_pnl - risk_cost - tot_trx_cost - tax

    return -ra_at_ret  # Change sign, since we want to maximize, using scipy.minimize
