# -*- coding: utf-8 -*-
""" Program to test single-step rebalance
Created on Fri Sun 30 October 2022
@author: vragulin
"""
import copy
import datetime as dt
from typing import Optional, Any

import pandas as pd

import numpy as np
import pytest
import pytest as pt

import config
from port_lots_class import PortLots
from heur_w_cash import heuristic_w_cash
from test_port_lot_class import port, uport

MAX_HARVEST = 0.6


@pt.fixture
def sim_data() -> dict:
    # Initialize market data
    px_dict = {
        'AAPL': 147.27,
        'BRK-B': 282.51,
        'GOOG': 101.48,
        'GOOGL': 101.13,
        'JNJ': 168.71,
        'META': 130.01,
        'MSFT': 242.12,
        'TSLA': 214.44,
        'UNH': 533.73,
        'V': 190.37}

    prices = np.array(list(px_dict.values())).reshape(len(px_dict), 1)

    weights = np.ones(len(prices)) / len(prices) * 0

    dates = [dt.date(2022, 10, 23)]

    params = {'harvest_thresh': -0.02,
              'max_active_wgt': 0.05}

    sim_data = {
        'dates': dates,
        'w_arr': weights,
        'px_arr': prices.T,
        'trx_cost': 0,
        'tax': config.tax,
        'params': params
    }
    return sim_data


@pt.fixture
def uport1(uport: PortLots) -> PortLots:
    """ uport without 'no buy / no sell restrictions """
    out = uport.copy()

    out.no_sells *= False
    out.df_stocks['no_sells'] = False

    out.no_buys *= False
    out.df_stocks['no_buys'] = False

    return out


def test_port_size(uport: PortLots):
    # Check that the loaded portfolio has the correct size

    nstocks = len(uport.df_stocks)
    nlots = len(uport.df_lots)

    assert (nstocks, nlots) == (10, 50)


def test_heur_w_cash_full_sale_with_no_sell_stks(uport: PortLots, sim_data: dict):
    """ This should throw exception, since there are 'no-sell' stocks
        and it's not possible to sell everything"""
    t = 0
    print()
    with pytest.raises(ValueError):
        opt_res = heuristic_w_cash(uport, t, sim_data, MAX_HARVEST)


def test_heur_w_cash_full_sale(uport1: PortLots, sim_data: dict):
    t = 0
    log = False
    if log: print()
    opt_res = heuristic_w_cash(uport1, t, sim_data, MAX_HARVEST, log=log)
    expected_AAPL = -102478
    expected_all = -180017 # sell all except 'no_sell'
    assert pt.approx(opt_res['opt_trades']['AAPL']) == expected_AAPL
    assert pt.approx(opt_res['opt_trades'].sum()) == expected_all


@pytest.mark.parametrize("cash_tgt", [0, 0.25, 0.5, 0.75, 1])
def test_goto_tgt_cash_wgt(uport1: PortLots, sim_data: dict, cash_tgt: float):
    """ Rebalance to achieve a target cash weight """

    # Set stock weights to add to 50%
    sim_d = copy.deepcopy(sim_data)
    n_stocks = len(sim_d['w_arr'])
    sim_d['w_arr'] = np.ones(n_stocks) / n_stocks * (1 - cash_tgt)

    # Update portfolio
    port = copy.deepcopy(uport1)
    port.update_sim_data(sim_d)

    # Start rebalance
    t = 0
    log = False
    if log: print()
    opt_res = heuristic_w_cash(port, t, sim_d, MAX_HARVEST, log=log)

    # Update portfolio
    rebal_res = port.rebal_sim(trades=opt_res['opt_trades'], sim_data=sim_d, t=t)
    w_cash = port.cash / port.port_value
    assert pt.approx(w_cash) == cash_tgt


# @pytest.mark.parametrize("cash_tgt", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("cash_tgt", [0, 0.25, 0.5, 0.75, 1])
def test_goto_tgt_cash_wgt_only_ST_gains(uport1: PortLots, sim_data: dict, cash_tgt: float) -> None:
    """  Test that the algo still works even if we we ony have short-term gains
         which normally receive lower priority
         Run this for different cash target values like the test above
    """

    # Set stock weights to add to (1-cash_tgt)
    sim_d = copy.deepcopy(sim_data)
    n_stocks = len(sim_d['w_arr'])
    sim_d['w_arr'] = np.ones(n_stocks) / n_stocks * (1 - cash_tgt)

    # Update portfolio
    port = copy.deepcopy(uport1)

    # Change lot dates so that all gains are short-term
    buy_date = sim_d['dates'][0] + dt.timedelta(-10)
    df_lots: pd.DataFrame = port.df_lots
    df_lots['start_date'] = np.where(df_lots['%_gain'] >= 0, buy_date,
                                     pd.to_datetime(df_lots['start_date'].values).date)
    port.update_sim_data(sim_d)

    # Start rebalance
    t = 0
    log = False
    if log: print()
    opt_res = heuristic_w_cash(port, t, sim_d, MAX_HARVEST, log=log)

    # Update portfolio
    rebal_res = port.rebal_sim(trades=opt_res['opt_trades'], sim_data=sim_d, t=t)
    w_cash = port.cash / port.port_value
    assert pt.approx(w_cash) == cash_tgt


