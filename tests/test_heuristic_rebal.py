# -*- coding: utf-8 -*-
""" Program to test single-step rebalance
Created on Fri Sun 30 October 2022
@author: vragulin
"""

import datetime as dt

import numpy as np
import pytest as pt

import config
from port_lots_class import PortLots
from sim_one_path import heuristic_rebalance
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

    weights = np.ones(len(prices)) / len(prices)

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


def test_port_size(uport: PortLots):
    # Check that the loaded portfolio has the correct size

    nstocks = len(uport.df_stocks)
    nlots = len(uport.df_lots)

    assert (nstocks, nlots) == (10, 50)


def test_heuristic_rebal(uport: PortLots, sim_data: dict):
    t = 0
    opt_res = heuristic_rebalance(uport, t, sim_data, MAX_HARVEST)
    expected_AAPL = -66313.11883275617
    assert pt.approx(opt_res['opt_trades']['AAPL']) == expected_AAPL

