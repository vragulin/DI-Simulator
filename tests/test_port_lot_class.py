""" Program to test  the PortLots class
    Written 10-Jul-2022
    @author vragulin
"""

import pandas as pd
import pytest as pt
import datetime as dt
import numpy as np

from load_port_lots_file import load_port_lots_file
from port_lots_class import PortLots


@pt.fixture
def port() -> PortLots:
    """ Load test portfolio """
    port_code = "lot_sp10"
    port_name = f"../inputs/port_{port_code}.xlsx"

    # Load portfolio data from excel
    stocks, lots, params = load_port_lots_file(port_name)

    # Instantiate PortLots
    test_port = PortLots(stocks.index, cash=params.loc['Cash', 'Value'], lots=lots,
                         w_tgt=stocks['Target Weight'], no_buys=stocks['No Buys'],
                         no_sells=stocks['No Sells'])

    return test_port


def test_load1(port: PortLots):
    """ Loaded correct number of stocks """
    assert len(port.df_stocks) == 10


def test_load2(port: PortLots):
    """ Loaded correct number of lots """
    assert len(port.df_lots) == 50


def test_load3(port: PortLots):
    """ Loaded correct number of lots """
    assert port.cash == 2500000


def test_tdate_default(port: PortLots):
    """ Trade date is set to today by default """
    assert port.t_date.date() == dt.datetime.today().date()


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

    sim_data = {
        'dates': dates,
        'w_arr': weights,
        'px_arr': prices.T,
        'trx_cost': 0
    }
    return sim_data

@pt.fixture
def uport(port: PortLots, sim_data: dict):
    # Update portfolio with the market data
    port.update_sim_data(sim_data)
    return port


def test_port_value(uport: PortLots):
    expected = 35506680.33
    assert pt.approx(uport.port_value) == expected


def test_reports(uport: PortLots):
    # Test loading market data and printing reports
    print("\n" + str(uport))

    print("\nLots details")

    print(str(uport.lots_report()))

    port_val = uport.port_value
    expected = 35506680.33
    assert pt.approx(port_val) == expected


def test_copy(uport: PortLots):
    new_port = uport.copy()

    print("\nA copy of the portfolio - no changes")
    print("\n" + str(new_port))

    print("\nLots details")
    print(new_port.lots_report())

    assert pt.approx(uport.cash_w_tgt) == new_port.cash_w_tgt


def test_rebalance_buy(uport: PortLots, sim_data: dict):
    """ Test rebalance method - buy 100 shares of AAPL """
    trades = pd.Series(data=0, index=uport.df_stocks.index)
    trades['AAPL'] = 100
    old_shrs = uport.df_stocks.loc['AAPL', 'shares']

    tax_rates = {'lt': 0.28, 'st': 0.5}
    uport.process_lots(tax_rates, uport.t_date)

    data_dict = uport.rebal_sim(trades, sim_data)
    assert pt.approx(old_shrs + 100) == uport.df_stocks.loc['AAPL', 'shares']
    assert pt.approx(data_dict['tax']) == 0


def test_rebalance_sell(uport: PortLots, sim_data: dict):
    """ Test rebalance method - buy 100 shares of AAPL """
    trades = pd.Series(data=0, index=uport.df_stocks.index)
    trades['AAPL'] = -100
    old_shrs = uport.df_stocks.loc['AAPL', 'shares']

    tax_rates = {'lt': 0.28, 'st': 0.5}
    uport.process_lots(tax_rates, uport.t_date)

    data_dict = uport.rebal_sim(trades, sim_data)
    assert pt.approx(old_shrs - 100) == uport.df_stocks.loc['AAPL', 'shares']
    assert pt.approx(data_dict['tax']) == 428.96


def test_split_lots_correct_nlots(uport: PortLots):
    max_size = 3e6

    tax_rates = {'lt': 0.28, 'st': 0.5}
    uport.process_lots(tax_rates, uport.t_date)

    n_lots_old = len(uport.df_lots)

    new_idx = uport.split_large_lots(max_size)
    n_lots_new = len(uport.df_lots)

    assert len(new_idx) == 4
    assert len(new_idx) == (n_lots_new - n_lots_old)


def test_split_lots_total_shrs_unched(uport: PortLots):

    tax_rates = {'lt': 0.28, 'st': 0.5}
    uport.process_lots(tax_rates, uport.t_date)
    aapl_shrs_old = uport.df_stocks.loc['AAPL', 'shares']

    max_size = 1e6
    uport.split_large_lots(max_size)
    uport.process_lots(tax_rates, uport.t_date)
    aapl_shrs_new = uport.df_stocks.loc['AAPL', 'shares']

    assert aapl_shrs_new == aapl_shrs_old


def test_split_lots_total_port_val_unched(uport: PortLots):
    tax_rates = {'lt': 0.28, 'st': 0.5}
    uport.process_lots(tax_rates, uport.t_date)
    port_val_old = uport.port_value

    max_size = 2e6
    uport.split_large_lots(max_size)
    uport.process_lots(tax_rates, uport.t_date)

    assert pt.approx(port_val_old) == uport.port_value


def test_split_lots_tax_unched(uport: PortLots, sim_data: dict):
    # Check that we get the same tax for selling 100 AAPL
    # with and without splitting
    """ Test rebalance method - buy 100 shares of AAPL """
    trades = pd.Series(data=0, index=uport.df_stocks.index)
    trades['AAPL'] = -100

    old_shrs = uport.df_stocks.loc['AAPL', 'shares']

    tax_rates = {'lt': 0.28, 'st': 0.5}
    uport.process_lots(tax_rates, uport.t_date)

    max_size = 2e6
    uport.split_large_lots(max_size)
    uport.process_lots(tax_rates, uport.t_date)

    data_dict = uport.rebal_sim(trades, sim_data)
    assert pt.approx(old_shrs - 100) == uport.df_stocks.loc['AAPL', 'shares']
    assert pt.approx(data_dict['tax']) == 428.96
