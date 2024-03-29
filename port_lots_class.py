# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 19:47:48 2022
Class for Portfolio data
@author: vragu
"""
from __future__ import annotations

import datetime as dt
from typing import Optional
from copy import deepcopy
from enum import Flag

import pandas as pd
import numpy as np

# Directory with on-period optimizer codebase
import config
import fast_lot_math as flm
from dateutil.relativedelta import relativedelta
from pretty_print import df_to_format


# Class for Lot Disposal Methods
class DispMethod(Flag):
    LTFO = 1  # Least Tax First Out
    TSST = 2  # Short-term High-to-Low basis, then LT HIFO
    LTFO_ETF = 3  # Least Tax First Out, trade lots opened on the same day together based on the idx
    TSST_ETF = 4  # Least Tax First Out, trade lots opened on the same day together based on the idx


# Class Portfolio
class PortLots:
    # Constant used for large numbers
    infinity = 9.999e15

    def __init__(self, tickers: list, **kwargs) -> None:
        """ Constructor
        """

        self.tickers = tickers
        self.port_value = None

        # If target weights are specified (as a list), load them.
        # Otherwise, use equal weights and zero cash
        w_tgt = kwargs.get('w_tgt', None)
        if w_tgt is not None:
            if isinstance(w_tgt, pd.Series):
                self.w_tgt = w_tgt.values
            else:
                self.w_tgt = w_tgt
        else:
            self.w_tgt = [1 / len(tickers)] * len(tickers)

        self.cash_w_tgt = 1.0 - sum(self.w_tgt)

        # Load trade date
        self.t_date = kwargs.get('t_date', dt.datetime.today())
        # Load current cash balance
        self.cash = kwargs.get('cash', 0)

        # Load lots
        self.df_lots = kwargs.get('lots', None)
        if self.df_lots is not None:
            self.df_lots.rename(columns={'Ticker': 'ticker', 'Start Date': 'start_date',
                                         'Shares': 'shares', 'Basis per Share': 'basis'},
                                inplace=True)

            self.df_lots.sort_values('ticker', inplace=True)

        # Load no buy/no sell / long term parameters
        for kw in ['no_buys', 'no_sells']:
            kw_input = kwargs.get(kw, None)
            if kw_input is not None:
                # If we were given a series for all ticker (dense format) just use it
                if isinstance(kw_input, pd.Series):
                    kw_val = (kw_input.values == 1)
                else:
                    # Expand sparse format into dense for all tickers
                    kw_val = [ticker in kw_input for ticker in tickers]
            else:
                kw_val = [False] * len(tickers)
            setattr(self, kw, kw_val)

        # Dataframe representation of the portfolio
        if self.df_lots is not None:
            self.df_stocks = self.build_df_stocks()
        else:
            self.df_stocks = None

        # A placeholder for lot-related data used for fast tax calculation
        self.lot_stats = {}

    @classmethod
    def init_portfolio_from_dict(cls, sim_data: dict) -> PortLots:
        """ Initialize simulation portfolio, establish positions matching the index """

        tickers = sim_data['tickers']

        if len(sim_data['w'].shape) == 1:
            w_tgt = sim_data['w_arr']
        else:
            w_tgt = sim_data['w_arr'][0, :]

        cash = 1 - w_tgt.sum()
        t_date = sim_data['dates'][0]

        # Initial purchase lots (for now ignore transactions costs)
        lots = pd.DataFrame(tickers, columns=['Ticker'])
        lots['Start Date'] = t_date
        lots['Shares'] = w_tgt
        lots['Basis per Share'] = 1

        # Instantiate the portfolio
        port = cls(tickers, w_tgt=w_tgt, cash=cash, lots=lots, t_date=t_date)
        return port

    @classmethod
    def init_from_db(cls, lots: pd.DataFrame, w_tgt: pd.Series, **kwargs) -> PortLots:
        """ Build a PortLots instance from the database data
        :param lots:  position info
        :param w_tgt:  target weight
        :return: PortLots structure
        """

        # Generate a ticker list for the optimization - the union of existing positiosn + new basket
        old_tickers = set(lots['bbg_ticker'])
        new_tickers = set(w_tgt.index)
        tickers = old_tickers | new_tickers
        tickers.discard('Cash')
        ticker_list = sorted(list(tickers))
        port = PortLots(tickers=ticker_list)

        # Set target weights
        w_tgt_series = pd.Series(0, index=ticker_list)
        w_tgt_series.update(w_tgt)
        port.w_tgt = w_tgt_series.values
        port.cash_w_tgt = 1.0 - sum(port.w_tgt)

        # Set trade date
        port.t_date = kwargs.get('t_date', dt.datetime.today())

        # Set cash balance
        port.cash = lots[lots['bbg_ticker'] == 'Cash']['quantity'].values[0]

        # Set lots
        df_lots = lots[['bbg_ticker', 'start_date', 'quantity', 'trade_px']]
        df_lots = df_lots[df_lots['bbg_ticker'] != 'Cash'].reset_index(drop=True)
        df_lots.columns = ['ticker', 'start_date', 'shares', 'basis']
        port.df_lots = df_lots

        # Position-level view
        port.df_stocks = port.build_df_stocks()

        # A placeholder for lot-related data used for fast tax calculation
        port.lot_stats = {}
        for field in ['exp_r_actv', 'sig_active', 'beta_active']:
            setattr(port, field, 0)

        return port

    def build_df_stocks(self) -> pd.DataFrame:

        """ Represent portfolio as a dataframe"""
        columns = ['tickers', 'w_tgt', 'no_buys', 'no_sells']
        df = pd.DataFrame(data=None, columns=columns)
        for column in columns:
            df[column] = getattr(self, column, None)

        df.set_index('tickers', inplace=True)

        # Add a column with the total number of share
        df['shares'] = self.df_lots.groupby('ticker')['shares'].sum()

        return df

    def __str__(self) -> str:
        """ Pretty print details of the portfolio"""

        output = '*' * 17 + '\n'
        output += "Portfolio Details\n"
        output += '*' * 17 + '\n'
        # Print out a dataframe with portfolio info
        # Specify format for the stocks matrix
        fmt_df = {'w_tgt': '{:.1f}',
                  'w': '{:.1f}',
                  'w_actv': '{:.1f}',
                  'shares': '{:,.0f}',
                  'no_buys': 'bool_shrt',
                  'no_sells': 'bool_shrt',
                  'mkt_vals': '{:,.0f}',
                  '_dflt': '{:.2f}'}

        mult_df = {'w': 100,
                   'w_tgt': 100,
                   'w_actv': 100,
                   'alpha': 100,
                   'xret': 100,
                   'vols': 100,
                   'exp_r_actv': 100}

        output += "\nStock-level view:\n"
        output += repr(df_to_format(self.df_stocks, formats=fmt_df, multipliers=mult_df).T) + "\n\n"

        # If portfolio has been revalued, calculate cash and target cash in percentage and abs terms, else leave NA
        try:
            cash_w = self.cash / self.port_value
            tgt_cash = self.cash_w_tgt * self.port_value
            xs_cash = self.cash - tgt_cash
            w_xs_cash = xs_cash / self.port_value

            output += f"Cash: {self.cash:,.0f}, Tgt Cash: {tgt_cash:,.0f}, " \
                      f": {xs_cash:,.0f}\n"
            output += f"Cash(%): {cash_w * 100:.1f}, Tgt Cash (%): {self.cash_w_tgt * 100:.1f}, "
            output += f"Excess: {w_xs_cash * 100:.1f}\n"

            output += f"Port Value: {self.port_value:,.0f}\n"
            output += f"Exp. Active Return(%): {self.exp_r_actv * 100:.3f}\n"
            output += f"Tracking Std. Dev (%): {self.sig_active * 100:.3f}\n"
            output += f"Active beta: {self.beta_active:.3f}\n"
        except (AttributeError, TypeError):  # Portfolio has not been revalued
            output += f"Tgt Cash (%): {self.cash_w_tgt * 100:.1f}\n"
            output += "Portfolio has not been revalued\n"

        return output

    def lots_report(self, to_str: bool = True) -> Optional[str]:
        """ Print report of lots """

        # Columns to include in the report
        # rpt_cols = ['ticker', 'start_date', 'shares', 'basis', 'price', '%_gain']
        # Specify format for the lots matrix
        fmt_df = {
            'ticker': '{}',
            'start_date': 'date_shrt',
            '_dflt': '{:.2f}'
        }

        mult_df = {'%_gain': 100}

        output = "\nLots:\n"
        output += repr(df_to_format(self.df_lots, formats=fmt_df, multipliers=mult_df).T) + "\n\n"

        if not to_str:
            print(output)
            return None

        return output

    def copy(self) -> PortLots:
        """ Create a copy of the existing portfoilo
        Return:
            - new portfolio, an idential copy
        """
        new_port = deepcopy(self)
        return new_port

    def split_large_lots(self, max_size: float) -> list:
        """ Split large lots into several pieces, so that we don't have
            very large lots that leave large gaps
            Update the lots portfolio attribute (dataframe) in place.

            :params max_size:  maximum lot size
            :return: list of indices of new lots
        """

        df_lots = self.df_lots
        n_lots = len(df_lots)
        # Identify large lots
        mkt_vals = df_lots['price'] * df_lots['shares']
        idx_large = mkt_vals.index[mkt_vals > max_size]
        n_new_lots = int((mkt_vals // max_size).sum())

        df_new = pd.DataFrame(np.nan, index=range(n_lots, n_lots + n_new_lots),
                              columns=df_lots.columns)
        idx = n_lots

        for i in idx_large:
            max_shares = max_size / df_lots.loc[i, 'price']
            for j in range(int(mkt_vals[i] // max_size)):
                df_new.loc[idx] = df_lots.loc[i]
                df_new.loc[idx, 'shares'] = max_shares
                df_lots.loc[i, 'shares'] -= max_shares
                idx += 1

        self.df_lots = pd.concat([df_lots, df_new], axis=0)
        return list(range(n_lots, n_lots + n_new_lots))

    def process_lots(self, taxes: dict, t_date: dt.datetime,
                     method: Optional[DispMethod] = None,
                     sim_data: Optional[dict] = None) -> None:
        """ Pre-process lots and create the necessary np arrays for fast calculation of taxes
            Equivalent to the LotsLiability.__init__()

            :param taxes: dictionary with tax rates
            :param t_date: rebalance date
            :param method: disposal method for lots.  If None use LTFO
            :param sim_data: simulation data dict if the method needs index or other data
            :return:  sorts lots in the right order (of disposal) and adds tax an other columns
        """
        # Alias key variables for brevity
        df_stocks = self.df_stocks
        df_lots = self.df_lots

        # Ensure that we have at least 1 lot per stock.  If for some stocks there are none,
        # add dummy lots with zero positions
        stk_not_in_lots = set(df_stocks.index).difference(set(df_lots['ticker'].values))
        if stk_not_in_lots:
            dummy_lots = pd.DataFrame(stk_not_in_lots, columns=['ticker'])
            dummy_lots['start_date'] = t_date
            dummy_lots['shares'] = 0
            dummy_lots['basis'] = np.finfo(float).eps
            dummy_lots['%_gain'] = np.finfo(float).max
            df_lots = pd.concat([df_lots, dummy_lots], axis=0)

        # Sort lots by ticker
        n_lots = len(df_lots)
        df_lots.sort_values(by='ticker', inplace=True)
        df_lots.reset_index(drop=True, inplace=True)

        # Calculate index of the first elements of each group of stocks
        tkr_block_idx = np.ravel(flm.block_start_index_pd(df_lots, col='ticker'))
        tkr_block_sizes = np.diff(tkr_block_idx, append=n_lots)

        # Index to broadcast stock data across lots
        tkr_broadcast_idx = flm.build_broadcast_index(tkr_block_sizes)

        # Add info about restrictions to the lots dataframe
        df_lots['no_sells'] = df_stocks['no_sells'].values[tkr_broadcast_idx]
        df_lots['no_buys'] = df_stocks['no_buys'].values[tkr_broadcast_idx]

        # Calculate maximum number of shares we can sell
        df_lots['shrs_4_sale'] = df_lots['shares'] * (1 - df_lots['no_sells'])
        df_stocks['max_sell'] = df_lots.groupby('ticker')['shrs_4_sale'].sum()

        # Classify lots into long-term and short-term
        lt_cutoff = t_date + relativedelta(years=-1)
        df_lots['long_term'] = df_lots['start_date'] <= np.datetime64(lt_cutoff)

        # Tax rate applicable to different lots
        df_lots['tax rate'] = np.where(df_lots['long_term'], taxes['lt'], taxes['st'])

        # Add a column with prices that we will need for calculations
        df_lots['price'] = df_stocks['price'].values[tkr_broadcast_idx]

        # Tax per share
        df_lots['tax per shr'] = np.where(df_lots['no_sells'] != 1,
                                          df_lots['tax rate'] * (df_lots['price'] - df_lots['basis']),
                                          df_lots['price'])

        #  Add index reference as of lot opening date if needed
        if method is None: method = DispMethod.LTFO

        #  Depending on the disposal method sort lots in order we will be selling them
        if method == DispMethod.LTFO:
            #  Sort by tax per share
            df_lots.sort_values(by=['ticker', 'tax per shr'], inplace=True)

        elif method == DispMethod.TSST:
            # Fido's Time Sensitive Short-Term disposal method
            df_lots.sort_values(by=['ticker', 'long_term', 'basis'], ascending=[True, True, False],
                                inplace=True)

        elif method in DispMethod.TSST_ETF | DispMethod.LTFO_ETF:
            # Allocte lots based on open date and index value at that time (as if we trade an ETF)
            df_idx = pd.DataFrame(sim_data['idx_vals'], index=sim_data['dates'], columns=['idx_val'])
            df_lots['idx_basis'] = df_lots['start_date'].apply(lambda x: df_idx.loc[x, 'idx_val'])

            if method == DispMethod.TSST_ETF:
                try:
                    df_lots.sort_values(by=['ticker', 'long_term', 'idx_basis'], ascending=[True, True, False],
                                        inplace=True)
                except KeyError:
                    raise KeyError("df_lots should contain columns ['ticker', 'long_term', 'idx_basis']")

            else:  # method == DispMethod.LTFO_ETF:
                idx_price = df_idx.loc[t_date, 'idx_val']
                df_lots['idx_tax_per_shr'] = np.where(df_lots['no_sells'] != 1,
                                                      df_lots['tax rate'] * (idx_price - df_lots['idx_basis']),
                                                      idx_price)

                df_lots.sort_values(by=['ticker', 'idx_tax_per_shr'], inplace=True)

        else:
            raise NotImplementedError(f"Method = {method} not implemented")

        # Calculate column with cumsum for each ticker block (to be use for tax calcs)
        df_lots['cum_shrs'] = flm.intervaled_cumsum(df_lots['shares'].values, tkr_block_idx)

        # Extract data needed for future calculations as np arrays
        self.lot_stats['lots_sorted'] = True
        self.lot_stats['lot_ids'] = df_lots.index
        self.lot_stats['tax_per_shr'] = df_lots['tax per shr'].values.astype(float)
        self.lot_stats['cum_shrs'] = df_lots['cum_shrs'].values.astype(float)
        self.lot_stats['price'] = df_lots['price'].values.astype(float)
        self.lot_stats['max_sell'] = df_stocks['max_sell'].values.astype(float)
        self.lot_stats['tkr_block_idx'] = tkr_block_idx
        self.lot_stats['tkr_block_sizes'] = tkr_block_sizes
        self.lot_stats['tkr_broadcast_idx'] = tkr_broadcast_idx

        # Update the lots table in the portfolio structure
        self.df_lots = df_lots

    # **************************************************
    # Functions used for simplified simulation
    # **************************************************
    def update_sim_data(self, sim_data=None, t=0):
        """ Update portfolio when we are doing a simplified simulation
            Don't worry about details like 'alpha', 'beta', etc. for now """

        # Update value date
        self.t_date = sim_data['dates'][t]

        # Update target stock weights and shares
        self.df_stocks['w_tgt'] = sim_data['w_arr'][t]
        self.df_stocks['shares'] = self.df_lots.groupby('ticker')['shares'].sum()

        # Drop df_stocks columns with old data that we will overwrite
        self.df_stocks = self.df_stocks[['w_tgt', 'shares', 'no_buys', 'no_sells']].fillna(0)

        self.df_stocks['price'] = sim_data['px_arr'][t, :]

        # Calculate market values, weights and active positions
        self.df_stocks['mkt_vals'] = self.df_stocks['shares'] * self.df_stocks['price']
        self.port_value = self.df_stocks['mkt_vals'].sum() + self.cash
        self.df_stocks['w'] = self.df_stocks['mkt_vals'] / self.port_value
        self.df_stocks['w_actv'] = self.df_stocks['w'] - self.df_stocks['w_tgt']

        # Add column of prices to the lots dataframe and calculate % gain
        self.df_lots['price'] = self.df_lots['ticker'].apply(lambda x: self.df_stocks.loc[x, 'price'])

        self.df_lots['%_gain'] = np.where(self.df_lots['basis'] > 0, self.df_lots['price']
                                          / self.df_lots['basis'] - 1, 999)

        # Rename and re-arrange columns in a logical order
        self.df_stocks = self.df_stocks[['w_tgt', 'w', 'w_actv', 'shares', 'price', 'mkt_vals',
                                         'no_buys', 'no_sells']]

        # Update other attributes
        self.w_tgt = self.df_stocks['w_tgt']
        self.cash_w_tgt = 1.0 - self.w_tgt.sum()

    def reval_from_db(self, prices: pd.DataFrame) -> None:
        """ Update position values and populate all market-dependent columns
        :param prices: dataframe with prices fetched from db
        :return:  None (update the PortLots instance in place)
        """

        # Add market prices
        df_stocks = self.df_stocks.copy()
        df_stocks['price'] = prices['value']

        # Check that we have prices for all stocks
        if df_stocks['price'].isnull().any():
            missing_px_stks = list(df_stocks[df_stocks['price'].isnull()].index)
            print("Missing prices for stocks: ", end="")
            print(missing_px_stks)
            raise ValueError("Missing stock prices - see message above.")

        # Calculate market values, weights and active positions
        df_stocks['mkt_vals'] = df_stocks['shares'] * df_stocks['price']
        self.port_value = df_stocks['mkt_vals'].sum() + self.cash
        df_stocks['w'] = df_stocks['mkt_vals'] / self.port_value
        df_stocks['w_actv'] = df_stocks['w'] - df_stocks['w_tgt']

        # Add column of prices to the lots dataframe and calculate % gain
        self.df_lots['price'] = self.df_lots['ticker'].apply(lambda x: df_stocks.loc[x, 'price'])

        self.df_lots['%_gain'] = np.where(self.df_lots['basis'] > 0, self.df_lots['price']
                                          / self.df_lots['basis'] - 1, self.infinity)

        # Rename and re-arrange columns in a logical order and write it back into self
        self.df_stocks = df_stocks[['w_tgt', 'w', 'w_actv', 'shares', 'price', 'mkt_vals',
                                    'no_buys', 'no_sells']]

    def sim_report(self) -> str:
        """ Pretty print details of the simplified simulation portfolio"""

        output = '*' * 17 + '\n'
        output += "Portfolio Details\n"
        output += '*' * 17 + '\n'
        # Print out a dataframe with portfolio info
        # Specify format for the stocks matrix
        fmt_df = {'w_tgt': '{:.1f}',
                  'w': '{:.1f}',
                  'w_actv': '{:.1f}',
                  'shares': '{:,.3f}',
                  'no_buys': 'bool_shrt',
                  'no_sells': 'bool_shrt',
                  'mkt_vals': '{:,.3f}',
                  '_dflt': '{:.2f}'}

        mult_df = {'w': 100,
                   'w_tgt': 100,
                   'w_actv': 100,
                   'alpha': 100,
                   'xret': 100,
                   'vols': 100,
                   'exp_r_actv': 100}

        output += "\nStock-level view:\n"
        df_to_print = self.df_stocks.drop(columns=['no_buys', 'no_sells'])
        output += repr(df_to_format(df_to_print, formats=fmt_df, multipliers=mult_df).T) + "\n\n"

        # If portfolio has been revalued, calculate cash and target cash in percentage and abs terms, else leave NA
        try:
            cash_w = self.cash / self.port_value
            tgt_cash = self.cash_w_tgt * self.port_value
            xs_cash = self.cash - tgt_cash
            w_xs_cash = xs_cash / self.port_value

            output += f"Cash: {self.cash:,.3f}, Tgt Cash: {tgt_cash:,.3f}, Excess: {xs_cash:,.3f}\n"
            output += f"Cash(%): {cash_w * 100:.1f}, Tgt Cash (%): {self.cash_w_tgt * 100:.1f}, "
            output += f"Excess: {w_xs_cash * 100:.1f}\n"

            output += f"Port Value: {self.port_value:,.3f}\n"
            # output += f"Exp. Active Return(%): {self.exp_r_actv * 100:.3f}\n"
            # output += f"Tracking Std. Dev (%): {self.sig_active * 100:.3f}\n"
            # output += f"Active beta: {self.beta_active:.3f}\n"
        except AttributeError:  # Portfolio has not been revalued
            output += f"Tgt Cash (%): {self.cash_w_tgt * 100:.3f}\n"
            output += "Portfolio has not been revalued\n"

        return output

    def rebal_sim(self, trades: pd.Series, sim_data: dict, t: int = 0,
                  method: DispMethod = DispMethod.LTFO) -> dict:
        """ Rebalance simulation"""

        # Lots must be sorted (pre-processed), otherwise we don't know how to do allocations
        assert ((self.lot_stats != {}) and self.lot_stats['lots_sorted']), \
            "Rebalance attempted, before pre-processing lots."

        # trades must be in the same stocks that we own (it's possible to buy new stocks, but we'll implement this later
        assert (trades.index.values == self.df_stocks.index.values).all(), \
            "Trades and Positions series indices are not the same."

        # Check if the rebalance is valid
        if (-trades > self.df_stocks['max_sell'] + np.finfo(
                float).eps * 1e4).any():  # Check within computational tolerance
            print("Some trades exceed max_sell.")
        # assert (-trades <= self.df_stocks['max_sell']).all(), \
        #     "Some trades exceeds max_sell."

        # Update macro variables
        # Calculate post-rebalance portfoio metrics
        trx_cost = (np.abs(trades) * self.df_stocks['price'] * sim_data['trx_cost']).sum()
        self.port_value = self.port_value - trx_cost
        net_buy = trades @ self.df_stocks['price']
        self.cash = self.cash - net_buy - trx_cost

        # =========================================================
        # Update lots - at this stage it does not need to be fast
        # =========================================================
        df_lots = self.df_lots  # set up an alias / view for brevity

        # For sells - we reduce the size
        max_id = df_lots.index.max()
        tkr_block_idx = self.lot_stats['tkr_block_idx']
        df_lots['old_shares'] = df_lots['shares']
        df_lots['to_sell'] = -np.minimum(trades.values[self.lot_stats['tkr_broadcast_idx']], 0)
        df_lots['cum_sold'] = np.minimum(df_lots['to_sell'], df_lots['cum_shrs'])
        df_lots['cum_left'] = df_lots['cum_shrs'] - df_lots['cum_sold']
        df_lots['shares'] = flm.intervaled_diff(df_lots['cum_left'].values, tkr_block_idx)

        # Work out total tqx paid on the sales
        df_lots['sold'] = flm.intervaled_diff(df_lots['cum_sold'].values, tkr_block_idx)
        tax = df_lots['sold'].values @ self.lot_stats['tax_per_shr']

        # Drop rows without positions
        df_lots = df_lots[df_lots['shares'] > config.tol]

        # Append new buy lots to the dataframe
        df_new_lots = pd.DataFrame(trades)
        df_new_lots['price'] = self.df_stocks['price']
        df_new_lots = df_new_lots[trades > config.tol].reset_index()
        if len(df_new_lots) > 0:
            df_new_lots.columns = ['ticker', 'shares', 'price']
            df_new_lots['Id'] = np.arange(len(df_new_lots)) + max_id + 1
            df_new_lots['start_date'] = self.t_date  # TODO - check if this is right
            df_new_lots['basis'] = df_new_lots['price'] * (1 + sim_data['trx_cost'])
            df_new_lots.set_index('Id', inplace=True)

            # Concatenate the old and the new lots
            self.df_lots = pd.concat([df_lots, df_new_lots])

        # Sort
        self.df_lots.sort_values(by=['ticker', 'start_date'], inplace=True)
        self.df_lots.reset_index(drop=True, inplace=True)

        # Update positions in the stocks dataframe, if we sold the entire pos in a stock, set pos=0
        self.df_stocks['shares'] = self.df_lots.groupby('ticker')['shares'].sum()
        self.df_stocks['shares'].fillna(0, inplace=True)

        # Run update_mkt_data to recalculate all new values from positions
        self.update_sim_data(sim_data=sim_data, t=t)

        # Collect and output rebalance stats
        data_dict = {'tax': tax,
                     'trx_cost': trx_cost,
                     'net_buy': net_buy}

        return data_dict

    #    def reset_clock(self, reset_thresh: float = 0, lt_cutoff: Optional[dt.date] = None) -> float:
    def reset_clock(self, reset_thresh: float = 0) -> float:

        """ Sell and buy-back long-term positions so that they can be used again
            for harvesting losses.  Assume that market data has been updated and
            the portfolio has been marked to market.
            For now, Don't worry about integrating with the harvesting/rebalance logic,
            we can explore later if there are synergies.
        """
        df_lots = self.df_lots

        # Update lots long-term eligibility and market data
        # if lt_cutoff is None:
        lt_cutoff = self.t_date + relativedelta(years=-1)

        df_lots['long_term'] = df_lots['start_date'] <= np.datetime64(lt_cutoff)
        df_lots['mv'] = df_lots['shares'] * df_lots['price']

        # Identify reset logs, update basis and start data
        df_lots['reset_indic'] = (df_lots['%_gain'] >= reset_thresh) & df_lots['long_term']
        df_lots['basis'] = np.where(df_lots['reset_indic'], df_lots['price'], df_lots['basis'])
        # df_lots['start_date'] = np.where(df_lots['reset_indic'], self.t_date, df_lots['start_date'])
        df_lots['start_date'] = df_lots['start_date'].where(~df_lots['reset_indic'], self.t_date)

        # Calculate tax incurred in the reset
        reset_tax = np.sum(df_lots['reset_indic'] * df_lots['tax per shr'] * df_lots['shares'])
        return reset_tax

    def liquid_tax(self, liq_pct: float = 1.0) -> float:
        """ Taxes incurred when selling fraction of the portfolio
            Assumes portfolio has been updated for the new market data
            And lots have been processed (i.e. calc tax_per_share)
        """

        assert liq_pct >= 0, f"Liquidation percentage should be positive, {liq_pct} given."

        df_lots = self.df_lots
        tax = df_lots['shares'] @ df_lots['tax per shr']

        return tax * liq_pct

    def harvest_inst_replace(self, t: float, sim_data: dict) -> dict:
        """Harvest all loss lots above threshold and replace them at the same price
            and do proper logging.  Assume that we have updated securities prices and
            portfolio value with the port.update_sim_data() method.
            :param t: rebalance time
            :param sim_data:  dictionary with sim data
            :return: dictionary with simulation data
        """

        # Process lots to speed up calculations
        t_date = sim_data['dates'][t]
        params = sim_data['params']
        self.process_lots(sim_data['tax'], t_date)

        df_lots = self.df_lots
        df_stocks = self.df_stocks

        # Identify loss lots below threshold
        df_lots['below_thresh'] = df_lots['%_gain'] <= params['harvest_thresh']

        # Update start date and cost basis for the harvested lots
        df_lots['basis'] = np.where(df_lots['below_thresh'], df_lots['price'], df_lots['basis'])
        # df_lots['start_date'] = np.where(df_lots['below_thresh'], self.t_date, df_lots['start_date'])
        df_lots['start_date'] = df_lots['start_date'].where(~df_lots['below_thresh'], self.t_date)
        df_lots['harvest_trade'] = np.where(df_lots['below_thresh'], df_lots['shares'], 0)

        # Calculate tax incurred in the reset
        tax = df_lots['below_thresh'] @ (df_lots['tax per shr'] * df_lots['shares'])

        # Insert a column with harvested trades
        df_stocks['harvest_trade'] = df_lots.groupby(by='ticker')['harvest_trade'].sum()

        # Reinvest (excess) cash at the current prices:
        xs_cash = self.cash - self.cash_w_tgt * self.port_value

        # Check that it's not a rounding error, if so doon't create new lots.
        if np.abs(xs_cash / self.port_value) > config.tol:
            factor = xs_cash / (self.port_value - self.cash)
            df_stocks['trade'] = df_stocks['shares'] * factor
        else:
            df_stocks['trade'] = 0

        # Pack trades into the structure
        res = {'opt_trades': df_stocks['trade'], 'potl_harvest': -tax, 'harvest': -tax}

        res['harvest_trades'] = df_stocks['harvest_trade']
        res['harv_ratio'] = 1.0
        res['port_val'] = self.port_value
        res['trx_cost'] = np.abs(df_stocks['harvest_trade']) @ df_stocks['price'] * config.trx_cost

        return res

    def cf_during_period(self, t: int, sim_data: dict, net: bool = True) -> tuple:
        """ Calculate portfolio divs and interest received during the period
            (assume that all stocks just went on ex-div right before rebalance.

            If t=0, there are no cash flows.

            Interest is are based on prior period's interest rate.

            Current positions (cash) are used for both divs (interest),
            in effect, we assume that we have not rebalanced yet.

            Div are calculated by multiplying current %div and position (as of t) by prior period's
            share price.  If this is too constraining, consider adding more parameters (e.g. last period's
            cash).

            :param self: portfolio
            :param t: index of the rebalance date
            :param sim_data: simulation data
            :param net: gross of net cash flows
            :return: (dividends, interest)
        """

        if t > 0:
            params = sim_data['params']
            n_stocks = sim_data['div'].shape[1]

            # Dividend calculation
            stock_divs = sim_data['div'][t, :]
            prices = sim_data['px'].values[t - 1, :]

            shares = self.df_stocks['shares'].values
            port_divs = (shares * prices) @ stock_divs

            # Interest calculation
            port_interest = self.cash * sim_data['cash_ret'].values.ravel()[t - 1]

            if net:
                port_divs *= (1 - config.tax['div'])
                port_interest *= (1 - config.tax['int'])

            return port_divs, port_interest

        else:
            return 0, 0
