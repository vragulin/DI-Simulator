# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 2022

@author: vragulin
"""
import pandas as pd


def load_port_lots_file(file_path):
    """ Load portfolio information into a dataframe
        Parameters:
            fname - portfolio name.  Program will look in DATA_DIR for a file
                    called {fname}.xlsx
        Returns:
            stocks - pandas dataframe with stock info (one row per stock)
            lots   - pandas dataframe with lot info (one row per lot)
            params - a dictionary of parameters
    """

    # Load stock-specific information
    stocks = pd.read_excel(file_path, sheet_name='stocks', index_col='Ticker')
    stocks.sort_index(inplace=True)

    # Load lot-specific information
    lots = pd.read_excel(file_path, sheet_name='lots', index_col='Id', parse_dates=True)
    lots['Start Date'] = pd.to_datetime(lots['Start Date'])

    params = pd.read_excel(file_path, sheet_name='params', index_col='Parameter')

    return stocks, lots, params


# %% Testing
if __name__ == "__main__":
    port_fpath = "inputs/port_lot_sp10.xlsx"

    stocks, lots, params = load_port_lots_file(port_fpath)

    for label, df in zip(['stocks', 'lots', 'params'], [stocks, lots, params]):
        print(f"\nLoaded {label}:")
        print(df)

    print('Done')
