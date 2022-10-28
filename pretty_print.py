# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 20:41:02 2022
Functions to print dataframes in a pretty way
@author: vragu
"""

import datetime as dt
import numpy as np
import pandas as pd


# %% Format a dataframe for printing
def df_to_format(df, formats=None, multipliers=None) -> pd.DataFrame:
    """ Print dataframe to string, individual formats for each column
        Parameters:
            df - pandas dataframe
            formats - dictionary {column : format}
                      can specify a default format with '_dflt' key
            multipliers  - conversion multiplier, a dictionary
            
        Result:
            df_out - dataframe, columns specified by the formats dictionary are
                     replaced by strings in correct formats
    """

    # If no formats dictionary specified, return original dataframe
    if formats is None:
        return df

    # Create copy for re-formatting without the loss of data
    dff = df.copy()

    # Multiply columns
    if multipliers is not None:
        for col, mult in multipliers.items():
            if col in dff.columns:
                dff[col] = dff[col] * mult

    # Check with default format has been specified
    dflt_key = '_dflt'
    dflt_given = dflt_key in formats.keys()

    # Iterate over columns, and re-format them one by one
    for col in df.columns:
        if col in formats.keys():
            if formats[col] == 'bool_shrt':
                dff[col] = dff[col].apply(lambda x: '*' if x else '')
            elif formats[col] == 'date_shrt':
                dff[col] = dff[col].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, dt.date) else '')
            else:
                dff[col] = dff[col].apply(lambda x: formats[col].format(x))
        else:
            if dflt_given:
                if formats[dflt_key] == 'bool_shrt':
                    dff[col] = dff[col].apply(lambda x: '*' if x else '')
                else:
                    dff[col] = dff[col].apply(lambda x: formats[dflt_key].format(x))

    return dff


# %% Entry point
if __name__ == "__main__":
    df = pd.DataFrame(np.random.random((5, 5)), columns=list('abcde'))
    df['f'] = [True, True, False, True, False]
    df['g'] = dt.date(2022, 10, 22)
    df.loc[0,'g'] = 0

    print("Unformatted dataframe")
    print(df)

    formats = {'a': '{:.2f}',
               'b': '{:,.0f}',
               'c': '${:.1f}%',
               'f': 'bool_shrt',
               'g': 'date_shrt',
               '_dflt': '_{}_'}

    multipliers = {'b': 1e6,
                   'c': 1000}

    dff = df_to_format(df, formats=formats, multipliers=multipliers)
    print("\nFormatted Dataframe")
    print(dff)
