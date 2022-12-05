"""
Test a function to crop a dataframe
"""
import pytest as pt
import pandas as pd
import datetime
from pathlib import Path
from load_mkt_data import crop_dframe
from pandas.testing import assert_frame_equal


@pt.fixture
def prices():
    # Load a sample data frame
    working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
    px_pickle = "idx_prices.pickle"

    source = Path(working_dir) / px_pickle

    px = px = pd.read_pickle(source).fillna(method='ffill').fillna(0)

    return px


def test_no_range_info(prices):
    px1 = crop_dframe(prices)
    assert_frame_equal(prices, px1)


def test_only_start_date(prices):

    t_start = datetime.date(2000, 1, 1)
    px1 = crop_dframe(prices, t_start=t_start)

    assert len(px1) == 5640
    assert px1.index[0] == datetime.date(2000, 1, 3)
    assert px1.index[-1] == prices.index[-1]


def test_only_end_date(prices):
    t_end =  datetime.date(2001, 1, 1)
    px1 = crop_dframe(prices, t_end=t_end)
    assert len(px1) == 2020
    assert px1.index[0] == prices.index[0]
    assert px1.index[-1] == datetime.date(2000, 12, 29)


def test_both_start_end_dates(prices):
    t_start = datetime.date(2000, 1, 1)
    t_end = datetime.date(2001, 1, 1)

    px1 = crop_dframe(prices, t_start=t_start, t_end=t_end)
    assert len(px1) == 252
    assert px1.index[0] == datetime.date(2000, 1, 3)
    assert px1.index[-1] == datetime.date(2000, 12, 29)
