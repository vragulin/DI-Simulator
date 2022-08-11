import pandas as pd

# Load market data
# import pytest
# import numpy as np
# from main import load_index_weights
#
#
# @pytest.fixture
# def idx_info():
#     fname = 'inputs/idx_weights_test5.xlsx'
#     df_idx = load_index_weights(fname)
#     return df_idx
#
#
# def test_load_index_weights_right_number(idx_info):
#     assert len(idx_info) == 5
#
#
# def test_load_index_weights_right_index(idx_info):
#     assert list(idx_info.index.values) == ['AAPL', 'BRK-B', 'GOOG', 'TSLA', 'V']
#
#
# def test_load_index_weights_right_values(idx_info):
#     assert list(idx_info['Initial Weight'].values) == [0.3, 0.15, 0.25, 0.2, 0.07]
