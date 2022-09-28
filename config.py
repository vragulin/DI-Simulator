"""
Config file
"""
import datetime
import numpy as np

# Annualization factor
ANN_FACTOR = 240
EMBA_IRR_Factor = 252/240
LT_TAX_DAYS = 252
DEBUG_MODE = True

# Position tolerance
tol = 1e-10

# Max move allowed % drop to be incuded into the sample
max_drop = 0.999
max_rise = 99  # this is recommended to avoid having index with very large single stock weights.

# Random weights or use the first row of the input dataframe
# RAND_WEIGHTS = True

# Final date of the simulation
t_last = datetime.date(2022, 8, 1)

# Tax Rates
#tax = {'st': 0., 'lt': 0}
tax = {'st': 0.50, 'lt': 0.28}
tax['div'] = tax['lt']

# Commissions
trx_cost = 0

# Random seed for numpy
np.random.seed(7)

# Data location for market data simulator
# working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp_clean/"
# PX_PICKLE = "idx_prices.pickle"
# TR_PICKLE = "idx_t_rets.pickle"
# W_PICKLE = "idx_daily_w.pickle"

working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/test_data_10/"
PX_PICKLE = "prices.pickle"
TR_PICKLE = "t_rets.pickle"
W_PICKLE = "daily_w.pickle"

# working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
# PX_PICKLE = "idx_prices.pickle"
# TR_PICKLE = "idx_t_rets.pickle"
# W_PICKLE = "idx_daily_w.pickle"