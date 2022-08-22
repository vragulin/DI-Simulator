"""
Config file
"""
import datetime
import numpy as np

# Annualization factor
ANN_FACTOR = 240

# Position tolerance
tol = 1e-10

# Max move allowed % drop to be incuded into the sample
max_drop = 0.999
max_rise = 50  # this is recommended to avoid having index with very large single stock weights.

# Final date of the simulation
t_last = datetime.date(2022, 8, 1)

# Tax Rates
tax = {'st': 0.50, 'lt': 0.28}

# Commissions
trx_cost = 0

# Random seed for numpy
np.random.seed(7)

# Data location for market data simulator
working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
PX_PICKLE = "idx_prices.pickle"
TR_PICKLE = "idx_t_rets.pickle"
W_PICKLE = "idx_daily_w.pickle"

