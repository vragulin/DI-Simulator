"""
Config file
"""
import datetime
import numpy as np

# Annualization factor
ANN_FACTOR = 240
EMBA_IRR_Factor = 252 / 240
LT_TAX_DAYS = 252
DEBUG_MODE = True
# USE_EMBA_LT_CUT_OFF = True  # Tried to match lt_classification from EMBA but it's too messy

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
# tax = {'st': 0., 'lt': 0}
tax = {'st': 0.50, 'lt': 0.28}
tax['div'] = tax['lt']

# Commissions
trx_cost = 0

# Random seed for numpy
np.random.seed(7)

# Maximum lot size (used in 3-step) as percent of max_harvest, split lots that are over this size
max_lot_prc = 0.25  # 0.25

# Max iterations in the optimizer
maxiter = 100

# Parameters for the 3-stage harvest
prc_3step_hvst = 0.4  # stocks to earmark for 3-stage harvest as % of total harvest candidates
prc_3step_buy = 0.5  # buy

# Data location for market data simulator
# working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp_clean/"
# PX_PICKLE = "idx_prices.pickle"
# TR_PICKLE = "idx_t_rets.pickle"
# W_PICKLE = "idx_daily_w.pickle"

# working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/test_data_10/"
# PX_PICKLE = "prices.pickle"
# TR_PICKLE = "t_rets.pickle"
# W_PICKLE = "daily_w.pickle"

# working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
# PX_PICKLE = "idx_prices.pickle"
# TR_PICKLE = "idx_t_rets.pickle"
# W_PICKLE = "idx_daily_w.pickle"
