"""
Config file
"""
import datetime
import numpy as np

# Annualization factor
ANN_FACTOR = 240
EMBA_IRR_Factor = 252 / 240
LT_TAX_DAYS = 252

# USE_EMBA_LT_CUT_OFF = True  # Tried to match lt_classification from EMBA but it's too messy

# Position tolerance
tol = 1e-10

# Max move allowed % drop to be incuded into the sample
max_drop = 0.999
max_rise = 99  # this is recommended to avoid having index with very large single stock weights.

# Random weights or use the first row of the input dataframe
# RAND_WEIGHTS = True


# -----------------------------
# Simulation parameters
# -----------------------------
DEBUG_MODE = True
VERBOSE_FREQ = 20  # frequency of printouts of simulation stats (in terms of # steps)

# Simulation range (crop the original data to only use this simulation range)
CROP_RANGE = False
t_start = datetime.date(2002, 6, 1)  # Start of the simulation
t_end = datetime.date(2022, 6, 3)  # Final date of the simulation

# Tax Rates
# tax = {'st': 0., 'lt': 0}
tax = {'st': 0.50, 'lt': 0.28}
# tax['st'] = tax['lt']
tax['div'] = tax['lt']
tax['int'] = tax['st']

# Commissions
trx_cost = 0

# Risk-free rate
int_rate = 0.02

# Random seed for numpy
np.random.seed(7)

# Max iterations in the optimizer
maxiter = 100

# Parameters for the 3-stage harvest
max_lot_prc = 0.25  # Max lot size as % of max_harvest, split lots that are over this size
prc_3step_hvst = 0.4  # stocks to earmark for 3-stage harvest as % of total harvest candidates
prc_3step_buy = 0.5  # buy

# Parameters of the heuristic with cash optimizer
threshold_step_prc = 0.05  # fraction of lots that we add at each iteration
MAX_HVST_DFLT = 0.6

# Data location for market data simulator
sim_code = 'mkt_20y'

# EQ_ALLOC_PICKLE = "eq_alloc.pickle"  # Use momentum based on 12m return
MOMENTUM_SIG_TYPE = 'TRAIL_EX_1'
if MOMENTUM_SIG_TYPE == 'TRAIL_EX_1':
    EQ_ALLOC_PICKLE = "eq_alloc_12_1.pickle"  # Use momentum based on 12m return excluding the latest month
else:
    EQ_ALLOC_PICKLE = "eq_alloc.pickle"  # Use momentum based on 12m return

if sim_code == 'test_10':
    WORKING_DIR = '../data/test_data_10'
    # WORKING_DIR = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/test_data_10/"
    PX_PICKLE = "prices.pickle"
    TR_PICKLE = "t_rets.pickle"
    W_PICKLE = "daily_w.pickle"

elif sim_code == 'mkt_clean':
    WORKING_DIR = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp_clean/"
    PX_PICKLE = "idx_prices.pickle"
    TR_PICKLE = "idx_t_rets.pickle"
    W_PICKLE = "idx_daily_w.pickle"

elif sim_code == 'mkt_full':
    WORKING_DIR = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
    PX_PICKLE = "idx_prices.pickle"
    TR_PICKLE = "idx_t_rets.pickle"
    W_PICKLE = "idx_daily_w.pickle"

elif sim_code == 'mkt_20y':
    WORKING_DIR = '../data/mkt_data_20y'
    PX_PICKLE = "idx_prices.pickle"
    TR_PICKLE = "idx_t_rets.pickle"
    W_PICKLE = "idx_daily_w.pickle"


else:
    raise ValueError(f"Uknown sim_code={sim_code}")
