# Passive Inputs
run_code = 'mkt_20'

if run_code == 'test_10':
    # testing files
    working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/test_data_10/"
    PX_PICKLE = "prices.pickle"
    TR_PICKLE = "t_rets.pickle"
    W_PICKLE = "daily_w.pickle"

    PATHS_DIR = f"{working_dir}paths/"

elif run_code == 'test_100':
    # testing files
    working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/test_data_100/"
    PX_PICKLE = "prices.pickle"
    TR_PICKLE = "t_rets.pickle"
    W_PICKLE = "daily_w.pickle"

    PATHS_DIR = f"{working_dir}paths/"

elif run_code == 'test_5y_100':
    # testing files
    working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/test_data_5y_100/"
    PX_PICKLE = "prices.pickle"
    TR_PICKLE = "t_rets.pickle"
    W_PICKLE = "daily_w.pickle"

    PATHS_DIR = f"{working_dir}paths/"

elif run_code == 'test_500':
    # testing files
    working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/test_data_500/"
    PX_PICKLE = "prices.pickle"
    TR_PICKLE = "t_rets.pickle"
    W_PICKLE = "daily_w.pickle"

    PATHS_DIR = f"{working_dir}paths/"

elif run_code == 'test_5y_500':
    # testing files
    working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/test_data_5y_500/"
    PX_PICKLE = "prices.pickle"
    TR_PICKLE = "t_rets.pickle"
    W_PICKLE = "daily_w.pickle"

    PATHS_DIR = f"{working_dir}paths/"

elif run_code == 'prod':
    # Production files
    # Which dataset to use - clean (if True) or full (if False)
    clean_flag = True

    if clean_flag:
        working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp_clean/"
    else:
        working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
    PX_PICKLE = "idx_prices.pickle"
    TR_PICKLE = "idx_t_rets.pickle"
    W_PICKLE = "idx_daily_w.pickle"

    PATHS_DIR = r"../data/paths"

elif run_code == 'spx_20':
    # testing files
    working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/mkt_data_spx_20y/"
    PX_PICKLE = "idx_prices.pickle"
    TR_PICKLE = "idx_t_rets.pickle"
    W_PICKLE = "idx_daily_w.pickle"

    PATHS_DIR = f"{working_dir}paths/"

elif run_code == 'mkt_20':
    # testing files
    working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/mkt_data_20y/"
    PX_PICKLE = "idx_prices.pickle"
    TR_PICKLE = "idx_t_rets.pickle"
    W_PICKLE = "idx_daily_w.pickle"

    PATHS_DIR = f"{working_dir}paths/"

elif run_code == 'mkt_15':
    # testing files
    working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DI Sim/data/mkt_data_15y/"
    PX_PICKLE = "idx_prices.pickle"
    TR_PICKLE = "idx_t_rets.pickle"
    W_PICKLE = "idx_daily_w.pickle"

    PATHS_DIR = f"{working_dir}paths/"

else:
    raise NotImplementedError(f"Run code {run_code} not implemented")
