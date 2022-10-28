import numpy as np
from embaTestbed2022_testing import run_scenario_vr, run_scenario

np.seterr('raise')


def run_test_one_setting(inputs):
    only_base = False
    base_scenario = inputs.copy()

    print('*** Test ***')
    if inputs['replace'] == True:
        inputs['path_file_code'] = 'replace'
    else:
        inputs['path_file_code'] = 'shuffle'

    run_scenario_vr(inputs, only_base=True)

    print('*** Test with Taxes ***')
    taxes = {'tau_div_start': 0.28,
             'tau_div_end': 0.28,
             'tau_st_start': 0.5,
             'tau_st_end': 0.5,
             'tau_lt_start': 0.28,
             'tau_lt_end': 0.28
             }
    inputs = base_scenario.copy()
    inputs.update(taxes)
    inputs['save_path_info'] = False  # No need to save the second time
    run_scenario_vr(inputs, only_base=only_base)

    print('*** Test with Taxes and Terminal Donation ***')
    inputs = base_scenario.copy()
    inputs.update(taxes)
    inputs['terminal_donation'] = 1
    run_scenario_vr(inputs, only_base=only_base)


# -----------------------------
# Entry Point
# -----------------------------

# Build inputs
test = {'dt': 20,
        'tau_div_start': 0.0,
        'tau_div_end': 0.0,
        'tau_st_start': 0.0,
        'tau_st_end': 0.0,
        'tau_lt_start': 0.0,
        'tau_lt_end': 0.0,
        'donate_start_pct': 0.0,
        'donate_end_pct': 0.0,
        'div_reinvest': False,
        'div_payout': True,
        'div_override': 0.02,
        'harvest': 'none',
        'harvest_thresh': -0.02,
        'harvest_freq': 20,
        'clock_reset': False,
        'rebal_freq': 20,
        'donate_freq': 1200,
        'donate_thresh': 0.0,
        'terminal_donation': 0,
        'donate': False,
        'replace': True,
        'randomize': True,
        'return_override': 0.07,
        'N_sim': 25,
        'savings_reinvest_rate': -1,
        'loss_offset_pct': 1,
        'vol_override': 0.16,
        'save_path_info': True
        }

test60 = test.copy()
test60['dt'] = 60
test60['harvest_freq'] = test60['dt']
test60['rebal_freq'] = test60['dt']

test_don25 = test.copy()
test_don25['donate'] = True
test_don25['donate_start_pct'] = test_don25['donate_end_pct'] = 0.25

test_no_random = test.copy()
test_no_random['N_sim'] = 1
test_no_random['randomize'] = False

test_no_random_60 = test_no_random.copy()
test_no_random_60['dt'] = 60
test_no_random_60['harvest_freq'] = test60['dt']
test_no_random_60['rebal_freq'] = test60['dt']

# Test one set of parameters
run_test_one_setting(test)
print("Done")
