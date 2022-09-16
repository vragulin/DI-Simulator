""" Run the simulation with a small test file so that I can compare results vs.
my simulation step-by-step
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os

# Passive Inputs
# VR - replace this with the name of my directory where i have the files.
working_dir = r"C:/Users/vragu/OneDrive/Desktop/Proj/DirectIndexing/data/overnight_intrp/"
PX_PICKLE = "idx_prices.pickle"
TR_PICKLE = "idx_t_rets.pickle"
W_PICKLE = "idx_daily_w.pickle"


# Active Inputs


# EMBA standalone sandbox
def load_data(data_freq: int, data_dict={}, replace=False, randomize=False, return_override=0) -> dict:
    if not len(data_dict):

        px = pd.read_pickle(os.path.join(working_dir, PX_PICKLE)).fillna(method='ffill').fillna(0)
        px.index = range(0, len(px.index))
        px = px.reindex(index=range(0, len(px.index), data_freq))

        tri = pd.read_pickle(os.path.join(working_dir, TR_PICKLE)).fillna(method='ffill').fillna(0)
        tri.index = range(0, len(tri.index))
        tri = tri.reindex(index=range(0, len(tri.index), data_freq))

        w = pd.read_pickle(os.path.join(working_dir, W_PICKLE)).fillna(0)
        w.index = range(0, len(w.index))
        w = np.maximum(0, w.reindex(index=range(0, len(w.index), data_freq)))
        w /= (np.sum(w.to_numpy(), axis=1)[:, None] @ np.ones((1, len(w.columns))))

        out_dict = {'px': px, 'tri': tri, 'w': w}

        # Build returns
        out_dict['d_px'] = out_dict['px'].pct_change().fillna(0).replace(np.inf, 0)
        out_dict['d_tri'] = out_dict['tri'].pct_change().fillna(0).replace(np.inf, 0)
        out_dict['div'] = out_dict['d_tri'] - out_dict['d_px']
    else:
        out_dict = data_dict

    # Randomize
    if randomize:
        np.random.seed(7)
        # find stocks present for the entire sample and pull out everything else
        full_indic = np.sum((out_dict['px'].to_numpy() > 0), axis=0) == len(out_dict['px'].index)
        d_px = out_dict['d_px'].iloc[:, full_indic].sample(frac=1, replace=replace).reset_index(drop=True)
        rand_weights = np.exp(np.random.normal(size=(1, d_px.shape[1])))
        rand_weights /= np.sum(rand_weights)

        # Get returns approximately equal to override
        if return_override > 0:
            # rand_return = np.log(np.sum(rand_weights * np.product(1+d_px.to_numpy(), axis=0))) / (d_px.shape[0] * data_freq / 252)
            rand_return = (240 / data_freq) * (np.sum(rand_weights *
                                                      np.product(1 + d_px.to_numpy()[1:, :], axis=0)) ** (
                                                           1 / (d_px.shape[0])) - 1)
            d_px += (return_override - rand_return) * (data_freq / 240)

        out_dict['w'] = np.ones((out_dict['w'].shape[0], 1)) @ rand_weights
        out_dict['d_px'] = d_px
        out_dict['div'] = out_dict['div'].iloc[:, full_indic]
        out_dict['px'] = out_dict['px'].iloc[:, full_indic]

    # Vectorize
    for k in ('px', 'd_px', 'd_tri', 'div', 'w'):
        if isinstance(out_dict[k], pd.DataFrame):
            out_dict[k + '_arr'] = out_dict[k].to_numpy()
        else:
            out_dict[k + '_arr'] = out_dict[k]

    return out_dict


def lot_resize(l: list) -> list:
    for i in range(0, len(l)):
        l[i] = np.pad(l[i], ((0, 0), (0, 1)))
    return l


def update_divs(dt: float, t: float, tau_div: float, cf: np.array, mv: np.array, lot_idx: np.array, lot_start: np.array,
                basis: np.array, px: np.array, div: np.array, div_reinvest: bool, div_payout: bool,
                div_override: float = -1) -> tuple:
    # Div CF
    if div_override >= 0 and (dt * t % 260) == 0:
        pct_div = div_override * np.ones(div.shape[1])
    elif div_override >= 0:
        pct_div = np.zeros(div.shape[1])
    else:
        pct_div = div[t, :]
    gross_div = np.sum(mv, axis=1) * pct_div
    net_div = gross_div * (1 - tau_div)
    sum_net_div = np.sum(net_div)
    if div_payout:
        cf[t] += sum_net_div
        cash = 0
    else:
        cash = sum_net_div

    # Update Div Lot info
    if sum_net_div and div_reinvest:
        div_indic = net_div > 0
        lot_idx += div_indic
        lot_resize([basis, mv, lot_start])  # Check this is working by ref
        lot_start[range(0, lot_start.shape[0])] += t * div_indic
        mv[range(0, mv.shape[0]), lot_idx] += net_div * div_indic
        basis[range(0, basis.shape[0]), lot_idx] += net_div * div_indic
        cash = 0

    return float(cash), basis, mv, lot_start


def update_harvest(dt, t, harvest_thresh, tau_st, tau_lt, mv, basis, lot_start, cf, clock_reset=False,
                   savings_reinvest_rate=-1, loss_offset_pct=1) -> tuple:
    gross_gains = mv - basis
    st_indic = dt * (t - lot_start) < 252
    lt_indic = 1 - st_indic
    if not clock_reset:
        harvest_indic = gross_gains / (np.finfo(float).eps + basis) < harvest_thresh
    else:
        harvest_indic = (gross_gains / (np.finfo(float).eps + basis) < harvest_thresh) + \
                        (gross_gains >= harvest_thresh) * lt_indic
    if savings_reinvest_rate > 0:
        T = dt * (len(cf) - t) / 252
        savings = -np.sum(gross_gains * harvest_indic * (st_indic * tau_st + lt_indic * tau_lt)) * \
                  (1 + savings_reinvest_rate) ** T
        cf[-1] += savings * (1 if savings < 0 else loss_offset_pct)
    else:
        savings = -np.sum(gross_gains * harvest_indic * (st_indic * tau_st + lt_indic * tau_lt))
        cf[t] += savings * (1 if savings < 0 else loss_offset_pct)

    # Update lots
    lot_start += (-lot_start + t) * harvest_indic
    basis += (-basis + mv) * harvest_indic

    return basis, lot_start


def update_rebal(dt: float, cash: float, t: float, tau_st: float, tau_lt: float, w: np.array, mv: np.array,
                 lot_idx: np.array, basis: np.array, lot_start: np.array, px: np.array, cf: np.array) -> tuple:
    target_mv = w[t, :] * (np.sum(mv) + cash)
    rebal_trade = target_mv - np.sum(mv, 1)

    # Buys
    buy_indic = rebal_trade > 0
    lot_idx += buy_indic
    (basis, mv, lot_start) = lot_resize([basis, mv, lot_start])
    basis[range(0, basis.shape[0]), lot_idx] += buy_indic * rebal_trade
    mv[range(0, mv.shape[0]), lot_idx] += buy_indic * rebal_trade
    lot_start[range(0, lot_start.shape[0]), lot_idx] += buy_indic * t

    # Sells, assumes LIFO (not totally realistic, but low impact and expedient)
    sell_idx = lot_idx.copy()
    sell_indic = rebal_trade < -1e-10
    cf_value = 0
    gross_gains = mv - basis
    n_pass = 1
    while np.sum(sell_indic * rebal_trade):
        # adjust MVs
        sells = sell_indic * np.maximum(-mv[range(0, mv.shape[0]), sell_idx], rebal_trade)

        # figure out the ST and LT gains
        st_indic = dt * (t - lot_start) < 252
        lt_indic = 1 - st_indic
        st_gains = np.sum(st_indic[range(0, st_indic.shape[0]), sell_idx] * \
                          gross_gains[range(0, st_indic.shape[0]), sell_idx] * \
                          abs(sells / (np.finfo(float).eps + mv[range(0, mv.shape[0]), sell_idx])))

        lt_gains = np.sum(lt_indic[range(0, lt_indic.shape[0]), sell_idx] * \
                          gross_gains[range(0, lt_indic.shape[0]), sell_idx] * \
                          abs(sells / (np.finfo(float).eps + mv[range(0, mv.shape[0]), sell_idx])))

        cf_value += -(st_gains * tau_st + lt_gains * tau_lt)
        basis[range(0, basis.shape[0]), sell_idx] *= 1 - abs(sells / (np.finfo(float).eps +
                                                                      mv[range(0, mv.shape[0]), sell_idx]))
        mv[range(0, mv.shape[0]), sell_idx] += sells
        sell_idx -= 1 * sell_indic
        rebal_trade -= sells
        sell_indic = rebal_trade < -1e-10
        n_pass += 1
    cf[t] += cf_value

    return basis, mv, lot_start


def update_donate(t: float, mv: np.array, basis: np.array, lot_start: np.array, donate_thresh: float,
                  donate_pct: float) -> tuple:
    gains_pct = (mv - basis) / (basis + np.finfo(float).eps)
    mv_tot = np.sum(mv)
    donate_amt = mv_tot
    while (donate_amt / mv_tot) >= donate_pct:
        donate_indic = gains_pct >= donate_thresh
        donate_amt = np.sum(donate_indic * mv)
        donate_thresh += 0.1
    basis += (-basis + mv) * donate_indic
    lot_start += (-lot_start + t) * donate_indic

    return basis, lot_start


def irr_obj(r: float, cf: np.array, dt: int) -> float:
    t = np.linspace(start=0, stop=dt * len(cf), num=len(cf), endpoint=False) / 252
    pv = float(np.sum(cf * np.exp(-r * t)))
    return abs(pv)


def run_sim_path(data_dict: dict, inputs: dict) -> dict:
    # unpack key inputs
    w = data_dict['w_arr']  # [Time, Tkr]
    px = data_dict['px_arr']
    dpx = data_dict['d_px_arr']
    div = data_dict['div_arr']
    n_t = w.shape[0]

    # Go through time
    cf = np.zeros(n_t)  # [Time]
    basis = np.zeros((w.shape[1], 1))  # [Tkr, Lot]
    mv = np.zeros((w.shape[1], 1))  # [Tkr, Lot]
    lot_start = np.zeros((w.shape[1], 1))  # [Tkr, Lot]
    lot_idx = np.zeros(w.shape[1]).astype(int)  # [Tkr]
    cf[0] = -1
    mv[:, 0] = w[0, :]
    basis[:, 0] = mv[:, 0]
    cash = 0

    for t in range(1, n_t):
        # Interp inputs
        tau_div = inputs['tau_div_start'] + (inputs['tau_div_end'] - inputs['tau_div_start']) * t / n_t
        tau_st = inputs['tau_st_start'] + (inputs['tau_st_end'] - inputs['tau_st_start']) * t / n_t
        tau_lt = inputs['tau_lt_start'] + (inputs['tau_lt_end'] - inputs['tau_lt_start']) * t / n_t
        donate_pct = inputs['donate_start_pct'] + (inputs['donate_end_pct'] - inputs['donate_start_pct']) * t / n_t

        # Divs
        cash_div, basis, mv, lot_start = update_divs(inputs['dt'], t, tau_div, cf, mv, lot_idx, lot_start, basis,
                                                     px, div, inputs['div_reinvest'], inputs['div_payout'],
                                                     inputs['div_override'])
        cash += cash_div

        # Update MV
        mv *= 1 + (dpx[t, :, None] @ np.ones((1, mv.shape[1])))

        # Harvest
        if inputs['dt'] * t % inputs['harvest_freq'] == 0:
            if inputs['harvest'] != 'none':
                (basis, lot_start) = update_harvest(inputs['dt'], t, inputs['harvest_thresh'], tau_st, tau_lt, mv,
                                                    basis, lot_start, cf, inputs['clock_reset'],
                                                    inputs['savings_reinvest_rate'], inputs['loss_offset_pct'])

        # Rebal
        if inputs['dt'] * t % inputs['rebal_freq'] == 0 and not inputs['randomize']:
            # print(np.sum((w[t, :] > 0) != (w[t-3, :] > 0)))
            (basis, mv, lot_start) = update_rebal(inputs['dt'], cash, t, tau_st, tau_lt, w, mv, lot_idx, basis,
                                                  lot_start, px, cf)
            cash = 0

        # Donate
        if inputs['dt'] * t % inputs['donate_freq'] == 0 and inputs['donate']:
            (basis, lot_start) = update_donate(t, mv, basis, lot_start, inputs['donate_thresh'], donate_pct)

    # Calc IRR
    st_indic = inputs['dt'] * (t - lot_start) < 252
    lt_indic = 1 - st_indic
    cf[n_t - 1] += np.sum(mv) - (1 - inputs['terminal_donation']) * (np.sum(st_indic * tau_st * (mv - basis)) -
                                                                     np.sum(lt_indic * tau_lt * (mv - basis)))
    x0 = np.asarray([0.09])
    irr = minimize(irr_obj, x0, args=(cf, inputs['dt']), bounds=[(0.01, 0.2)]).x

    return irr


def run_sim(inputs: dict) -> float:
    N = inputs['N_sim']
    irr = np.zeros(N)
    data_dict = load_data(inputs['dt'], replace=inputs['replace'], randomize=inputs['randomize'],
                          return_override=inputs['return_override'])
    for n in range(0, N):
        irr[n] = run_sim_path(data_dict, inputs)
        data_dict = load_data(inputs['dt'], replace=inputs['replace'], randomize=inputs['randomize'],
                              data_dict=data_dict, return_override=inputs['return_override'])
    mean_irr = float(np.mean(irr))

    return mean_irr


def run_scenario(inputs: dict):
    inputs['harvest'] = 'none'
    irr = run_sim(inputs)
    print('No Harvest: {}'.format(irr))

    inputs['harvest'] = 'std'
    inputs['clock_reset'] = False
    irr = run_sim(inputs)
    print('Std Harvest: {}'.format(irr))

    inputs['harvest'] = 'std'
    inputs['clock_reset'] = True
    irr = run_sim(inputs)
    print('Clock Reset Harvest: {}'.format(irr))
    print('\n')


# Build inputs
test = {'dt': 60,
        'tau_div_start': 0.0,
        'tau_div_end': 0.0,
        'tau_st_start': 0.0,
        'tau_st_end': 0.0,
        'tau_lt_start': 0.0,
        'tau_lt_end': 0.0,
        'donate_start_pct': 0.00,
        'donate_end_pct': 0.00,
        'div_reinvest': False,
        'div_payout': True,
        'div_override': 0.00,
        'harvest': 'none',
        'harvest_thresh': -0.02,
        'harvest_freq': 60,
        'clock_reset': False,
        'rebal_freq': 60,
        'donate_freq': 240,
        'donate_thresh': 0.0,
        'terminal_donation': 0,
        'donate': False,
        'replace': False,
        'randomize': False,
        'return_override': -1,
        'N_sim': 1,
        'savings_reinvest_rate': -1,
        'loss_offset_pct': 1,
        }

scenario1 = {'dt': 60,
             'tau_div_start': 0.28,
             'tau_div_end': 0.28,
             'tau_st_start': 0.5,
             'tau_st_end': 0.5,
             'tau_lt_start': 0.28,
             'tau_lt_end': 0.28,
             'donate_start_pct': 0.05,
             'donate_end_pct': 0.05,
             'div_reinvest': False,
             'div_payout': True,
             'div_override': 0.02,
             'harvest': 'none',
             'harvest_thresh': -0.02,
             'harvest_freq': 60,
             'clock_reset': False,
             'rebal_freq': 240,
             'donate_freq': 240,
             'donate_thresh': 0.0,
             'terminal_donation': 0,
             'donate': True,
             'replace': False,
             'randomize': False,
             'return_override': -1,
             'N_sim': 1,
             'savings_reinvest_rate': -1,
             'loss_offset_pct': 1,
             }

scenario2 = {'dt': 60,
             'tau_div_start': 0.28,
             'tau_div_end': 0.20,
             'tau_st_start': 0.5,
             'tau_st_end': 0.35,
             'tau_lt_start': 0.28,
             'tau_lt_end': 0.20,
             'donate_start_pct': 0.2,
             'donate_end_pct': 0.2,
             'div_reinvest': False,
             'div_payout': True,
             'div_override': 0.02,
             'harvest': 'none',
             'harvest_thresh': -0.02,
             'harvest_freq': 60,
             'clock_reset': False,
             'rebal_freq': 240,
             'donate_freq': 240 * 5,
             'donate_thresh': 0.0,
             'terminal_donation': 0.5,
             'donate': True,
             'replace': True,
             'randomize': True,
             'return_override': 0.05,
             'N_sim': 50,
             'savings_reinvest_rate': -1,
             'loss_offset_pct': 1,
             }

scenario3 = {'dt': 60,
             'tau_div_start': 0.28,
             'tau_div_end': 0.38,
             'tau_st_start': 0.5,
             'tau_st_end': 0.6,
             'tau_lt_start': 0.28,
             'tau_lt_end': 0.38,
             'donate_start_pct': 0.3,
             'donate_end_pct': 0.3,
             'div_reinvest': False,
             'div_payout': True,
             'div_override': 0.02,
             'harvest': 'none',
             'harvest_thresh': -0.02,
             'harvest_freq': 60,
             'clock_reset': False,
             'rebal_freq': 240,
             'donate_freq': 240,
             'donate_thresh': 0.0,
             'terminal_donation': 0,
             'donate': True,
             'replace': False,
             'randomize': False,
             'return_override': -1,
             'N_sim': 1,
             'savings_reinvest_rate': 0.1,
             'loss_offset_pct': 1,
             }

scenario4 = {'dt': 60,
             'tau_div_start': 0.28,
             'tau_div_end': 0.20,
             'tau_st_start': 0.5,
             'tau_st_end': 0.35,
             'tau_lt_start': 0.28,
             'tau_lt_end': 0.20,
             'donate_start_pct': 0.1,
             'donate_end_pct': 0.1,
             'div_reinvest': False,
             'div_payout': True,
             'div_override': 0.02,
             'harvest': 'none',
             'harvest_thresh': -0.02,
             'harvest_freq': 60,
             'clock_reset': False,
             'rebal_freq': 240,
             'donate_freq': 240,
             'donate_thresh': 0.0,
             'terminal_donation': 1,
             'donate': True,
             'replace': True,
             'randomize': True,
             'return_override': 0.05,
             'N_sim': 50,
             'savings_reinvest_rate': -1,
             'loss_offset_pct': 0.75,
             }

scenario5 = {'dt': 60,
             'tau_div_start': 0.28,
             'tau_div_end': 0.38,
             'tau_st_start': 0.5,
             'tau_st_end': 0.6,
             'tau_lt_start': 0.28,
             'tau_lt_end': 0.38,
             'donate_start_pct': 0.50,
             'donate_end_pct': 0.50,
             'div_reinvest': False,
             'div_payout': True,
             'div_override': 0.02,
             'harvest': 'none',
             'harvest_thresh': -0.02,
             'harvest_freq': 60,
             'clock_reset': False,
             'rebal_freq': 240,
             'donate_freq': 240,
             'donate_thresh': 0.0,
             'terminal_donation': 0,
             'donate': True,
             'replace': True,
             'randomize': True,
             'return_override': 0.05,
             'N_sim': 50,
             'savings_reinvest_rate': 0.1,
             'loss_offset_pct': 1,
             }

print('*** Test ***')
run_scenario(test)

print('*** Scenario 1 ***')
run_scenario(scenario1)

print('*** Scenario 2 ***')
run_scenario(scenario2)

print('*** Scenario 3 ***')
run_scenario(scenario3)

print('*** Scenario 4 ***')
run_scenario(scenario4)

print('*** Scenario 5 ***')
run_scenario(scenario5)
