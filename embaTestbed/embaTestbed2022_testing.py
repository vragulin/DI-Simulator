import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import copy
import time
import pickle
import emba_config as cnf

np.seterr('raise')

# EMBA standalone sandbox
def load_data(data_freq: int, data_dict={}, replace=False, randomize=False, return_override=0,
              vol_override=0, rand_state=0) -> dict:

    if not len(data_dict):
        px = pd.read_pickle(os.path.join(cnf.working_dir, cnf.PX_PICKLE)).fillna(method='ffill').fillna(0)

        # Patch -- Need to save dates for downstream simulations
        dates = px.index
        dates_idx = range(0, len(px.index), data_freq)
        # -------------------------------
        px.index = range(0, len(px.index))
        px = px.reindex(index=range(0, len(px.index), data_freq))

        tri = pd.read_pickle(os.path.join(cnf.working_dir, cnf.TR_PICKLE)).fillna(method='ffill').fillna(0)
        tri.index = range(0, len(tri.index))
        tri = tri.reindex(index=range(0, len(tri.index), data_freq))

        w = pd.read_pickle(os.path.join(cnf.working_dir, cnf.W_PICKLE)).fillna(0)
        w.index = range(0, len(w.index))
        w = np.maximum(0, w.reindex(index=range(0, len(w.index), data_freq)))
        w /= (np.sum(w.to_numpy(), axis=1)[:, None] @ np.ones((1, len(w.columns))))

        out_dict = {'px': px, 'tri': tri, 'w': w}

        # Build returns
        out_dict['d_px'] = out_dict['px'].pct_change().fillna(0).replace(np.inf, 0)
        out_dict['d_tri'] = out_dict['tri'].pct_change().fillna(0).replace(np.inf, 0)
        out_dict['div'] = out_dict['d_tri'] - out_dict['d_px']

        # Set random seed for future iterations
        np.random.seed(7)

        # Also save dates
        out_dict['dates'] = dates
        out_dict['dates_idx'] = dates_idx
    else:
        out_dict = copy.deepcopy(data_dict)
        # out_dict = data_dict.copy()

    # print("First 5 weights:", out_dict['w'].values[0, :5])
    # print("First 5 returns:", out_dict['d_px'].values[1, :5])

    # Randomize
    if randomize:

        # find stocks present for the entire sample and pull out everything else
        full_indic = np.sum((out_dict['px'].to_numpy() > 0), axis=0) == len(out_dict['px'].index)
        # Reshuffle returns, don't touch the first element since it's zero
        d_px_new = out_dict['d_px'].iloc[1:, full_indic]
        d_px_new = d_px_new.sample(frac=1, replace=replace)
        d_px = pd.concat([pd.DataFrame(0, index=[0], columns=d_px_new.columns), d_px_new]).reset_index()
        out_dict['shuffle'] = d_px['index'].values
        # d_px = out_dict['d_px'].iloc[:, full_indic].sample(frac=1, replace=replace).reset_index()
        d_px.drop(columns='index', inplace=True)

        # print("First 5 d_px", end=": ")
        # print(d_px.iloc[:5,0])
        # print(f"rand_state={rand_state}")

        rand_weights = np.exp(np.random.normal(size=(1, d_px.shape[1])))
        rand_weights /= np.sum(rand_weights)
        # print(f"Max Weight: {rand_weights.max():.4f}")
        # print("First 5 weights:", rand_weights[0,:5])
        # print("First 5 returns:", d_px.values[0, :5])

        # Get returns approximately equal to override
        if vol_override > 0:
            vol = np.sqrt(252 / data_freq) * np.var(d_px.to_numpy() @ rand_weights.transpose()) ** 0.5
            d_px *= vol_override / vol
            d_px = np.maximum(d_px, -1 + np.finfo(float).eps)  # Ensure no neg prices

        try:
            if return_override > 0:
                rand_return = (252 / data_freq) * (np.sum(rand_weights *
                                                          np.product(1 + d_px.to_numpy()[1:, :], axis=0)) ** (
                                                           1 / (d_px.shape[0])) - 1)
                d_px += (return_override - rand_return) * (data_freq / 252)
                d_px.iloc[0, :] = 0
                d_px = np.maximum(d_px, -1 + np.finfo(float).eps)  # Ensure no neg prices

        except FloatingPointError:
            raise FloatingPointError("Negative prices")

        # **************************************************
        # VR Patch: Check that the resulting index has the right vol
        # **************************************************
        # n_steps, n_stocks = d_px.shape
        # px1 = np.ones((n_steps+1, n_stocks))
        # px1[1:, :] = (1 + d_px).cumprod()
        # idx_vals = px1 @ rand_weights.T
        # idx_rets = idx_vals[1:]/idx_vals[:-1]-1
        # # idx_return = (252/data_freq) * (idx_vals[-1] ** (1 / n_steps) - 1)[0]
        # idx_vol = np.sqrt(252/data_freq) * np.std(idx_rets)
        # # print(f"# shares: {full_indic.sum()}", end=" ; ")
        # # print(f"Index ret: {idx_return*100:.2f}, tgt: {return_override*100:.2f} ", end=" ; ")
        # # print(f"Index vol: {idx_vol*100:.2f}, tgt: {vol_override*100:.2f}", end = " ; ")
        # # print(f"Median final stk px: {np.median(px1[-1,:]):.5f}")
        # out_dict['idx_vol'] = idx_vol
        # **************************************************
        # End of patch
        # **************************************************

        out_dict['w'] = np.ones((out_dict['w'].shape[0], 1)) @ rand_weights
        out_dict['d_px'] = d_px
        out_dict['div'] = out_dict['div'].iloc[:, full_indic]
        out_dict['px'] = out_dict['px'].iloc[:, full_indic]  # NB - prices are not reshuffled

    else:
        # Create a trivial shuffle  array even there
        # is no randomization
        out_dict['shuffle'] = out_dict['d_px'].index

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
    if div_override >= 0 and (dt * t % 240) == 0:
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
                        (gross_gains / (np.finfo(float).eps + basis) >= harvest_thresh) * lt_indic
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

    return basis, lot_start, savings


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
    cf[t] += cf_value

    return basis, mv, lot_start


def update_donate(t: float, mv: np.array, basis: np.array, lot_start: np.array, donate_thresh: float,
                  donate_pct: float) -> tuple:
    gains_pct = (mv - basis) / (basis + np.finfo(float).eps)
    mv_tot = np.sum(mv)
    donate_amt = mv_tot
    donate_indic = False # Initialize to have some value
    while (donate_amt / mv_tot) >= donate_pct:
        donate_indic = gains_pct >= donate_thresh
        donate_amt = np.sum(donate_indic * mv)
        # donate_thresh += min(max(donate_thresh * 0.1, 0.02), 0.1)
        donate_thresh += max(donate_thresh * 0.05, 0.05)
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
    n_t = px.shape[0]

    # Go through time
    cf = np.zeros(n_t)  # [Time]
    basis = np.zeros((w.shape[1], 1))  # [Tkr, Lot]
    mv = np.zeros((w.shape[1], 1))  # [Tkr, Lot]
    harvest = np.zeros(w.shape[0])
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
        # print(f"t={t}, port_val = {mv.sum():.6f}, basis = {basis.sum():.6f}")

        # Harvest
        if inputs['dt'] * t % inputs['harvest_freq'] == 0:
            if inputs['harvest'] != 'none':
                (basis, lot_start, savings) = update_harvest(inputs['dt'], t, inputs['harvest_thresh'], tau_st, tau_lt,
                                                             mv,
                                                             basis, lot_start, cf, inputs['clock_reset'],
                                                             inputs['savings_reinvest_rate'], inputs['loss_offset_pct'])
                # print(f"Savings: {savings}")
                harvest[t] = savings / mv.sum()
        # Rebal
        if inputs['dt'] * t % inputs['rebal_freq'] == 0 and not inputs['randomize']:
            # print(np.sum((w[t, :] > 0) != (w[t-3, :] > 0)))
            (basis, mv, lot_start) = update_rebal(inputs['dt'], cash, t, tau_st, tau_lt, w, mv, lot_idx, basis,
                                                  lot_start, px, cf)
            cash = 0

        # Donate
        if inputs['dt'] * t % inputs['donate_freq'] == 0 and inputs['donate'] and (donate_pct > np.finfo(float).eps):
            (basis, lot_start) = update_donate(t, mv, basis, lot_start, inputs['donate_thresh'], donate_pct)

    # Calc IRR
    st_indic = inputs['dt'] * (t - lot_start) < 252
    lt_indic = 1 - st_indic
    cf[n_t - 1] += np.sum(mv) - (1 - inputs['terminal_donation']) * (np.sum(st_indic * tau_st * (mv - basis)) +
                                                                     np.sum(lt_indic * tau_lt * (mv - basis)))
    x0 = np.asarray([0.09])
    irr = minimize(irr_obj, x0, args=(cf, inputs['dt']), bounds=[(0.01, 0.2)]).x

    return irr, harvest


def run_sim(inputs: dict) -> tuple:
    N = inputs['N_sim']
    irr = np.zeros(N)
    harv_avg = None
    base_data_dict = load_data(inputs['dt'], replace=False, randomize=False, return_override=-1, vol_override=-1)

    # Start simulation - loop over paths
    for n in range(0, N):
        # print(f"max_px: {(1+base_data_dict['d_px']).product().max()}")
        data_dict = load_data(inputs['dt'], replace=inputs['replace'], randomize=inputs['randomize'],
                              data_dict=base_data_dict, return_override=inputs['return_override'],
                              vol_override=inputs['vol_override'])

        if n == 0:
            # Set up arrays to store shuffle and weight info for paths
            n_steps, n_stocks = data_dict['px'].shape + np.array((-1, 0))
            path_shuffles = np.zeros((N, n_steps + 1), dtype=np.int_)
            path_weights = np.zeros((N, n_stocks))

        # Record the shuffle and the weights used for the path
        # noinspection PyUnboundLocalVariable
        path_shuffles[n] = data_dict['shuffle']
        # noinspection PyUnboundLocalVariable
        path_weights[n] = data_dict['w_arr'][0, :]

        # ------------------------
        # VR patch - flag is vol is too high, and replace with average
        # ------------------------
        # if data_dict['idx_vol'] > 1:
        #     print(f"Iter = {n}, idx_vol = {data_dict['idx_vol']}")
        #     if n > 0:
        #         irr[n] = np.mean(irr[:n])
        #     else:
        #         irr[n] = 0.08
        #     continue
        # ------------------------
        # End of patch
        # ------------------------
        irr[n], harvest = run_sim_path(data_dict, inputs)

        # Accumulate the vector of harvest over time
        if harv_avg is None:
            harv_avg = harvest / N
        else:
            harv_avg += harvest / N

    # print("IRR by paths", end=": ")
    # print(irr)
    mean_irr = float(np.mean(irr))

    if inputs['save_path_info']:

        #  Save file with path info.  Label with indicative info
        try:
            file_code = inputs['path_file_code']
            code_string = f"_{file_code}"
        except KeyError:
            code_string = ""

        code_string += f"_{inputs['dt']}"
        data_dir = cnf.PATHS_DIR
        pd.DataFrame(path_shuffles, index=range(N), columns=range(n_steps + 1)
                     ).to_csv(f"{data_dir}/path_shuffles{code_string}.csv")
        pd.DataFrame(path_weights, index=range(N), columns=base_data_dict['px'].columns
                     ).to_csv(f"{data_dir}/path_weights{code_string}.csv")
        with open(f"{data_dir}/base_data_dict{code_string}.pickle", 'wb') as handle:
            pickle.dump(base_data_dict, handle)

    return mean_irr, harv_avg


def run_scenario_vr(inputs: dict, only_base: bool = False):
    res_dir = '../results/emba/'

    inputs['harvest'] = 'none'
    # inputs['harvest'] = 'reset'  # The program only checks that harvest!='none'
    inputs['clock_reset'] = False
    irr, _ = run_sim(inputs)
    # print('Std Harvest: {}'.format(irr))
    print('No Harvest: {}'.format(irr))

    if not only_base:
        # Don't save the paths again - they are the same
        inputs['save_path_info'] = False

        # Run Standard Harvest
        inputs['harvest'] = 'std'
        inputs['clock_reset'] = False
        irr, harvest_avg = run_sim(inputs)
        print('Std Harvest: {}'.format(irr))
        # Save hist_harvest
        f_path = os.path.join(res_dir, 'harvest_std.csv')
        harvest_avg.tofile(f_path, sep=',', format='%.5f')

        # Run Harvest with clock reset
        inputs['harvest'] = 'reset'
        inputs['clock_reset'] = True
        irr, harvest_avg = run_sim(inputs)
        print('Clock Reset Harvest: {}'.format(irr))
        # Save hist_harvest
        f_path = os.path.join(res_dir, 'harvest_reset.csv')
        harvest_avg.tofile(f_path, sep=',', format='%.5f')

        inputs['harvest'] = 'none'
        inputs['tau_div_start'] = 0
        inputs['tau_div_end'] = 0
        inputs['tau_st_start'] = 0
        inputs['tau_st_end'] = 0
        inputs['tau_lt_start'] = 0
        inputs['tau_lt_end'] = 0
        irr, _ = run_sim(inputs)
        print('No Harvest / No Taxes: {}'.format(irr))

    print('\n')


def run_scenario(inputs: dict):
    inputs['harvest'] = 'none'
    irr, _ = run_sim(inputs)
    print('No Harvest: {}'.format(irr))
    print('\n')

    inputs['harvest'] = 'std'
    inputs['clock_reset'] = False
    irr, _ = run_sim(inputs)
    print('Std Harvest: {}'.format(irr))
    print('\n')

    inputs['harvest'] = 'std'
    inputs['clock_reset'] = True
    irr, _ = run_sim(inputs)
    print('Clock Reset Harvest: {}'.format(irr))
    print('\n')

    inputs['harvest'] = 'none'
    inputs['tau_div_start'] = 0
    inputs['tau_div_end'] = 0
    inputs['tau_st_start'] = 0
    inputs['tau_st_end'] = 0
    inputs['tau_lt_start'] = 0
    inputs['tau_lt_end'] = 0
    irr, _ = run_sim(inputs)
    print('No Harvest / No Taxes: {}'.format(irr))
    print('\n')


def run_scenario_hypercube(inputs: dict):
    dir_path = "../results/emba/"
    if cnf.clean_flag:
        fcode = 'clean'
    else:
        fcode = 'full'

    f = open(os.path.join(dir_path, f'emba_irr_hypercube_20220921_60_{fcode}.csv'), 'w')
    header = ''
    for k in inputs:
        header += k + ','
    header += 'irr\n'
    f.write(header)

    repls = [True, False]
    harvests = ['none', 'std', 'reset']
    donations = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    returns = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    vols = [-1, 0.12, 0.14, 0.16, 0.18, 0.2]

    # harvests = ['none', 'std']
    # donations = [0.05]
    # returns = [0.11]
    # vols = [-1]

    counter = 0
    for repl in repls:
        for donation in donations:
            for r in returns:
                for v in vols:
                    for harvest in harvests:
                        counter += 1
                        tic = time.perf_counter()  # Timer start
                        inputs['replace'] = repl
                        inputs['return_override'] = r - 0.02
                        inputs['vol_override'] = v
                        inputs['donate_start_pct'] = donation
                        inputs['donate_end_pct'] = donation
                        if donation == 0:
                            inputs['donate'] = False
                        else:
                            inputs['donate'] = True
                        inputs['harvest'] = harvest
                        if harvest == 'reset':
                            inputs['clock_reset'] = True
                        else:
                            inputs['clock_reset'] = False

                        irr, _ = run_sim(inputs)
                        output = ''
                        for k in inputs:
                            if k == 'return_override':
                                output += str(inputs[k] + 0.02) + ','
                            else:
                                output += str(inputs[k]) + ','
                        output += str(irr) + '\n'
                        f.write(output)
                        toc = time.perf_counter()  # Timer end
                        t_path = toc - tic
                        print('{}: Repl: {}, Harvest: {}, Donation: {}, Return: {}, Vol: {}, '
                              'IRR: {:.6f}, t={:.3f} sec'.format(counter, repl, harvest, donation,
                                                                 r, v, irr, t_path))

    f.close()
    print("Done")


def run_test_one_path(inputs):
    only_base = False
    base_scenario = inputs.copy()
    # base_scenario['div_override'] = 0.02
    # base_scenario['div_reinvest'] = True <- this functionality is not working

    print('*** Test ***')
    if inputs['replace'] == True:
        inputs['path_file_code'] = 'replace'
    else:
        inputs['path_file_code'] = 'shuffle'

    run_scenario_vr(inputs, only_base=True)

    # print('*** Test with Taxes ***')
    # taxes = {'tau_div_start': 0.28,
    #          'tau_div_end': 0.28,
    #          'tau_st_start': 0.5,
    #          'tau_st_end': 0.5,
    #          'tau_lt_start': 0.28,
    #          'tau_lt_end': 0.28
    #          }
    # inputs = base_scenario
    # inputs.update(taxes)
    # inputs['save_path_info'] = False # No need to save the second time
    # run_scenario_vr(inputs, only_base=only_base)

    # print('*** Test with Taxes and Terminal Donation ***')
    # test = base_scenario
    # test.update(taxes)
    # test['terminal_donation'] = 1
    # run_scenario(test, only_base=only_base)

# Run to catch a floating point error
# run_scenario(test_catch_float)
# run_test_one_path(scen_cube_test)


if __name__ == "__main__":
    # Build inputs
    test = {'dt': 60,
            'tau_div_start': 0.0,
            'tau_div_end': 0.0,
            'tau_st_start': 0.0,
            'tau_st_end': 0.0,
            'tau_lt_start': 0.0,
            'tau_lt_end': 0.0,
            'donate_start_pct': 0.05,
            'donate_end_pct': 0.05,
            'div_reinvest': False,
            'div_payout': True,
            'div_override': 0.02,
            'harvest': 'none',
            'harvest_thresh': -0.02,
            'harvest_freq': 60,
            'clock_reset': False,
            'rebal_freq': 60,
            'donate_freq': 1200,
            'donate_thresh': 0.0,
            'terminal_donation': 0,
            'donate': True,
            'replace': True,
            'randomize': True,
            'return_override': 0.09,
            'N_sim': 10,
            'savings_reinvest_rate': -1,
            'loss_offset_pct': 1,
            'vol_override': -1,
            'save_path_info': True
            }

    scenario1 = {
        'dt': 60,
        'tau_div_start': 0.28,
        'tau_div_end': 0.28,
        'tau_st_start': 0.5,
        'tau_st_end': 0.5,
        'tau_lt_start': 0.28,
        'tau_lt_end': 0.28,
        'donate_start_pct': 0.2,
        'donate_end_pct': 0.2,
        'div_reinvest': False,
        'div_payout': True,
        'div_override': -1,
        'harvest': 'none',
        'harvest_thresh': -0.02,
        'harvest_freq': 60,
        'clock_reset': False,
        'rebal_freq': 240,
        'donate_freq': 240 * 5,
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

    scen_cube_test = {'dt': 20,
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
                      'harvest_freq': 20,
                      'clock_reset': True,
                      'rebal_freq': 20,
                      'donate_freq': 240 * 5,
                      'donate_thresh': 0.0,
                      'terminal_donation': 0,
                      'donate': False,
                      'replace': True,
                      'randomize': True,
                      'return_override': 0.09,
                      'N_sim': 2,
                      'savings_reinvest_rate': -1,
                      'loss_offset_pct': 1,
                      'vol_override': -1,
                      'save_path_info': False
                      }
    scenario1_cube = {'dt': 60,
                      'tau_div_start': 0.28,
                      'tau_div_end': 0.28,
                      'tau_st_start': 0.5,
                      'tau_st_end': 0.5,
                      'tau_lt_start': 0.28,
                      'tau_lt_end': 0.28,
                      'donate_start_pct': 0.2,
                      'donate_end_pct': 0.2,
                      'div_reinvest': False,
                      'div_payout': True,
                      'div_override': 0.02,
                      'harvest': 'none',
                      'harvest_thresh': -0.02,
                      'harvest_freq': 60,
                      'clock_reset': False,
                      'rebal_freq': 60,
                      'donate_freq': 240 * 5,
                      'donate_thresh': 0.0,
                      'terminal_donation': 0,
                      'donate': True,
                      'replace': True,
                      'randomize': True,
                      'return_override': -1,
                      'N_sim': 100,
                      'savings_reinvest_rate': -1,
                      'loss_offset_pct': 1,
                      'vol_override': -1,
                      'save_path_info': False
                      }


    # test_catch_float = scenario1_cube.copy()
    # test_catch_float['return_override'] = 0.03
    # test_catch_float['vol_override'] = 0.2


    print('*** Run Hypercube Batch ***')
    run_scenario_hypercube(scenario1_cube)
    #
    # print('*** Scenario 1 ***')
    # run_scenario(scenario1)
    #
    # print('*** Scenario 2 ***')
    # run_scenario(scenario2)
    #
    # print('*** Scenario 3 ***')
    # run_scenario(scenario3)
    #
    # print('*** Scenario 4 ***')
    # run_scenario(scenario4)
    #
    # print('*** Scenario 5 ***')
    # run_scenario(scenario5)
