import numpy as np
from typing import Optional


def calc_path_stats(dt: int, d_px: np.array, w0: np.array,
                    vol_fixed_w: bool = True,
                    calc_res_vol: bool = False,
                    ann_factor: float = 252) -> tuple:
    """ Generate statistics of a set of paths
    :param dt:  trading days between data points
    :param d_px:  array of simple returns (n_steps x n_stocks)
    :param vol_fixed_w: if True: calculate vol assuming we rebalance to fixed weights w0
                        otherwise, assume we keep fixed basket
    :param calc_res_vol:  calculate average residual vol of stocks
    :param ann_factor: # business days per year

    :return: tuple
                * idx_return - return of the index, freq = (ann_factor / dt)
                * idx_vol - annualized index vol (log returns)
                * resid_vol - weighted average annualized vol of residual returns
    """

    n_steps = d_px.shape[0]
    n_stocks = d_px.shape[1]
    px1 = np.ones((n_steps+1, n_stocks))
    px1[1:] = (1 + d_px).cumprod(axis=0)
    idx_vals = px1 @ w0
    # idx_rets = np.zeros(n_steps + 1)
    idx_rets = idx_vals[1:] / idx_vals[:-1] - 1

    # Calculate index return
    idx_return = (ann_factor / dt) * (idx_vals[-1] ** (1 / n_steps) - 1)

    # Calculate index vol
    if vol_fixed_w:
        idx_vol = np.sqrt(ann_factor / dt) * np.std(np.log(1 + d_px @ w0))
    else:
        idx_vol = np.sqrt(ann_factor / dt) * np.std(np.log(1 + idx_rets))

    # Calculate residual vols
    if calc_res_vol:
        coef = np.apply_along_axis(lin_model, 0, d_px, idx_rets)
        alpha = coef[0, :]
        beta = coef[1, :]

        # Residual returns
        res_ret = d_px - alpha[None, :] - idx_rets[:, None] @ beta[None, :]

        # Sqrt of weighted variance (i.e. as if they are uncorrelated)
        resid_vol = np.sqrt(ann_factor / dt) * np.std(res_ret, axis=0) @ w0
    else:
        resid_vol = None

    return idx_return, idx_vol, resid_vol


def lin_model(y: np.array, x: np.array) -> tuple:
    """ Calculate alpha and beta via linear regression
        :param y: endogenous (dependent) variable
        :param x: independent variable
        :return tuple: beta (float), alpha (float)
    """

    covmat = np.cov((y, x))
    beta = covmat[0, 1] / covmat[1, 1]
    alpha = np.mean(y) - beta * np.mean(x)
    return alpha, beta


def rescale_stocks_to_match_idx(d_px: np.array, w0: np.array,
                                idx_ann_ret: Optional[float] = None,
                                idx_ann_vol: Optional[float] = None,
                                stk_res_vol_factor: Optional[float] = None,
                                ann_factor: float = 1.0,
                                output_rets=True) -> tuple:
    """ Rescale a dataframe of individual stock moves so that annualized
        volatility of the index (fixed shares) matches a target in vol (std of log ret)
        and average geometric return, and that average vol of stocks also matches the target
    :param d_px: 2-d (t x s) array of price changes, axis 0 is time, axis1 is securities
    :param w0: 1-d array size s with initial weights for each stock, can be thought as the
                number of shares if initial price for all stocks is 1.0.
    :param idx_ann_ret: target geometric avg annual return of the index,
                    same freq as data points, set by ann_factor
    :param idx_ann_vol: target period index vol
    :param stk_res_vol_factor: factor by which we adjust residual stock vol (to match single stock target)
    :param ann_factor: number of observations per annum (for scaling)
    :param output_rets: if True output returns (t x s), else output prices (t+1 x s)
    :return: tuple
            If output_rets is True:
                    (1) np.array (t x s) - a new series of re-scaled percent changes
                    (2) np.array (t) - series of target index changes
                        If output_rets is True:
            If output_rets is False:
                    (1) np.array (t+1 x s) - a new series of re-scaled stock prices
                    (2) np.array (t+1) - series of target index values

    """

    n_steps, n_stocks = d_px.shape

    if idx_ann_ret is not None:
        idx_avg_ret_tgt = (1 + idx_ann_ret) ** (1 / ann_factor) - 1

    if idx_ann_vol is not None:
        idx_vol_tgt = idx_ann_vol / np.sqrt(ann_factor)

    # -----------------------------------
    # Calculate target index values
    # -----------------------------------
    # Stock prices
    px = np.ones((n_steps + 1, n_stocks))
    px[1:, :] = (1 + d_px).cumprod(axis=0)
    idx_val_init = px @ w0

    # Match target vol, for now don't worry about return
    idx_ret_init = idx_val_init[1:] / idx_val_init[:-1] - 1
    idx_lret_init = np.log(1 + idx_ret_init)
    idx_vol_init = np.std(idx_lret_init)
    if idx_ann_vol is not None:
        vol_adj_factor = idx_vol_tgt / idx_vol_init

    else:
        vol_adj_factor = 1.0

    idx_lret_adj4vol = idx_lret_init * vol_adj_factor

    # Now match return
    if idx_ann_ret is not None:
        idx_tgt_final_val = (1 + idx_avg_ret_tgt) ** n_steps
    else:
        idx_tgt_final_val = np.exp(idx_lret_init.sum())

    idx_ret_adj_factor = (np.log(idx_tgt_final_val) - idx_lret_adj4vol.sum()) / n_steps

    idx_lret_out = idx_lret_adj4vol + idx_ret_adj_factor
    idx_vals_out = idx_val_init.copy()
    idx_vals_out[1:] = np.exp(idx_lret_out.cumsum())
    idx_rets_out = np.exp(idx_lret_out) - 1

    # --------------------------------------------------------------------
    # Now re-scale residual vols, while matching index target values above
    # --------------------------------------------------------------------
    if (stk_res_vol_factor is not None) and (stk_res_vol_factor != 1.0):
        # Fit a linear model (alpha, beta)
        coef = np.apply_along_axis(lin_model, 0, d_px, idx_ret_init)
        alpha = coef[0, :]
        beta = coef[1, :]

        # Residual returns
        res_ret = d_px - alpha[None, :] - idx_ret_init[:, None] @ beta[None, :]

        # Calculate intermediate (phase1) stock paths with desired residual returns
        d_px_phase1 = d_px + res_ret * (stk_res_vol_factor - 1)
        px_phase1 = px.copy()
        px_phase1[1:, :] = (1 + d_px_phase1).cumprod(axis=0)
        idx_val_phase1 = px_phase1 @ w0
    else:
        px_phase1 = px
        idx_val_phase1 = idx_val_init

    # Rescale price path to match the desired index values
    px_out = px_phase1 * (idx_vals_out / idx_val_phase1)[:, None]
    d_px_out = px_out[1:, :] / px_out[:-1, :] - 1

    # Depending on the flag output either prices or returns
    if output_rets:
        return d_px_out, idx_rets_out
    else:
        return px_out, idx_vals_out
