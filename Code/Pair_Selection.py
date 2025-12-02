import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


class PairConfirmation():

    def __init__(self, adf_threshold=0.05, hurst_max=0.45,
                 hl_min=1, hl_max=60):
        """
        adf_threshold : max p-value for Engle–Granger cointegration
        hurst_max     : H must be < hurst_max for mean reversion
        hl_min        : minimum acceptable half-life (days)
        hl_max        : maximum acceptable half-life (days)
        """
        self.adf_threshold = adf_threshold
        self.hurst_max = hurst_max
        self.hl_min = hl_min
        self.hl_max = hl_max

    # ============================
    # 1. OLS hedge ratio
    # ============================
    @staticmethod
    def hedge_ratio(y, x):
        x_ = sm.add_constant(x)
        model = sm.OLS(y, x_).fit()
        return model.params.iloc[1]


    # ============================
    # 2. Spread estimation
    # ============================
    def compute_spread(self, y, x):
        beta = self.hedge_ratio(y, x)
        return y - beta * x, beta

    # ============================
    # 3. Engle–Granger cointegration
    # ============================
    def engle_granger_adf(self, spread):
        try:
            pval = adfuller(spread, regression="ct")[1]
        except Exception:
            return False, 1.0
        return pval < self.adf_threshold, pval

    # ============================
    # 4. Hurst exponent estimation
    # ============================
    @staticmethod
    def hurst_exponent(ts):
        ts = ts.dropna().values
        N = len(ts)
        if N < 50:
            return np.nan  # too short

        lags = range(2, min(50, N // 2))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    # ============================
    # 5. Half-life estimation
    # ============================
    @staticmethod
    def half_life(spread):
        spread = spread.dropna()
        if len(spread) < 30:
            return np.nan

        y = spread[1:].values
        x = spread[:-1].values

        rho = np.polyfit(x, y, 1)[0]

        if rho >= 1 or rho <= -1:
            return np.inf 

        try:
            hl = -1 / np.log(abs(rho))
        except Exception:
            hl = np.inf

        return hl

    # ============================
    # 6. Full confirmation pipeline
    # ============================
    def confirm_pair(self, y, x):

        idx = y.dropna().index.intersection(x.dropna().index)
        y2, x2 = y.loc[idx], x.loc[idx]

        spread, beta = self.compute_spread(y2, x2)

        eg_ok, adf_p = self.engle_granger_adf(spread)

        H = self.hurst_exponent(spread)
        hurst_ok = (not np.isnan(H)) and (H < self.hurst_max)

        hl = self.half_life(spread)
        hl_ok = (hl >= self.hl_min) and (hl <= self.hl_max)

        is_valid = eg_ok and hurst_ok and hl_ok

        diagnostics = {
            "hedge_ratio": beta,
            "adf_pvalue": adf_p,
            "eg_stationary": eg_ok,
            "hurst": H,
            "hurst_ok": hurst_ok,
            "half_life": hl,
            "half_life_ok": hl_ok,
            "is_valid": is_valid,
        }

        return is_valid, diagnostics
