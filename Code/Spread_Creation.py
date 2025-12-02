import numpy as np
import pandas as pd
import statsmodels.api as sm

class SpreadConstructor():

    def __init__(self, prices: pd.DataFrame, pairs, method="beta"):
        self.prices = prices
        self.pairs = [tuple(p) for p in pairs]
        self.method = method

    @staticmethod
    def _hedge_ratio_ols(y, x):
        
        x_ = sm.add_constant(x)
        model = sm.OLS(y, x_).fit()
        return model.params.iloc[1]

    def _beta_neutral_spread(self, a, b):

        # align timestamps and drop NaNs pairwise
        pa = self.prices[a].dropna()
        pb = self.prices[b].dropna()
        idx = pa.index.intersection(pb.index)
        pa, pb = pa.loc[idx], pb.loc[idx]

        beta = self._hedge_ratio_ols(pa, pb)
        spread = pa - beta * pb

        df = pd.DataFrame(
            {
                a: pa,
                b: pb,
                "spread": spread,
                "w_a": 1.0,
                "w_b": -beta,
            },
            index=idx,
        )
        return df

    def _dollar_neutral_spread(self, a, b):
        """
        Idea:
            Choose weights w_a, w_b such that at initial time t0:
                w_a * P_a(t0) + w_b * P_b(t0) = 0

            For example:
                w_a = 1
                w_b = -P_a(t0) / P_b(t0)
        """
        pa = self.prices[a].dropna()
        pb = self.prices[b].dropna()
        idx = pa.index.intersection(pb.index)
        pa, pb = pa.loc[idx], pb.loc[idx]

        p0_a = pa.iloc[0]
        p0_b = pb.iloc[0]

        w_a = 1.0
        w_b = -p0_a / p0_b

        spread = w_a * pa + w_b * pb

        df = pd.DataFrame(
            {
                a: pa,
                b: pb,
                "spread": spread,
                "w_a": w_a,
                "w_b": w_b,
            },
            index=idx,
        )
        return df

    def build_pair_spread(self, pair):

        a, b = tuple(pair)
        if a not in self.prices.columns or b not in self.prices.columns:
            raise ValueError(f"Pair {pair} not in price columns")

        if self.method == "beta":
            return self._beta_neutral_spread(a, b)
        elif self.method == "dollar":
            return self._dollar_neutral_spread(a, b)
        else:
            raise ValueError(f"Unknown method '{self.method}'")

    def build_all_spreads(self):

        spread_dict = {}
        for pair in self.pairs:
            try:
                df = self.build_pair_spread(pair)
                if not df.empty:
                    spread_dict[pair] = df
            except Exception as e:
                print(f"Skipping pair {pair}: {e}")
        return spread_dict
