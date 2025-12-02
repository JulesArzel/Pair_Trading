# backtester.py

import numpy as np
import pandas as pd


class PairBacktester:
    """
    Assumptions:
      - Same capital allocated to each pair.
      - No transaction costs (for now).
      - Margin call: if equity <= 0, the pair is liquidated permanently.
    """

    def __init__(self, signals_dict, capital=1_000_000.0):
        self.signals_dict = signals_dict
        self.capital = capital
        self.pair_results = {}
        self.portfolio = None

    def _compute_pair_pnl(self, pair, df_pair, capital_per_pair):

        a, b = pair

        if a not in df_pair.columns or b not in df_pair.columns:
            raise ValueError(
                f"Pair {pair}: price columns {a} and {b} not found in DataFrame"
            )

        df = df_pair.copy()

        w_a_series = df["w_a"].astype(float) if "w_a" in df.columns else pd.Series(
            1.0, index=df.index
        )
        w_b_series = df["w_b"].astype(float) if "w_b" in df.columns else pd.Series(
            -1.0, index=df.index
        )

        pa = df[a].astype(float)
        pb = df[b].astype(float)

        valid = df[[a, b]].dropna()
        if valid.empty:
            out = pd.DataFrame(index=df.index)
            out["pair_pnl"] = 0.0
            out["pair_equity"] = capital_per_pair
            out["units"] = 0.0
            out["margin_call"] = False
            return out

        first_idx = valid.index[0]

        w_a0 = w_a_series.loc[first_idx]
        w_b0 = w_b_series.loc[first_idx]
        pa0 = pa.loc[first_idx]
        pb0 = pb.loc[first_idx]

        base_notional = abs(w_a0) * pa0 + abs(w_b0) * pb0
        if base_notional <= 0 or np.isnan(base_notional):
            units = 0.0
        else:
            units = capital_per_pair / base_notional

        dPa = pa.diff().fillna(0.0)
        dPb = pb.diff().fillna(0.0)
        dSpread = w_a_series * dPa + w_b_series * dPb

        pos_prev = df["position"].shift(1).fillna(0.0)

        pair_pnl = np.zeros(len(df))
        equity = np.zeros(len(df))
        margin_call = np.zeros(len(df), dtype=bool)

        equity[0] = capital_per_pair 

        for i in range(1, len(df)):
            if margin_call[i - 1]:
                equity[i] = equity[i - 1]
                pair_pnl[i] = 0.0
                margin_call[i] = True
                continue

            pnl_today = pos_prev.iloc[i] * units * dSpread.iloc[i]
            pair_pnl[i] = pnl_today

            equity[i] = equity[i - 1] + pnl_today

            if equity[i] <= 0:
                equity[i] = 0.0
                margin_call[i] = True

                continue

        out = df.copy()
        out["units"] = units
        out["pair_pnl"] = pair_pnl
        out["pair_equity"] = equity
        out["margin_call"] = margin_call

        return out

    def run(self):
        if not self.signals_dict:
            raise ValueError("signals_dict is empty, nothing to backtest")

        n_pairs = len(self.signals_dict)
        capital_per_pair = self.capital / n_pairs

        pair_pnl_frames = []
        all_dates = set()

        for pair, df_pair in self.signals_dict.items():
            result_df = self._compute_pair_pnl(pair, df_pair, capital_per_pair)
            self.pair_results[pair] = result_df

            pnl_col = result_df["pair_pnl"].rename(f"pnl_{pair}")
            pair_pnl_frames.append(pnl_col)
            all_dates.update(result_df.index)

        global_index = pd.Index(sorted(all_dates))
        pnl_df = pd.DataFrame(index=global_index)

        for s in pair_pnl_frames:
            pnl_df[s.name] = s.reindex(global_index).fillna(0.0)

        pnl_df["total_pnl"] = pnl_df.sum(axis=1)
        pnl_df["equity"] = self.capital + pnl_df["total_pnl"].cumsum()

        self.portfolio = pnl_df
        return pnl_df

    def summary(self, trading_days_per_year=252):
        if self.portfolio is None:
            raise ValueError("Run the backtest first with .run()")

        eq = self.portfolio["equity"]

        if eq.empty:
            return {}

        total_return = eq.iloc[-1] / eq.iloc[0] - 1.0

        ret = eq.pct_change().dropna()
        if ret.empty:
            ann_ret = ann_vol = sharpe = 0.0
        else:
            avg_daily = ret.mean()
            vol_daily = ret.std()

            ann_ret = (1 + avg_daily) ** trading_days_per_year - 1
            ann_vol = vol_daily * np.sqrt(trading_days_per_year)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        roll_max = eq.cummax()
        drawdown = eq / roll_max - 1.0
        max_dd = drawdown.min()

        return {
            "total_return": float(total_return),
            "annualized_return": float(ann_ret),
            "annualized_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
        }






def stats_to_table(stats_dict):

    df = pd.DataFrame.from_dict(stats_dict, orient="index", columns=["Value"])

    df["Value"] = df["Value"].astype("object")

    df.loc["total_return", "Value"]          = f"{float(stats_dict['total_return'])*100:.2f}%"
    df.loc["annualized_return", "Value"]     = f"{float(stats_dict['annualized_return'])*100:.2f}%"
    df.loc["annualized_vol", "Value"]        = f"{float(stats_dict['annualized_vol'])*100:.2f}%"
    df.loc["sharpe", "Value"]                = f"{float(stats_dict['sharpe']):.2f}"
    df.loc["max_drawdown", "Value"]          = f"{float(stats_dict['max_drawdown'])*100:.2f}%"

    return df