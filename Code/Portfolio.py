import numpy as np
import pandas as pd
import statsmodels.api as sm


class SignalGenerator():

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.0,
        time_stop_k: float = 2.0
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.time_stop_k = time_stop_k

    def generate_signals_for_pair(self, spread_df, half_life):

        if "spread" not in spread_df.columns:
            raise ValueError("spread_df must contain a 'spread' column")

        df = spread_df.copy()

        roll_mean = df["spread"].rolling(60).mean()
        roll_std = df["spread"].rolling(60).std()
        df["z"] = (df["spread"] - roll_mean) / roll_std
        z = df["z"].fillna(0).values

        n = len(df)
        max_holding_days = int(self.time_stop_k * half_life)

        pos = 0
        days_in_pos = 0

        positions = np.zeros(n, dtype=int)
        signals = np.zeros(n, dtype=int)
        reasons = np.array([""] * n, dtype=object)

        for i in range(n):
            zi = z[i]
            signal = 0
            reason = ""

            if pos == 0:

                if zi <= -self.entry_z:
                    pos = 1
                    signal = 1
                    days_in_pos = 1
                    reason = "entry_long"

                elif zi >= self.entry_z:
                    pos = -1
                    signal = -1
                    days_in_pos = 1
                    reason = "entry_short"

            else:
                days_in_pos += 1
                exit_now = False

                if abs(zi) <= self.exit_z:
                    exit_now = True
                    reason = "exit_mean_revert"

                if pos == 1 and zi <= -self.stop_z:
                    exit_now = True
                    reason = "stop_loss"

                elif pos == -1 and zi >= self.stop_z:
                    exit_now = True
                    reason = "stop_loss"

                if days_in_pos >= max_holding_days:
                    exit_now = True
                    reason = "time_stop"

                if exit_now:
                    signal = -pos
                    pos = 0
                    days_in_pos = 0

            positions[i] = pos
            signals[i] = signal
            reasons[i] = reason

        df["position"] = positions
        df["signal"] = signals
        df["signal_reason"] = reasons
        df["half_life"] = half_life
        df["time_stop_limit"] = max_holding_days

        return df

    def generate_signals_for_pairs(self, spread_dict, half_lives):
        out = {}
        for pair, df in spread_dict.items():
            hl = half_lives.get(pair)
            if hl is None:
                print(f"No half-life for pair {pair}, skipping.")
                continue
            out[pair] = self.generate_signals_for_pair(df, hl)
        return out


def estimate_half_life(delta_norm):
    spread_lag = delta_norm.shift(1)
    spread_diff = delta_norm - spread_lag

    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()

    X = sm.add_constant(spread_lag)
    model = sm.OLS(spread_diff, X).fit()

    beta = model.params.iloc[1]
    phi = beta + 1
    if phi <= 0:
        return np.nan

    theta = -np.log(phi)
    return np.log(2) / theta
