import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class DataVisualization():

    def __init__(self):
        pass

    # ================================================================
    # ----------- INTERNAL GENERIC DROPDOWN BUILDER ------------------
    # ================================================================
    def _dropdown_plot(self, traces, labels, title, x_title, y_title):
        fig = go.Figure()

        # Add all traces
        for t in traces:
            fig.add_trace(t)

        n = len(traces)
        buttons = []

        # One button per trace
        for i in range(n):
            visibility = [False] * n
            visibility[i] = True
            buttons.append(
                dict(
                    label=labels[i],
                    method="update",
                    args=[{"visible": visibility}]
                )
            )

        # Show all / show none
        buttons.append(dict(
            label="Show All",
            method="update",
            args=[{"visible": [True] * n}]
        ))

        buttons.append(dict(
            label="Hide All",
            method="update",
            args=[{"visible": [False] * n}]
        ))

        fig.update_layout(
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "x": 0.5,
                "y": 1.12,
                "xanchor": "center",
                "yanchor": "top",
                "showactive": True
            }]
        )

        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            hovermode="x unified"
        )

        return fig

    # ================================================================
    # --------------------- HEATMAPS --------------------------------
    # ================================================================
    def heatmap(self, matrix, title="Heatmap", xlabel="", ylabel=""):
        plt.figure(figsize=(8,6))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(matrix, cmap=cmap, linewidths=0.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    # ================================================================
    # --------------------- PRICE SERIES -----------------------------
    # ================================================================
    def plot_prices(self, prices):
        traces = []
        labels = []

        for col in prices.columns:
            traces.append(go.Scatter(
                x=prices.index,
                y=prices[col],
                mode="lines",
                name=col
            ))
            labels.append(col)

        return self._dropdown_plot(
            traces,
            labels,
            "Price Series",
            "Date",
            "Price"
        )

    # ================================================================
    # --------------------- RETURNS PLOT -----------------------------
    # ================================================================
    def plot_returns(self, returns):
        traces = []
        labels = []

        for col in returns.columns:
            traces.append(go.Scatter(
                x=returns.index,
                y=returns[col],
                mode="lines",
                name=col
            ))
            labels.append(col)

        return self._dropdown_plot(
            traces,
            labels,
            "Returns",
            "Date",
            "Returns"
        )

    # ================================================================
    # --------------------- PAIR HISTORICAL --------------------------
    # ================================================================
    def plot_pair_prices(self, pair_dict):
        """
        pair_dict: { (A,B): DataFrame with columns [A, B] }
        """
        traces = []
        labels = []

        for pair, df in pair_dict.items():
            A, B = pair
            traces.append(go.Scatter(
                x=df.index,
                y=df[A],
                mode="lines",
                name=f"{A} (pair {pair})"
            ))
            traces.append(go.Scatter(
                x=df.index,
                y=df[B],
                mode="lines",
                name=f"{B} (pair {pair})"
            ))
            labels.append(str(pair))
            labels.append(str(pair))

        return self._dropdown_plot(
            traces,
            labels,
            "Pair Historical Prices",
            "Date",
            "Price"
        )

    # ================================================================
    # ----------------------- PCA SCREE PLOT -------------------------
    # ================================================================
    def pca_scree(self, pca):
        """
        pca: fitted PCA object
        """
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
            y=pca.explained_variance_ratio_,
            name="Explained Variance"
        ))

        fig.update_layout(
            title="PCA Scree Plot",
            xaxis_title="Principal Component",
            yaxis_title="Explained Variance Ratio"
        )

        return fig

    # ================================================================
    # ---------------- PCA 2D SCATTER ON RETURNS ---------------------
    # ================================================================
    def pca_scatter(self, returns, cluster_labels=None):
        from sklearn.decomposition import PCA

        returns_clean = returns.dropna()
        tickers = returns_clean.columns.tolist()

        # PCA for visualization (2 components)
        pca = PCA(n_components=2)
        scores_2d = pca.fit_transform(returns_clean.T)

        if cluster_labels is None:
            cluster_labels = ["All"] * len(tickers)

        fig = go.Figure()

        # ----- GROUP POINTS BY CLUSTER -----
        unique_clusters = sorted(set(cluster_labels))

        for cl in unique_clusters:
            idx = [i for i, lab in enumerate(cluster_labels) if lab == cl]

            fig.add_trace(go.Scatter(
                x=scores_2d[idx, 0],
                y=scores_2d[idx, 1],
                mode="markers+text",
                text=[tickers[i] for i in idx],
                textposition="top center",
                name=f"Cluster {cl}",
                marker=dict(size=10),
            ))

        fig.update_layout(
            title="PCA Projection (Colored by Cluster)",
            xaxis_title="PC1",
            yaxis_title="PC2",
            legend_title="Clusters",
        )

        return fig

    # ================================================================
    # ----------------------- SPREAD PLOT ----------------------------
    # ================================================================
    def plot_spread(self, spread_dict):
        """
        spread_dict: { (A,B): DataFrame with column 'spread' or 'Delta_norm' }
        """
        traces = []
        labels = []

        for pair, df in spread_dict.items():
            col = "Delta_norm" if "Delta_norm" in df.columns else "spread"

            traces.append(go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=str(pair)
            ))
            labels.append(str(pair))

        return self._dropdown_plot(
            traces,
            labels,
            "Spread (Normalized)",
            "Date",
            col
        )

    # ================================================================
    # -------- SPREAD + SIGNALS --------------------------------------
    # ================================================================
    def plot_spread_with_signals(self, signals_dict):

        fig = go.Figure()
        all_traces = []
        pair_trace_counts = {}   
        total_traces = 0

        for pair, df in signals_dict.items():

            yname = "spread"
            y = df[yname]

            traces_for_pair = []

            traces_for_pair.append(go.Scatter(
                x=df.index, y=y, mode="lines",
                name=f"{pair} spread",
                visible=False
            ))

            long_ent = df[df["signal"] == 1]
            traces_for_pair.append(go.Scatter(
                x=long_ent.index,
                y=y.loc[long_ent.index],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="green"),
                name=f"{pair} long entry",
                visible=False
            ))

            short_ent = df[df["signal"] == -1]
            traces_for_pair.append(go.Scatter(
                x=short_ent.index,
                y=y.loc[short_ent.index],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                name=f"{pair} short entry",
                visible=False
            ))

            exit_mask = (df["position"] == 0) & (df["signal"] != 0)
            exit_pts = df[exit_mask]
            traces_for_pair.append(go.Scatter(
                x=exit_pts.index,
                y=y.loc[exit_pts.index],
                mode="markers",
                marker=dict(symbol="x", size=10, color="black"),
                name=f"{pair} exit",
                visible=False
            ))

            if "margin_call" in df.columns:
                mc = df[df["margin_call"] == True]
                traces_for_pair.append(go.Scatter(
                    x=mc.index,
                    y=y.loc[mc.index],
                    mode="markers",
                    marker=dict(symbol="circle", size=12, color="orange"),
                    name=f"{pair} margin call",
                    visible=False
                ))

            for t in traces_for_pair:
                fig.add_trace(t)

            pair_trace_counts[pair] = len(traces_for_pair)
            total_traces += len(traces_for_pair)

        buttons = []
        trace_start = 0

        for pair, count in pair_trace_counts.items():
            vis = [False] * total_traces
            for i in range(trace_start, trace_start + count):
                vis[i] = True

            buttons.append(dict(
                label=str(pair),
                method="update",
                args=[{"visible": vis}]
            ))

            trace_start += count

        buttons.append(dict(
            label="Show All",
            method="update",
            args=[{"visible": [True] * total_traces}]
        ))

        buttons.append(dict(
            label="Hide All",
            method="update",
            args=[{"visible": [False] * total_traces}]
        ))

        fig.update_layout(
            title="Spread with Trading Signals",
            xaxis_title="Date",
            yaxis_title="Spread",
            hovermode="x unified",
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.5,
                "y": 1.15,
                "xanchor": "center",
                "yanchor": "top"
            }]
        )

        return fig






def plot_full_tearsheet(portfolio_df):
    eq = portfolio_df["equity"]
    returns = eq.pct_change().fillna(0)
    roll_max = eq.cummax()
    drawdown = eq / roll_max - 1
    rolling_vol = returns.rolling(20).std() * (252 ** 0.5)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # --- Equity curve ---
    axes[0].plot(eq.index, eq, linewidth=2, color="darkblue")
    axes[0].set_title("Portfolio Equity Curve", fontsize=18, pad=15)
    axes[0].set_ylabel("Equity ($)", fontsize=14)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # --- Daily returns ---
    axes[1].plot(returns.index, returns, linewidth=1, color="grey")
    axes[1].set_title("Daily Returns", fontsize=16, pad=10)
    axes[1].set_ylabel("Returns", fontsize=14)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # --- Rolling volatility ---
    axes[2].plot(rolling_vol.index, rolling_vol, linewidth=2, color="orange")
    axes[2].set_title("20-Day Rolling Volatility (Annualized)", fontsize=16, pad=10)
    axes[2].set_ylabel("Volatility", fontsize=14)
    axes[2].set_xlabel("Date", fontsize=14)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
