# Pair Trading Research Project

This repository contains a complete research pipeline for building, evaluating, and backtesting statistical arbitrage strategies based on equity pair trading. It implements a full end-to-end workflow, starting from data collection and preprocessing, continuing through pair selection and signal generation, and ending with a complete multi-pair backtester.

This project is developed by two students from Université Paris Dauphine–PSL as part of a quantitative finance research initiative.

## Overview

Pair trading is a market-neutral strategy that exploits temporary divergences between two historically related assets. The goal is to identify pairs of stocks whose spread exhibits mean reversion, construct a hedged spread that neutralizes market exposure, generate systematic trading signals based on statistical deviations, and evaluate the resulting strategy through a robust backtesting engine.

The focus of this project is to implement a clean, modular, and reproducible pipeline that includes:

- reliable data engineering for equity price series  
- rigorous pair selection based on both statistical and clustering methods  
- spread construction using beta-neutral or dollar-neutral hedging  
- signal generation based on Z-scores, stop-loss rules, and time stops  
- a multi-pair backtester with capital allocation and margin-call mechanics  
- visualization tools for spreads, signals, PnL, and clustering results  

## Methodology

### 1. Data Collection and Preprocessing

Historical equity prices are retrieved from Yahoo Finance. The preprocessing step includes timestamp alignment, forward-filling missing values, filtering out illiquid tickers, and computing log-returns for PCA and clustering.

A curated list of robust S&P stocks is included to ensure consistent data availability.

### 2. Pair Selection

Candidate pairs are identified using two complementary approaches:

- Sector-neutral distance metrics (correlation, Euclidean distance on normalized prices) within each GICS sector.  
- PCA + K-means clustering on returns, then pair selection within clusters.

Each candidate pair is validated using:

- Engle–Granger cointegration test  
- Hurst exponent estimation  
- half-life estimation through AR(1) / OU approximation  

Only statistically consistent pairs are retained.

### 3. Spread Construction

For every validated pair (A, B), a hedged spread is built using either:

Beta-neutral spread:  
A − beta × B

Dollar-neutral spread:  
A − B

The spread is standardized via rolling Z-score so that signals are based on statistical deviations relative to its long-run mean.

### 4. Signal Generation

Trading signals follow a deterministic rule set:

- Enter long when Z ≤ −entry threshold  
- Enter short when Z ≥ entry threshold  
- Exit when |Z| ≤ exit threshold  
- Stop loss when Z exceeds a specified limit  
- Time stop based on a multiple of the pair’s half-life  

The engine ensures that only one position per pair is active at any time.

Each signal includes a textual reason indicating the event (entry, exit, stop-loss, time stop).

### 5. Backtesting Framework

The backtester simulates the performance of multiple pairs traded simultaneously. Features include:

- equal allocation of initial capital across pairs  
- position sizing based on the hedge ratio at trade entry  
- PnL derived from daily changes in the hedged spread  
- margin-call logic: if a pair’s equity falls to zero, trading on that pair stops  
- aggregation of PnL across all pairs into a portfolio equity curve  

Performance metrics include total return, annualized return, volatility, Sharpe ratio, and maximum drawdown.

### 6. Visualization

The project includes a visualization module offering:

- price and return series  
- PCA scatter plots with cluster assignments  
- spread charts with entry and exit markers  
- pair-by-pair backtest results  
- equity curve visualization  

These plots support analysis and interpretation of the trading strategy.

## Getting Started

Clone the repository

Install dependencies:
pip install -r requirements.txt

Open the main notebook to run the full workflow, including data engineering, pair selection, spread construction, signal generation, and backtesting.

## Extensions and Future Work

Potential directions for further research include:

-Stochastic Control Approach <br>
-Make sure the pair holds one by checking if there are any ruptures in the beta between the stocks <br>
-Make the strategy regime adaptive, not enter at the same level for different volatility regimes <br>
-Volatility scaled allocation, allocate less to volatile spreads <br>
-Add transaction costs <br>
-Don't allow negative equity in a pair, shut it down if I lose all the capital <br>
-Better assess performance by obviously seperating the data used to select pairs and the one used for backtest

## Contact

Jules Arzel  
Email: jules.arzel@dauphine.eu  
LinkedIn: https://www.linkedin.com/in/jules-arzel

Antoine Battini  
Email: antoine.battini@dauphine.eu  
LinkedIn: https://www.linkedin.com/in/antoine-battini




