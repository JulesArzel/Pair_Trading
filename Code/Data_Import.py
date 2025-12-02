import yfinance as yf
import pandas as pd 
import numpy as np


class DataImportEnginnering():

    def __init__(self, stock_list, start_date, end_date, interval="1d", column="Close"):
        self.stock_list = stock_list
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.column = column

    def load_raw(self):
        data = {}
        for stock in self.stock_list:
            df = yf.download(
                stock,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                auto_adjust=False
            )[self.column]

            if df is None or len(df) < 200:
                print(f"Dropping {stock}: insufficient data ({len(df)} rows)")
                continue
        
            data[stock] = df

        return data

    def to_dataframe(self, rawdict):
        df = pd.concat(rawdict.values(), axis=1, join="outer")
        df.columns = list(rawdict.keys())
        return df

    def clean(self, df):
        df = df.ffill()
        return df

    def load(self):
        raw = self.load_raw()
        df = self.to_dataframe(raw)
        df = self.clean(df)
        return df
    
class ReturnCalculator():
    
    @staticmethod
    def log_returns(prices):
        returns = np.log(prices / prices.shift(1))
        returns = returns.dropna()
        return returns

    @staticmethod
    def pct_returns(prices):
        returns = prices.pct_change()
        returns = returns.dropna()
        return returns



    
if __name__ == '__main__':
    stocks = ['AAPL','AMZN']
    start = '2020-01-01'
    end = '2025-01-01'
    interval = '1d'
    column = 'Close'
    data = DataImportEnginnering(stocks,start,end,interval,column)
    print(data.load())
    