import yfinance as yf
import pandas as pd
import numpy as np

def load_data(ticker, period="1y"):
    data = yf.download(ticker, period=period)
    return data

def moving_average_strategy(data, short_window=20, long_window=50):
    data["Short_MA"] = data["Close"].rolling(window=short_window).mean()
    data["Long_MA"] = data["Close"].rolling(window=long_window).mean()
    data["Signal"] = np.where(data["Short_MA"] > data["Long_MA"], 1, -1)
    return data

def backtest(data):
    data["Returns"] = data["Close"].pct_change()
    data["Strategy_Returns"] = data["Returns"] * data["Signal"].shift(1)
    return data

def performance_metrics(data):
    pnl = data["Strategy_Returns"].sum()
    sharpe = (data["Strategy_Returns"].mean() / data["Strategy_Returns"].std()) * np.sqrt(252)
    drawdown = (data["Strategy_Returns"].cumsum().cummax() - data["Strategy_Returns"].cumsum()).max()
    return pnl, sharpe, drawdown

if __name__ == "__main__":
    df = load_data("AAPL")
    df = moving_average_strategy(df)
    df = backtest(df)

    pnl, sharpe, drawdown = performance_metrics(df)

    print("Total P&L:", pnl)
    print("Sharpe Ratio:", sharpe)
    print("Max Drawdown:", drawdown)
