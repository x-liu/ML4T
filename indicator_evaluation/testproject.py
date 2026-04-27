import pandas as pd
import numpy as np
import TheoreticallyOptimalStrategy as tos
import datetime as dt
import marketsimcode as msc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from indicators import Indicators


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "xliu397"  # Change this to your user ID

def study_group():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "xliu397"  # Change this to your user ID

def getStats(optimalPortVals, actualPortVals):
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = optimalPortVals.index.min()
    end_date = optimalPortVals.index.max()

    optimalPortVals = optimalPortVals / optimalPortVals.iloc[0]
    actualPortVals = actualPortVals / actualPortVals.iloc[0]

    dailyReturns = (optimalPortVals / optimalPortVals.shift(1)) - 1
    dailyReturns = dailyReturns[1:]
    actual_returns = (actualPortVals / actualPortVals.shift(1)) - 1
    actual_returns = actual_returns[1:]

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        optimalPortVals.iloc[-1] - 1,
        np.mean(dailyReturns),
        np.std(dailyReturns, ddof=1),
        np.sqrt(252) * np.mean(dailyReturns) / np.std(dailyReturns, ddof=1),
    ]
    cum_ret_actual, avg_daily_ret_actual, std_daily_ret_actual, sharpe_ratio_actual = [
        actualPortVals.iloc[-1] - 1,
        np.mean(actual_returns),
        np.std(actual_returns, ddof=1),
        np.sqrt(252) * np.mean(actual_returns) / np.std(actual_returns, ddof=1),
    ]
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio:.6f}")
    print(f"Sharpe Ratio of actual : {sharpe_ratio_actual:.6f}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret:.6f}")
    print(f"Cumulative Return of actual : {cum_ret_actual:.6f}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret:.6f}")
    print(f"Standard Deviation of actual : {std_daily_ret_actual:.6f}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret:.6f}")
    print(f"Average Daily Return of actual : {avg_daily_ret_actual:.6f}")
    print()
    print(f"Final Optimal Portfolio Value: {optimalPortVals.iloc[-1]:.6f}")
    print(f"Final Actual Portfolio Value: {actualPortVals.iloc[-1]:.6f}")
    plt.figure()
    plt.title("JPM Benchmark vs Theoretical Optimal Portfolio")
    plt.plot(optimalPortVals, label='Theoretical Optimal Portfolio', color='red')
    plt.plot(actualPortVals, label='Benchmark Portfolio', color='purple')
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("Figure1.png")

def indicator_plot(title, data, label, figName):
    plt.figure()
    plt.title(title)
    plt.plot(data, label=label)
    plt.xlabel("Date")
    plt.ylabel("Indicator Value")
    plt.legend()
    plt.savefig(figName)

if __name__ == "__main__":
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv = 100000)
    control_trades = pd.DataFrame(0, index = df_trades.index, columns = df_trades.columns)
    control_trades.iloc[0,0] = 1000
    portvals = msc.compute_portvals(df_trades)
    control_portvals = msc.compute_portvals(control_trades)
    getStats(optimalPortVals=portvals, actualPortVals=control_portvals)
    #indicators
    ind = Indicators()
    rsi = ind.rsi(symbol="JPM", sd=sd, ed=ed, lookback=14)
    momentum = ind.momentum(symbol="JPM", sd=sd, ed=ed, lookback=14)
    ema = ind.ema(symbol="JPM", sd=sd, ed=ed, lookback=14)
    macd = ind.macd(symbol="JPM", sd=sd, ed=ed)
    bollingerBands = ind.bollingerBands(symbol="JPM", sd=sd, ed=ed, lookback=14)
    indicator_plot("JPM RSI Indicator 2008/1/1 to 2009/12/31", rsi, "RSI", "Figure2.png")
    indicator_plot("JPM Momentum Indicator 2008/1/1 to 2009/12/31", momentum, "Momentum", "Figure3.png")
    indicator_plot("JPM EMA Indicator 2008/1/1 to 2009/12/31", ema, "EMA", "Figure4.png")
    indicator_plot("JPM MACD Indicator 2008/1/1 to 2009/12/31", macd, "MACD Histogram", "Figure5.png")
    indicator_plot("JPM Bollinger Band Indicator 2008/1/1 to 2009/12/31", bollingerBands, "Bollinger Band", "Figure6.png")


