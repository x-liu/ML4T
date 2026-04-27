import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
import marketsimcode as msc
import pandas as pd
from ManualStrategy import ManualStrategy
import experiment1
import experiment2

def getStats(control, experiment, outputFileName):
    # Get portfolio stats
    start_date = control.index.min()
    end_date = control.index.max()

    control = control / control.iloc[0]
    experiment = experiment / experiment.iloc[0]

    dailyReturns = (control / control.shift(1)) - 1
    dailyReturns = dailyReturns[1:]
    actual_returns = (experiment / experiment.shift(1)) - 1
    actual_returns = actual_returns[1:]

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        control.iloc[-1] - 1,
        np.mean(dailyReturns),
        np.std(dailyReturns, ddof=1),
        np.sqrt(252) * np.mean(dailyReturns) / np.std(dailyReturns, ddof=1),
    ]
    cum_ret_actual, avg_daily_ret_actual, std_daily_ret_actual, sharpe_ratio_actual = [
        experiment.iloc[-1] - 1,
        np.mean(actual_returns),
        np.std(actual_returns, ddof=1),
        np.sqrt(252) * np.mean(actual_returns) / np.std(actual_returns, ddof=1),
    ]

    with open(outputFileName, "w") as file:
        file.write((f"Date Range: {start_date} to {end_date}\n"))
        file.write(f"Sharpe Ratio of Benchmark: {sharpe_ratio:.6f}\n")
        file.write(f"Sharpe Ratio of Manual : {sharpe_ratio_actual:.6f}\n")
        file.write(f"Cumulative Return of Benchmark: {cum_ret:.6f}\n")
        file.write(f"Cumulative Return of Manual : {cum_ret_actual:.6f}\n")
        file.write(f"Standard Deviation of Benchmark: {std_daily_ret:.6f}\n")
        file.write(f"Standard Deviation of Manual : {std_daily_ret_actual:.6f}\n")
        file.write(f"Average Daily Return of Benchmark: {avg_daily_ret:.6f}\n")
        file.write(f"Average Daily Return of Manual : {avg_daily_ret_actual:.6f}\n")
        file.write(f"Final Benchmark Portfolio Value: {control.iloc[-1]:.6f}\n")
        file.write(f"Final Manual Portfolio Value: {experiment.iloc[-1]:.6f}\n")


def plot_chart(portvals, bench_vals, df_trades, title, fname):
    symbol = df_trades.columns[0]

    # normalize
    portvals = portvals / portvals.iloc[0]
    bench_vals = bench_vals / bench_vals.iloc[0]

    # derive positions from trades, then find entry days
    positions = df_trades[symbol].cumsum()
    prev = positions.shift(1).fillna(0)
    long_entries = positions.index[(positions == 1000) & (prev != 1000)]
    short_entries = positions.index[(positions == -1000) & (prev != -1000)]

    plt.figure(figsize=(12, 6))
    plt.plot(portvals, color='red', label='Manual Strategy')
    plt.plot(bench_vals, color='purple', label='Benchmark')
    for d in long_entries:
        plt.axvline(d, color='blue', linestyle='--', alpha=0.6)
    for d in short_entries:
        plt.axvline(d, color='black', linestyle='--', alpha=0.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.savefig(fname)
    plt.close()

def indicator_plot(title, data, label, figName):
    plt.figure()
    plt.title(title)
    plt.plot(data, label=label)
    plt.xlabel("Date")
    plt.ylabel("Indicator Value")
    plt.legend()
    plt.savefig(figName)

if __name__ == "__main__":
    np.random.seed(903011732)
    sd_in_sample = dt.datetime(2008, 1, 1)
    ed_in_sample = dt.datetime(2009, 12, 31)
    sd_out_sample = dt.datetime(2010, 1, 1)
    ed_out_sample = dt.datetime(2011, 12, 31)

    startVal = 100000
    commission = 9.95
    impact = 0.005
    ms = ManualStrategy()

    symbol = "JPM"

    #in-sample
    df_in_sample = ms.testPolicy(symbol=symbol, sd=sd_in_sample, ed=ed_in_sample, sv = startVal)
    pv_in_sample = msc.compute_portvals(df_in_sample, start_val = startVal, commission = commission, impact = impact)
    control_is_trades = pd.DataFrame(0, index = df_in_sample.index, columns = df_in_sample.columns)
    control_is_trades.iloc[0,0] = 1000

    #out-of-sample
    df_out_sample = ms.testPolicy(symbol=symbol, sd=sd_out_sample, ed=ed_out_sample, sv = startVal)
    pv_out_sample = msc.compute_portvals(df_out_sample, start_val = startVal, commission = commission, impact = impact)
    control_os_trades = pd.DataFrame(0, index = df_out_sample.index, columns = df_out_sample.columns)
    control_os_trades.iloc[0,0] = 1000

    bm_in_sample = msc.compute_portvals(control_is_trades, start_val=startVal, commission=commission, impact=impact)
    bm_out_sample = msc.compute_portvals(control_os_trades, start_val=startVal, commission=commission, impact=impact)

    plot_chart(pv_in_sample, bm_in_sample, df_in_sample, title=f"{symbol} In-Sample {sd_in_sample.year}-{ed_in_sample.year}" , fname=f"images/manual_in_sample.png")
    plot_chart(pv_out_sample, bm_out_sample, df_out_sample, title = f"{symbol} Out-of-Sample {sd_out_sample.year}-{ed_out_sample.year}", fname=f"images/manual_out_sample.png")

    getStats(control=bm_in_sample, experiment=pv_in_sample, outputFileName="images/InSampleManualStrategyResults.txt")
    getStats(control=bm_out_sample, experiment=pv_out_sample, outputFileName="images/OutSampleManualStrategyResults.txt")

    experiment1.run()
    experiment2.run()


def author():
    return "xliu397"
def study_group():
    return "xliu397"