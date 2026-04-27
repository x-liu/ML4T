import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
import marketsimcode as msc
import pandas as pd
from StrategyLearner import StrategyLearner
from ManualStrategy import ManualStrategy

def getStats(benchmark, manualStrategy, randomForestStrategy, outputFileName):
    # Get portfolio stats
    start_date = benchmark.index.min()
    end_date = benchmark.index.max()

    benchmark = benchmark / benchmark.iloc[0]
    manualStrategy = manualStrategy / manualStrategy.iloc[0]
    randomForestStrategy = randomForestStrategy / randomForestStrategy.iloc[0]

    bmDailyReturns = (benchmark / benchmark.shift(1)) - 1
    bmDailyReturns = bmDailyReturns[1:]
    manualDailyReturns = (manualStrategy / manualStrategy.shift(1)) - 1
    manualDailyReturns = manualDailyReturns[1:]
    randomForestReturns = (randomForestStrategy / randomForestStrategy.shift(1)) - 1
    randomForestReturns = randomForestReturns[1:]

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        benchmark.iloc[-1] - 1,
        np.mean(bmDailyReturns),
        np.std(bmDailyReturns, ddof=1),
        np.sqrt(252) * np.mean(bmDailyReturns) / np.std(bmDailyReturns, ddof=1),
    ]
    cum_ret_actual, avg_daily_ret_actual, std_daily_ret_actual, sharpe_ratio_actual = [
        manualStrategy.iloc[-1] - 1,
        np.mean(manualDailyReturns),
        np.std(manualDailyReturns, ddof=1),
        np.sqrt(252) * np.mean(manualDailyReturns) / np.std(manualDailyReturns, ddof=1),
    ]
    rf_cum_ret_actual, rf_avg_daily_ret_actual, rf_std_daily_ret_actual, rf_sharpe_ratio_actual = [
        randomForestStrategy.iloc[-1] - 1,
        np.mean(randomForestReturns),
        np.std(randomForestReturns, ddof=1),
        np.sqrt(252) * np.mean(randomForestReturns) / np.std(randomForestReturns, ddof=1),
    ]

    with open(outputFileName, "w") as file:
        file.write((f"Date Range: {start_date} to {end_date}\n"))
        file.write(f"Sharpe Ratio of Benchmark: {sharpe_ratio:.6f}\n")
        file.write(f"Sharpe Ratio of manual Learner : {sharpe_ratio_actual:.6f}\n")
        file.write(f"Sharpe Ratio of Random Forest Learner : {rf_sharpe_ratio_actual:.6f}\n")
        file.write(f"Cumulative Return of Benchmark: {cum_ret:.6f}\n")
        file.write(f"Cumulative Return of manual Learner : {cum_ret_actual:.6f}\n")
        file.write(f"Cumulative Return of Random Forest Learner : {rf_cum_ret_actual:.6f}\n")
        file.write(f"Standard Deviation of Benchmark: {std_daily_ret:.6f}\n")
        file.write(f"Standard Deviation of manual Learner : {std_daily_ret_actual:.6f}\n")
        file.write(f"Standard Deviation of Random Forest Learner : {rf_std_daily_ret_actual:.6f}\n")
        file.write(f"Average Daily Return of Benchmark: {avg_daily_ret:.6f}\n")
        file.write(f"Average Daily Return of manual Learner : {avg_daily_ret_actual:.6f}\n")
        file.write(f"Average Daily Return of Random Forest Learner : {rf_avg_daily_ret_actual:.6f}\n")
        file.write(f"Final Benchmark Portfolio Value: {benchmark.iloc[-1]:.6f}\n")
        file.write(f"Final Manual Strategy Learner Portfolio Value: {manualStrategy.iloc[-1]:.6f}\n")
        file.write(f"Final Random Forest Learner Portfolio Value: {randomForestStrategy.iloc[-1]:.6f}\n")


def plot_chart(ms_vals, bench_vals, rf_vals, title, fname):

    # normalize
    ms_vals = ms_vals / ms_vals.iloc[0]
    bench_vals = bench_vals / bench_vals.iloc[0]
    rf_vals = rf_vals / rf_vals.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(ms_vals, color='red', label='Manual Strategy')
    plt.plot(bench_vals, color='purple', label='Benchmark')
    plt.plot(rf_vals, color='blue', label='Random Forest Strategy')
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

def run():
    sd_in_sample = dt.datetime(2008, 1, 1)
    ed_in_sample = dt.datetime(2009, 12, 31)
    sd_out_sample = dt.datetime(2010, 1, 1)
    ed_out_sample = dt.datetime(2011, 12, 31)

    startVal = 100000
    commission = 9.95
    impact = 0.005
    symbol = "JPM"

    ms = ManualStrategy()
    randForestStrategy = StrategyLearner(verbose=False, impact=impact, commission=commission)
    randForestStrategy.add_evidence(symbol=symbol, sd = sd_in_sample, ed=ed_in_sample)

    #in-sample
    ms_df_in_sample = ms.testPolicy(symbol=symbol, sd=sd_in_sample, ed=ed_in_sample, sv = startVal)
    ms_pv_in_sample = msc.compute_portvals(ms_df_in_sample, start_val = startVal, commission = commission, impact = impact)
    control_is_trades = pd.DataFrame(0, index = ms_df_in_sample.index, columns = ms_df_in_sample.columns)
    control_is_trades.iloc[0,0] = 1000

    #out-of-sample
    ms_df_out_sample = ms.testPolicy(symbol=symbol, sd=sd_out_sample, ed=ed_out_sample, sv = startVal)
    ms_pv_out_sample = msc.compute_portvals(ms_df_out_sample, start_val = startVal, commission = commission, impact = impact)
    control_os_trades = pd.DataFrame(0, index = ms_df_out_sample.index, columns = ms_df_out_sample.columns)
    control_os_trades.iloc[0,0] = 1000

    #in-sample
    rf_df_in_sample = randForestStrategy.testPolicy(symbol=symbol, sd=sd_in_sample, ed=ed_in_sample, sv = startVal)
    rf_pv_in_sample = msc.compute_portvals(rf_df_in_sample, start_val = startVal, commission = commission, impact = impact)
    control_is_trades = pd.DataFrame(0, index = rf_df_in_sample.index, columns = rf_df_in_sample.columns)
    control_is_trades.iloc[0,0] = 1000

    #out-of-sample
    rf_df_out_sample = randForestStrategy.testPolicy(symbol=symbol, sd=sd_out_sample, ed=ed_out_sample, sv = startVal)
    rf_pv_out_sample = msc.compute_portvals(rf_df_out_sample, start_val = startVal, commission = commission, impact = impact)
    control_os_trades = pd.DataFrame(0, index = rf_df_out_sample.index, columns = rf_df_out_sample.columns)
    control_os_trades.iloc[0,0] = 1000

    bm_in_sample = msc.compute_portvals(control_is_trades, start_val=startVal, commission=commission, impact=impact)
    bm_out_sample = msc.compute_portvals(control_os_trades, start_val=startVal, commission=commission, impact=impact)

    plot_chart(ms_pv_in_sample, bm_in_sample, rf_pv_in_sample, title=f"{symbol} In-Sample {sd_in_sample.year}-{ed_in_sample.year}" , fname=f"images/randForest_in_sample.png")
    plot_chart(ms_pv_out_sample, bm_out_sample, rf_pv_out_sample, title = f"{symbol} Out-of-Sample {sd_out_sample.year}-{ed_out_sample.year}", fname=f"images/randForest_out_sample.png")

    getStats(benchmark=bm_in_sample, manualStrategy=ms_pv_in_sample, randomForestStrategy=rf_pv_in_sample, outputFileName="images/InSampleRandomForestStrategyResults.txt")
    getStats(benchmark=bm_out_sample,  manualStrategy=ms_pv_out_sample,randomForestStrategy=rf_pv_out_sample, outputFileName="images/OutSampleRandomForestStrategyResults.txt")


def author():
    return "xliu397"
def study_group():
    return "xliu397"

if __name__ == "__main__":
    run()