import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
import marketsimcode as msc
import pandas as pd
from StrategyLearner import StrategyLearner

def plot_chart(portvals, metrics_df):
    plt.figure(figsize=(12, 6))
    for impact, pv in portvals.items():
        normalized = pv / pv.iloc[0]
        plt.plot(normalized, label=f'impact={impact}')
    plt.legend()
    plt.title('Portfolio Value by Impact (in-sample JPM)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.savefig('images/experiment2_portvals.png')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(metrics_df['impact'].astype(str), metrics_df['cum_return'])
    axes[0].set_title('Cumulative Return vs Impact')
    axes[1].bar(metrics_df['impact'].astype(str), metrics_df['num_trades'])
    axes[1].set_title('# Trades vs Impact')
    axes[0].set_xlabel('Impact')
    axes[0].set_ylabel('Cumulative Return')
    axes[1].set_xlabel('Impact')
    axes[1].set_ylabel('Number of Trades')
    plt.savefig('images/experiment2_metrics.png')
    plt.close()

def run():
    sd_in_sample = dt.datetime(2008, 1, 1)
    ed_in_sample = dt.datetime(2009, 12, 31)

    startVal = 100000
    commission = 0
    symbol = "JPM"

    impactValues = [0, 0.005, 0.01, 0.05]
    portvals = {}
    trades = {}
    metrics = []
    for impact in impactValues:
        randForestStrategy = StrategyLearner(verbose=False, impact=impact, commission=commission)
        randForestStrategy.add_evidence(symbol=symbol, sd = sd_in_sample, ed=ed_in_sample)

        #in-sample
        rf_df_in_sample = randForestStrategy.testPolicy(symbol=symbol, sd=sd_in_sample, ed=ed_in_sample, sv = startVal)
        rf_pv_in_sample = msc.compute_portvals(rf_df_in_sample, start_val = startVal, commission = commission, impact = impact)

        #record metrics
        trades[impact] = rf_df_in_sample
        portvals[impact] = rf_pv_in_sample
        dailyMetrics = (rf_pv_in_sample/rf_pv_in_sample.shift(1) - 1).iloc[1:]
        metrics.append({
            'impact': impact,
            'cum_return': rf_pv_in_sample.iloc[-1]/rf_pv_in_sample.iloc[0]-1,
            'num_trades': (rf_df_in_sample.iloc[:,0] != 0).sum(),
            'sharpe': np.sqrt(252) * dailyMetrics.mean()/dailyMetrics.std(ddof=1),
            'stdev': dailyMetrics.std(ddof = 1)
        })

    metrics_df = pd.DataFrame(metrics)
    with open('images/experiment2_results.txt', 'w') as f:
        f.write(metrics_df.to_string(index=False))

    plot_chart(portvals, metrics_df)



def author():
    return "xliu397"
def study_group():
    return "xliu397"

if __name__ == "__main__":
    run()