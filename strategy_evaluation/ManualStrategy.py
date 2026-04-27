from util import get_data

import pandas as pd
import datetime as dt
from indicators import Indicators

class ManualStrategy:

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
        prices = get_data([symbol], pd.date_range(sd,ed),addSPY=True)
        prices = prices.drop(columns=['SPY'])
        prices.ffill(inplace=True, axis=0)
        prices.bfill(inplace=True, axis=0)
        #get 3 indicators
        ind = Indicators()
        rsi = ind.rsi(symbol=symbol, sd=sd, ed=ed, lookback=14)
        # momentum = ind.momentum(symbol=symbol, sd=sd, ed=ed, lookback=14)
        ema = ind.ema(symbol=symbol, sd=sd, ed=ed, lookback=14)
        # macd = ind.macd(symbol=symbol, sd=sd, ed=ed)
        bollingerBands = ind.bollingerBands(symbol=symbol, sd=sd, ed=ed, lookback=14)

        rsi_vote = pd.Series(0, index=prices.index)
        rsi_vote[rsi<30] = 1
        rsi_vote[rsi>70] = -1

        boll_vote = pd.Series(0, index=prices.index)
        boll_vote[bollingerBands<0] = 1
        boll_vote[bollingerBands>1] = -1

        ema_vote = pd.Series(0, index=prices.index)
        ema_vote[ema>1.02] = -1
        ema_vote[ema<0.98] = 1

        combined = rsi_vote+boll_vote+ema_vote
        #have a -1, 0, or 1 for trades where the indicator matches criteria
        target = pd.Series(0, index=prices.index)

        target[combined >= 1] = 1000
        target[combined <= -1] = -1000
        df_trades = pd.DataFrame(target.diff().fillna(target.iloc[0]).values,index=prices.index,columns=[symbol])
        return df_trades


if __name__ == "__main__":
    ManualStrategy().testPolicy()

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
