import pandas as pd
from pandas import DataFrame as df
from util import get_data
import datetime as dt


def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    prices = get_data([symbol], pd.date_range(sd,ed),addSPY=False)
    prices.ffill(inplace=True, axis=0)
    prices.bfill(inplace=True, axis=0)
    df_trades = pd.DataFrame(0, index = prices.index, columns = [symbol])
    activePos = 0
    for i in range(len(prices)):
        if (i+1 >= prices.size):
            if activePos != 0:
                df_trades.iloc[i,0] = activePos*-1
                activePos = 0
        else:
            if (prices.iloc[i+1, 0] > prices.iloc[i,0]):
                df_trades.iloc[i,0] = 1000 - activePos
                activePos = 1000
            elif (prices.iloc[i+1,0] < prices.iloc[i,0]):
                df_trades.iloc[i,0] = -1000 - activePos
                activePos = -1000
    return df_trades

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