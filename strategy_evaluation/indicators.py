# This code is illustrative only.  It demonstrates that if one thinks
# of Indicators as a class, then you should have exactly five
# public indicators.
#
# Rules:
#  1. There must be exactly 5 public indicators
#  2. You may have unlimited private methods and private indicators
#  3. Public indicators must return single results vectors
#  4. NO PLOTTING anOR CHARTING from within the Indicators class.
#  5. Code external to the Indicators class can only call the 5 public indicators.
#  6. You may invoke the Public indicators with one argument set alone
#     However, indicators can have as many parameters as you want. What this means is
#     if you call SMA(30, startDate, endDate),
#     you cannot also call SMA(15, startDate, EndDate), when we get to P8. If this is
#     is your intent, then you need to makae a custom indicator:
#     like mySMA(30,15, StartDate, EndDate), and mySMA() can call SMA as many
#     times as desired.
#  7. Private methods can be invoked multiple times and with different arguments.
#  8. You can change the values of your arguments in P8. They do not need to match
#     what is used in P6.
#
# You are not required to implement Indicators as a class.
#
import pandas as pd
from util import get_data

class Indicators:
    def __init__(self):
        # Initialize any necessary variables or data structures here
        pass

    def rsi(self, symbol,sd,ed, lookback):
        prices = self._getPrices(symbol, sd, ed)
        delta= prices-prices.shift(1)
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss

        avg_gain = gain.rolling(window=lookback).mean()
        avg_loss = loss.rolling(window=lookback).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi[symbol]

    def momentum(self, symbol, sd, ed, lookback):
        # Public method ind2
        prices= self._getPrices(symbol,sd,ed)
        roc_df = (prices / prices.shift(lookback)) - 1
        return roc_df[symbol]

    def ema(self, symbol, sd, ed, lookback):
        # Public method ind3
        # Implement your logic here
        prices = self._getPrices(symbol,sd,ed)
        ret = prices/self._ema(prices, lookback)
        return ret[symbol]

    def macd(self, symbol, sd, ed):
        # Public method ind4
        # Implement your logic here
        prices = self._getPrices(symbol,sd,ed)
        macd_line = self._ema(prices, 12) - self._ema(prices, 26)
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        ret = macd_line - signal_line
        return ret[symbol]

    def bollingerBands(self, symbol, sd, ed, lookback):
        # Public method ind5
        # Implement your logic here
        prices = self._getPrices(symbol,sd,ed)
        sma = prices.rolling(window=lookback).mean()
        std = prices.rolling(window=lookback).std()
        upper = sma + (2*std)
        lower = sma - (2*std)
        bollingerBands = (prices-lower)/(upper-lower)
        return bollingerBands[symbol]

    # Unlimited PRIVATE methods and indicators
    # These can only be called by methods and indicators in this class
    #
    def _getPrices(self, symbol, sd, ed):
        # Private method 1
        # Implement any helper functions/indicators or logic here
        prices = get_data([symbol], pd.date_range(sd, ed), addSPY=True)
        prices = prices.drop(columns=['SPY'])
        prices.ffill(inplace=True, axis=0)
        prices.bfill(inplace=True, axis=0)
        return prices

    def _ema(self, prices, lookback):
        # Private method 2
        # Implement any helper functions/indicators or logic here
        return prices.ewm(span=lookback, adjust=False).mean()

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
