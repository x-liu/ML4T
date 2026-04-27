""""""  		  	   		 		  			  		 			     			  	 
"""  		  	   		 		  			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 		  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 		  			  		 			     			  	 
All Rights Reserved  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 		  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 		  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 		  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 		  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 		  			  		 			     			  	 
or edited.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 		  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 		  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 		  			  		 			     			  	 
GT honor code violation.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
Student Name: Xing Liu (replace with your name)  		  	   		 		  			  		 			     			  	 
GT User ID: xliu397 (replace with your User ID)  		  	   		 		  			  		 			     			  	 
GT ID: 903011732 (replace with your GT ID)  		  	   		 		  			  		 			     			  	 
"""  		  	   		 		    		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import datetime as dt
import random  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import pandas as pd  		  	   		 		  			  		 			     			  	 
import util as ut
from indicators import Indicators
from BagLearner import BagLearner
from RTLearner import RTLearner
import numpy as np


class StrategyLearner(object):
    """  		  	   		 		  			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 		  			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 		  			  		 			     			  	 
    :type verbose: bool  		  	   		 		  			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 		  			  		 			     			  	 
    :type impact: float  		  	   		 		  			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 		  			  		 			     			  	 
    :type commission: float  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    # constructor  		  	   		 		  			  		 			     			  	 
    def __init__(self, verbose=False, impact=0, commission=0.0):
        """  		  	   		 		  			  		 			     			  	 
        Constructor method  		  	   		 		  			  		 			     			  	 
        """  		  	   		 		  			  		 			     			  	 
        self.verbose = verbose  		  	   		 		  			  		 			     			  	 
        self.impact = impact  		  	   		 		  			  		 			     			  	 
        self.commission = commission
        self.learner = BagLearner(learner=RTLearner, kwargs={"leaf_size":5}, bags= 20)
        self.lookAhead = 10
        self.buyHurdle = 0.02 + impact
        self.sellHurdle = 0.02 + impact
  		  	   		 		  			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		 		  			  		 			     			  	 
    def add_evidence(  		  	   		 		  			  		 			     			  	 
        self,  		  	   		 		  			  		 			     			  	 
        symbol="IBM",  		  	   		 		  			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 		  			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		 		  			  		 			     			  	 
        sv=10000,  		  	   		 		  			  		 			     			  	 
    ):  		  	   		 		  			  		 			     			  	 
        """  		  	   		 		  			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 		  			  		 			     			  	 
        :type symbol: str  		  	   		 		  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 		  			  		 			     			  	 
        :type sd: datetime  		  	   		 		  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 		  			  		 			     			  	 
        :type ed: datetime  		  	   		 		  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 		  			  		 			     			  	 
        :type sv: int  		  	   		 		  			  		 			     			  	 
        """

        x = self._features(symbol, sd, ed)
        y = self._labels(symbol, sd, ed)

        df = pd.concat([x, y.rename('y')], axis=1).dropna()
        self.learner.add_evidence(df.iloc[:,:-1].values, df['y'].values)

    # this method should use the existing policy and test it against new data  		  	   		 		  			  		 			     			  	 
    def testPolicy(  		  	   		 		  			  		 			     			  	 
        self,  		  	   		 		  			  		 			     			  	 
        symbol="IBM",  		  	   		 		  			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		 		  			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		 		  			  		 			     			  	 
        sv=10000,  		  	   		 		  			  		 			     			  	 
    ):  		  	   		 		  			  		 			     			  	 
        """  		  	   		 		  			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 		  			  		 			     			  	 
        :type symbol: str  		  	   		 		  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 		  			  		 			     			  	 
        :type sd: datetime  		  	   		 		  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 		  			  		 			     			  	 
        :type ed: datetime  		  	   		 		  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 		  			  		 			     			  	 
        :type sv: int  		  	   		 		  			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 		  			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 		  			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 		  			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 		  			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 		  			  		 			     			  	 
        """
        x = self._features(symbol, sd, ed)
        x_valid = x.dropna()
        y_raw_valid = self.learner.query(x_valid.values)
        y = pd.Series(0, index=x.index)
        y.loc[x_valid.index] = y_raw_valid

        target = y*1000
        df_trades = pd.DataFrame(target.diff().fillna(target.iloc[0]).values, index = x.index, columns=[symbol])
        return df_trades


    def _labels(self, symbol, sd, ed):
        prices = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=True)
        prices = prices.drop(columns=['SPY'])
        prices.ffill(inplace=True)
        prices.bfill(inplace=True)
        future_ret = prices[symbol].shift(-self.lookAhead)/prices[symbol]-1
        y = pd.Series(0, index=prices.index)
        y[future_ret > self.buyHurdle] = 1
        y[future_ret < -self.sellHurdle] = -1
        return y

    def _features(self, symbol, sd, ed):
        ind = Indicators()
        # macd = ind.macd(symbol=symbol, sd=sd, ed=ed)
        # momentum = ind.momentum(symbol=symbol, sd=sd, ed=ed, lookback=14)
        rsi = ind.rsi(symbol=symbol, sd=sd, ed=ed, lookback=14)
        ema = ind.ema(symbol=symbol, sd=sd, ed=ed, lookback=14)
        bollingerBands = ind.bollingerBands(symbol=symbol, sd=sd, ed=ed, lookback=14)
        return pd.concat([rsi, ema, bollingerBands], axis=1)

    def author(self):
        return "xliu397"
    def study_group(self):
        return "xliu397"
def author():
    return "xliu397"
def study_group():
    return "xliu397"
if __name__ == "__main__":  		  	   		 		  			  		 			     			  	 
    print("One does not simply think up a strategy")
    strategyLeaner  = StrategyLearner(verbose=False, impact = 0.0, commission = 0.0)
    strategyLeaner.add_evidence(symbol = "AAPL", sd=dt.datetime(2008,1,1),
                                ed=dt.datetime(2009,12,31), sv =100000)
