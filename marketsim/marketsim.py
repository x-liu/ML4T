""""""  		  	   		 		  			  		 			     			  	 
"""MC2-P1: Market simulator.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
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
  		  	   		 		  			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 		  			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 		  			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 		  			  		 			     			  	 
"""  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import datetime as dt  		  	   		 		  			  		 			     			  	 
import os  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import numpy as np  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import pandas as pd  		  	   		 		  			  		 			     			  	 
from util import get_data, plot_data  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
def author():
    return 'xliu397' # replace tb34 with your Georgia Tech username.

def compute_portvals(  		  	   		 		  			  		 			     			  	 
    orders_file="./orders/orders-01.csv",
    start_val=1000000,  		  	   		 		  			  		 			     			  	 
    commission=9.95,  		  	   		 		  			  		 			     			  	 
    impact=0.005,  		  	   		 		  			  		 			     			  	 
):  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    Computes the portfolio values.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    :param orders_file: Path of the order file or the file object  		  	   		 		  			  		 			     			  	 
    :type orders_file: str or file object  		  	   		 		  			  		 			     			  	 
    :param start_val: The starting value of the portfolio  		  	   		 		  			  		 			     			  	 
    :type start_val: int  		  	   		 		  			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 		  			  		 			     			  	 
    :type commission: float  		  	   		 		  			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 		  			  		 			     			  	 
    :type impact: float  		  	   		 		  			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 		  			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    # this is the function the autograder will call to test your code  		  	   		 		  			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 		  			  		 			     			  	 
    # code should work correctly with either input  		  	   		 		  			  		 			     			  	 
    # TODO: Your code here  		  	   		 		  			  		 			     			  	 
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    # In the template, instead of computing the value of the portfolio, we just  		  	   		 		  			  		 			     			  	 
    # read in the value of IBM over 6 months
    orders_df.sort_index(inplace=True)
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    symbols = set(orders_df['Symbol'])
    portvals = get_data(symbols, pd.date_range(start_date, end_date))
    prices = portvals[list(symbols)]  # remove SPY
    prices['Cash'] = 1
    trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    trades['Cash'] = trades['Cash'].astype(float)
    for curOrder in orders_df.itertuples(index=True):
        if curOrder.Order == 'BUY':
            trades.loc[curOrder.Index, curOrder.Symbol] += curOrder.Shares
            trades.loc[curOrder.Index, 'Cash'] -= curOrder.Shares * prices.loc[curOrder.Index, curOrder.Symbol]*(1+impact)+commission
        else:
            trades.loc[curOrder.Index, curOrder.Symbol] -= curOrder.Shares
            trades.loc[curOrder.Index, 'Cash'] += curOrder.Shares * prices.loc[curOrder.Index, curOrder.Symbol]*(1-impact)-commission
    trades.loc[trades.index[0],'Cash'] += start_val
    holdings=trades.cumsum()
    values = prices*holdings
    rv = values.sum(axis=1)
    return rv
  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
def test_code():  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    Helper function to test code  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		 		  			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		 		  			  		 			     			  	 
    # Define input parameters  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    of = "./orders/orders-02.csv"
    sv = 1000000  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    # Process orders  		  	   		 		  			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):  		  	   		 		  			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 		  			  		 			     			  	 
    else:  		  	   		 		  			  		 			     			  	 
        "warning, code did not return a DataFrame"  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		 		  			  		 			     			  	 
    start_date = portvals.index.min()
    end_date = portvals.index.max()

    dates = pd.date_range(start_date, end_date)
    symbols = ['SPY']
    df = get_data(symbols, dates)
    prices = df.copy()

    normed = prices / prices.iloc[0, :]
    portvals= portvals/portvals.iloc[0]
    SPY_portvals = normed

    dailyReturns = (portvals/portvals.shift(1)) - 1
    dailyReturns = dailyReturns[1:]
    SPY_daily_rets = (SPY_portvals / SPY_portvals.shift(1)) - 1
    SPY_daily_rets = SPY_daily_rets[1:]

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        portvals.iloc[-1]-1,
        np.mean(dailyReturns),
        np.std(dailyReturns, ddof=1),
        np.sqrt(252)*np.mean(dailyReturns)/np.std(dailyReturns, ddof=1),
    ]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		 		  			  		 			     			  	 
        SPY_portvals.iloc[-1]-1,
        np.mean(SPY_daily_rets),
        np.std(SPY_daily_rets, ddof=1),
        np.sqrt(252)*np.mean(SPY_daily_rets)/np.std(SPY_daily_rets, ddof=1),
    ]  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    # Compare portfolio against $SPX  		  	   		 		  			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 		  			  		 			     			  	 
    print()  		  	   		 		  			  		 			     			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 		  			  		 			     			  	 
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 		  			  		 			     			  	 
    print()  		  	   		 		  			  		 			     			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 		  			  		 			     			  	 
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 		  			  		 			     			  	 
    print()  		  	   		 		  			  		 			     			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 		  			  		 			     			  	 
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 		  			  		 			     			  	 
    print()  		  	   		 		  			  		 			     			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 		  			  		 			     			  	 
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 		  			  		 			     			  	 
    print()  		  	   		 		  			  		 			     			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
if __name__ == "__main__":  		  	   		 		  			  		 			     			  	 
    test_code()  		  	   		 		  			  		 			     			  	 
