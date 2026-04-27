import datetime as dt
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data


def author():
    return "xliu397"


def study_group():
    return "xliu397"


def compute_portvals(
        orders_df,
        start_val=100000,
        commission=0,
        impact=0
):
    """
    Computes the portfolio values.

    :param orders_df: orders dataframe
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    symbol = orders_df.columns[0]
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()

    prices = get_data([symbol], pd.date_range(start_date, end_date), addSPY=False)
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    prices['Cash'] = 1.0

    trades = pd.DataFrame(0, index = prices.index, columns=prices.columns)
    trades[symbol] = orders_df[symbol]
    trades.fillna(0, inplace=True)
    trades['Cash'] = -(trades[symbol]*prices[symbol])
    trades.fillna(0, inplace=True)

    trade_days = trades[symbol] != 0
    trades.loc[trade_days, 'Cash'] -= commission
    trades.loc[trade_days, 'Cash'] -= abs(trades.loc[trade_days, symbol]) * prices.loc[trade_days, symbol]*impact
    trades.loc[start_date, 'Cash'] +=start_val

    holdings = trades.cumsum()
    portval = (prices*holdings).sum(axis=1)
    return portval