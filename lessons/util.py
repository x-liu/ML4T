import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#----------function to get path of the symbol-------------
def symbol_to_path(symbol, base_dir="../data"):
	"""Return CSV file path given ticker symbol."""
	return os.path.join(base_dir, "{}.csv".format(str(symbol)))

#--------------------Reads csv----------------------------
def get_data(symbollist, dates):
	df_final=pd.DataFrame(index=dates)
	if "SPY" not in symbollist:
		symbollist.insert(0,"SPY")
	for symbol in symbollist:
		file_path=symbol_to_path(symbol)
		df_temp=pd.read_csv(file_path, parse_dates=True, index_col="Date",usecols=["Date", "Adj Close"], na_values=["nan"])
		df_temp=df_temp.rename(columns={'Adj Close':symbol})
		df_final=df_final.join(df_temp)
		if symbol == "SPY":
			df_final=df_final.dropna(subset=['SPY'])
	return df_final

def plot_data(df_data, title = None, xlabel = None, ylabel = None):
    """Plot stock data with appropriate axis labels."""
    ax = df_data.plot(title=title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()