""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
  		  	   		 	 	 		  		  		    	 		 		   		 		  
BLACK_PROB = 18/38

def author():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return "xliu397"  # replace tb34 with your Georgia Tech username.  		  	   		 	 	 		  		  		    	 		 		   		 		  

def study_group():
    return "xliu397"

def gtid():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return 903011732  # replace with your GT ID number  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    result = False  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        result = True  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return result
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    win_prob = 0.4 # set appropriately to the probability of a win  		  	   		 	 	 		  		  		    	 		 		   		 		  
    np.random.seed(gtid())  # do this only once  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # add your code here to implement the experiments

    # Experiment 1:
    plot1Title = "Figure 1"
    runFig1(plot1Title, 10, 1000)
    plot2Title = "Figure 2"
    runFig2(plot2Title, 1000, 1000)
    plot3Title = "Figure 3"
    runFig3(plot3Title, 1000, 1000)

    # Experiment 2:
    bankroll = 256
    plot4Title = "Figure 4"
    runFig4(plot4Title, 1000, 1000, bankroll)
    plot5Title = "Figure 5"
    runFig5(plot5Title, 1000, 1000, bankroll)



def runFig1(plotTitle, episodes, bets):
    simulations = np.full((episodes, bets + 1), np.nan)
    simulations[:,0] = 0
    for episode in simulations:
        n = 1
        while n < len(episode) and episode[n-1] < 80:
            won = False
            bet_amount = 1
            while not won and n<len(episode):
                won = get_spin_result(BLACK_PROB)
                if won:
                    episode[n] = episode[n - 1] + bet_amount
                else:
                    episode[n] = episode[n - 1] - bet_amount
                    bet_amount *= 2
                n+=1
    df = pd.DataFrame(simulations.T)
    df.ffill(inplace=True)
    df.plot(title=plotTitle, xlabel="bet", ylabel="winnings", fontsize=10, xlim=(0, 300),
                      ylim=(-256, 100))
    plt.savefig("images/Figure1")

def runFig2(plotTitle, episodes, bets):
    simulations = np.full((episodes, bets + 1), np.nan)
    simulations[:,0] = 0
    for episode in simulations:
        n = 1
        while n < len(episode) and episode[n-1] < 80:
            won = False
            bet_amount = 1
            while not won and n<len(episode):
                won = get_spin_result(BLACK_PROB)
                if won:
                    episode[n] = episode[n - 1] + bet_amount
                else:
                    episode[n] = episode[n - 1] - bet_amount
                    bet_amount *= 2
                n+=1
    df = pd.DataFrame(simulations.T)
    df.ffill(inplace=True)
    mean=df.mean(axis=1)
    stdev=df.std(axis=1, ddof=0)
    plt.figure()
    plt.title(plotTitle)
    plt.plot(mean, label='mean')
    plt.plot(mean+stdev, label='upper band')
    plt.plot(mean-stdev, label='lower band')
    plt.xlabel("Spin")
    plt.ylabel("Winnings")
    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.legend(loc='best')
    plt.savefig("images/Figure2")

def runFig3(plotTitle, episodes, bets):
    simulations = np.full((episodes, bets + 1), np.nan)
    simulations[:,0] = 0
    for episode in simulations:
        n = 1
        while n < len(episode) and episode[n-1] < 80:
            won = False
            bet_amount = 1
            while not won and n<len(episode):
                won = get_spin_result(BLACK_PROB)
                if won:
                    episode[n] = episode[n - 1] + bet_amount
                else:
                    episode[n] = episode[n - 1] - bet_amount
                    bet_amount *= 2
                n+=1
    df = pd.DataFrame(simulations.T)
    df.ffill(inplace=True)
    median=df.median(axis=1)
    stdev=df.std(axis=1, ddof=0)
    plt.figure()
    plt.title(plotTitle)
    plt.plot(median, label='median')
    plt.plot(median+stdev, label='upper band')
    plt.plot(median-stdev, label='lower band')
    plt.xlabel("Spin")
    plt.ylabel("Winnings")
    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.legend(loc='best')
    plt.savefig("images/Figure3")
    print(f"In experiment 1, {sum(df.iloc[-1,:]>=80)} episodes reach 80 after 1000 runs")
    print(f"In experiment 1 expected value of winnings after 1000 runs is {df.iloc[-1].mean()}")

def runFig4(plotTitle, episodes, bets, bankroll):
        simulations = np.full((episodes, bets + 1), np.nan)
        simulations[:, 0] = 0
        for episode in simulations:
            n = 1
            while n < len(episode) and episode[n-1] < 80 and episode[n-1] > bankroll*-1:
                won = False
                bet_amount = 1
                while not won and n<len(episode):
                    won = get_spin_result(BLACK_PROB)
                    if won:
                        episode[n] = episode[n - 1] + bet_amount
                    else:
                        episode[n] = episode[n - 1] - bet_amount
                        bet_amount *= 2
                        bet_amount = min(bankroll + episode[n], bet_amount)
                    n+=1
        df = pd.DataFrame(simulations.T)
        df.ffill(inplace=True)
        mean = df.mean(axis=1)
        stdev = df.std(axis=1, ddof=0)
        plt.figure()
        plt.title(plotTitle)
        plt.plot(mean, label='mean')
        plt.plot(mean + stdev, label='upper band')
        plt.plot(mean - stdev, label='lower band')
        plt.xlabel("Spin")
        plt.ylabel("Winnings")
        plt.xlim(0, 300)
        plt.ylim(-256, 100)
        plt.legend(loc='best')
        plt.savefig("images/Figure4")


def runFig5(plotTitle, episodes, bets, bankroll):
    simulations = np.full((episodes, bets + 1), np.nan)
    simulations[:,0] = 0
    for episode in simulations:
        n = 1
        while n < len(episode) and episode[n-1] < 80 and episode[n-1] > bankroll * -1:
            won = False
            bet_amount = 1
            while not won and n<len(episode):
                won = get_spin_result(BLACK_PROB)
                if won:
                    episode[n] = episode[n - 1] + bet_amount
                else:
                    episode[n] = episode[n - 1] - bet_amount
                    bet_amount *= 2
                    bet_amount = min(bankroll + episode[n], bet_amount)
                n += 1
    df = pd.DataFrame(simulations.T)
    df.ffill(inplace=True)
    median=df.median(axis=1)
    stdev=df.std(axis=1, ddof=0)
    plt.figure()
    plt.title(plotTitle)
    plt.plot(median, label='median')
    plt.plot(median+stdev, label='upper band')
    plt.plot(median-stdev, label='lower band')
    plt.xlabel("Spin")
    plt.ylabel("Winnings")
    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.legend(loc='best')
    plt.savefig("images/Figure5")
    print(f"In experiment 2, {sum(df.iloc[-1,:]>=80)/10}% episodes reach 80 after 1000 runs")
    print(f"In Experiment 2, expected value after 1000 runs is {df.iloc[-1].mean()}")

if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	 	 		  		  		    	 		 		   		 		  
