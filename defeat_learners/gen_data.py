""""""  		  	   		 		  			  		 			     			  	 
"""  		  	   		 		  			  		 			     			  	 
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		 		  			  		 			     			  	 
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
  		  	   		 		  			  		 			     			  	 
import math  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import numpy as np  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
# this function should return a dataset (X and Y) that will work  		  	   		 		  			  		 			     			  	 
# better for linear regression than decision trees  		  	   		 		  			  		 			     			  	 
def best_4_lin_reg(seed=1489683273):  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 		  			  		 			     			  	 
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 		  			  		 			     			  	 
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    :param seed: The random seed for your data generation.  		  	   		 		  			  		 			     			  	 
    :type seed: int  		  	   		 		  			  		 			     			  	 
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 		  			  		 			     			  	 
    :rtype: numpy.ndarray  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    np.random.seed(seed)  		  	   		 		  			  		 			     			  	 
    x = np.random.randn(800,4)
    # y = np.random.random(size=(100,)) * 200 - 100
    # Here's is an example of creating a Y from randomly generated  		  	   		 		  			  		 			     			  	 
    # X with multiple columns  		  	   		 		  			  		 			     			  	 
    y = x[:,0]*5+x[:,1]+4+x[:,2]/10+x[:,3]
    return x, y  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
def best_4_dt(seed=1489683273):  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 		  			  		 			     			  	 
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 		  			  		 			     			  	 
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    :param seed: The random seed for your data generation.  		  	   		 		  			  		 			     			  	 
    :type seed: int  		  	   		 		  			  		 			     			  	 
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 		  			  		 			     			  	 
    :rtype: numpy.ndarray  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    np.random.seed(seed)
    x = np.random.randn(800,4)
    y = np.zeros(x.shape[0])
    medians = np.median(x,axis=0)
    stdevs = np.std(x,axis=0)
    mask1 = (x[:, 0] <= medians[0]-stdevs[0]*2)
    mask2 =  ((x[:, 1] > medians[1]-stdevs[1]*2) & (x[:, 1] <= medians[1]))
    mask3 = ((x[:, 2] > medians[2]) & (x[:, 2] <= medians[2] +stdevs[2]*1.5))
    mask4 = (x[:, 3] > medians[3] + stdevs[3]*1.5)
    y[mask1] = x[mask1,2] / -1
    y[mask2]= np.cos(x[mask2,3])+1
    y[mask3] = np.sin(x[mask3, 1])-1
    y[mask4] = x[mask4, 0] **2
    return x, y  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
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

if __name__ == "__main__":  		  	   		 		  			  		 			     			  	 
    print("they call me Xing.")
