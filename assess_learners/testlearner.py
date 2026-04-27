""""""

"""  		  	   		 		  			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
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
"""  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import math  		  	   		 		  			  		 			     			  	 
import sys  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import numpy as np  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il
import matplotlib.pyplot as plt
import time


def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def runLearner(learner, train_x, train_y, test_x, test_y):
    # create a learner and train it
    learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    in_rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    in_c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    out_rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    out_c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0, 1]}")
    return np.array([in_rmse, in_c[0,1], out_rmse, out_c[0,1]])

def runExp3Learner(learner, train_x, train_y, test_x, test_y):
    # create a learner and train it
    start = time.time()
    learner.add_evidence(train_x, train_y)  # train it
    end = time.time()
    timeElapse = end - start

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    in_maxError = abs(((train_y - pred_y))).max()

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    out_maxError = abs((test_y - pred_y)).max()
    return np.array([in_maxError, out_maxError, timeElapse])

if __name__ == "__main__":
    if len(sys.argv) != 2:  		  	   		 		  			  		 			     			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		 		  			  		 			     			  	 
        sys.exit(1)  		  	   		 		  			  		 			     			  	 
    inf = open(sys.argv[1])
    i = 0
    data=[]
    floatCols = []
    headerFirstRow= True
    for s in inf.readlines():
        row = s.strip().split(",")
        if i == 0:
            headerFirstRow = not np.all(list(map(is_num, row)))
            if not headerFirstRow:
                for index, item in enumerate(list(map(is_num, row))):
                    if item:
                        floatCols.append(index)
        if i == 1 and headerFirstRow:
            for index, item in enumerate(list(map(is_num, row))):
                if item:
                    floatCols.append(index)
        if len(floatCols) > 0:
            data.append([float(row[j]) for j in floatCols])
        i+=1
    data = np.array(data)
    # compute how much of the data is training and testing  		  	   		 		  			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		 		  			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		 		  			  		 			     			  	 

    randIndices = np.arange(data.shape[0])
    np.random.shuffle(randIndices)
    # separate out training and testing data  		  	   		 		  			  		 			     			  	 
    train_x = data[randIndices[:train_rows], 0:-1]
    train_y = data[randIndices[:train_rows], -1]
    test_x = data[randIndices[train_rows:], 0:-1]
    test_y = data[randIndices[train_rows:], -1]
  		  	   		 		  			  		 			     			  	 
    print(f"{test_x.shape}")  		  	   		 		  			  		 			     			  	 
    print(f"{test_y.shape}")  		  	   		 		  			  		 			     			  	 

    # print("Linear Regression---------------------- ")
    # runLearner(lrl.LinRegLearner(verbose=True), train_x, train_y, test_x, test_y)
    print("Decision Tree-------------------------")
    exp1Result = []
    for leafSize in range(1, 50, 1):
        result = runLearner(dt.DTLearner(leaf_size = leafSize, verbose = False), train_x, train_y, test_x, test_y)
        exp1Result.append([leafSize]+list(result))
    exp1Result = np.array(exp1Result)
    plt.figure(figsize=(10, 6))
    plt.plot(exp1Result[:,0], exp1Result[:,3], label='out of sample RMSE Error')
    plt.plot(exp1Result[:, 0], exp1Result[:, 1], label='in sample RMSE Error')
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.title('DT RMSE vs Leaf Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/experiment1.png')
    plt.close()
    # print("Random Tree-------------------------")
    # runLearner(rt.RTLearner(leaf_size = 1, verbose = False), train_x, train_y, test_x, test_y)
    print("Bag Learner-------------------------")
    exp2Result = []
    for leafSize in range(1, 50, 1):
        result = runLearner(bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":leafSize}, bags = 10, boost = False, verbose = False), train_x, train_y, test_x, test_y)
        exp2Result.append([leafSize]+list(result))
    exp2Result = np.array(exp2Result)
    plt.figure(figsize=(10, 6))
    plt.plot(exp2Result[:,0], exp2Result[:,3], label='out of sample RMSE Error')
    plt.plot(exp2Result[:, 0], exp2Result[:, 1], label='in sample RMSE Error')
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.title('Bagging DT RMSE vs Leaf Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/experiment2.png')
    plt.close()
    # print("Insane Learner-------------------------")
    # runLearner(il.InsaneLearner(),train_x, train_y, test_x, test_y)
    exp3Result = []
    for leafSize in range(1, 50, 1):
        resultRt = runExp3Learner(rt.RTLearner(leaf_size = leafSize, verbose = False), train_x, train_y, test_x, test_y)
        resultDt = runExp3Learner(dt.DTLearner(leaf_size = leafSize, verbose = False), train_x, train_y, test_x, test_y)
        exp3Result.append([leafSize]+list(resultRt)+list(resultDt))
    exp3Result = np.array(exp3Result)
    plt.figure(figsize=(10, 6))
    plt.plot(exp3Result[:,0], exp3Result[:,6], label='DT build time')
    plt.plot(exp3Result[:, 0], exp3Result[:, 3], label='RT build time')
    plt.xlabel('leaf size')
    plt.ylabel('Build Time')
    plt.title('Build Time vs Leaf Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/experiment3.1.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(exp3Result[:, 0], exp3Result[:, 5], label='DT out of sample Maximum Error')
    plt.plot(exp3Result[:, 0], exp3Result[:, 2], label='RT out of sample Maximum Error')
    plt.xlabel('leaf size')
    plt.ylabel('Max Error')
    plt.title('Max Error vs Leaf Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/experiment3.2.png')
    plt.close()