import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "xliu397"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # build and save the model
        self.tree = self.build_tree(
            data_x, data_y
        )



    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        # go through tree
        # return self.tree item after going through points
        retVals = np.zeros(len(points))
        for i in range(len(points)):
            foundLeaf = False
            index = 0
            while not foundLeaf:
                if self.tree[index,0] == -1:
                    retVals[i] = self.tree[index, 1]
                    foundLeaf = True
                elif points[i, int(self.tree[index, 0])]<= self.tree[index, 1]:
                    index+=int(self.tree[index, 2])
                else:
                    index+=int(self.tree[index, 3])
        return retVals



    def build_tree(self, x, y):
        if x.shape[0] <= self.leaf_size:
            return np.array([[-1, np.average(y), np.nan, np.nan]])
        if y.min() == y.max():
            return np.array([[-1, y[0], np.nan, np.nan]])
        else:
            maxCorrIndex = np.random.default_rng().integers(low=0, high=x.shape[1])
            splitVal = np.median(x[:, maxCorrIndex])
            leftXData = x[x[:, maxCorrIndex]<= splitVal]
            rightXData = x[x[:, maxCorrIndex] > splitVal]
            if leftXData.shape[0] == 0 or rightXData.shape[0] == 0:
                return np.array([[-1, np.average(y), np.nan, np.nan]])
            leftTree = self.build_tree(leftXData, y[x[:, maxCorrIndex] <= splitVal])
            rightTree = self.build_tree(rightXData, y[x[:, maxCorrIndex] > splitVal])
            root = np.array([[maxCorrIndex, splitVal, 1, leftTree.shape[0] + 1]])
            return np.vstack([root, leftTree, rightTree])


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")


