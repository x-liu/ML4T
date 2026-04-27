import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs={}, bags=20, boost = False, verbose = False):
        """
        Constructor method
        """
        self.learner = learner
        self.bags = bags
        self.verbose = verbose
        self.learners = [learner(**kwargs) for i in range(self.bags)]

    def author(self):
        return "xliu397"
    def study_group(self):
        return "xliu397"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # build and save the model
        n=data_x.shape[0]
        for learner in self.learners:
            indices = np.random.choice(n, size=n, replace=True)
            # separate out training and testing data
            train_x = data_x[indices]
            train_y = data_y[indices]
            learner.add_evidence(train_x, train_y)


    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        queries = np.zeros((self.bags,len(points)))
        for i in range(len(self.learners)):
            queries[i] = self.learners[i].query(points)
        return np.sign(queries.sum(axis=0))

def author():
    return "xliu397"
def study_group():
    return "xliu397"

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")


