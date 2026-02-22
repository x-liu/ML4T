import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.bags = 20
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, bags=20) for i in range(self.bags)]
    def author(self):
        return "xliu397"  # replace tb34 with your Georgia Tech username
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        queries = np.zeros((self.bags,len(points)))
        for i in range(len(self.learners)):
            queries[i] = self.learners[i].query(points)
        return np.mean(queries, axis=0)
