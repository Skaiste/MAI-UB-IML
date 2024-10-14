from enum import Enum


def euclidean_distance(x, y):
    pass

def minkowski_distance(x, y):
    pass

def manhattan_distance(x, y):
    pass

class DistanceType(Enum):
    EUCLIDEAN = 0
    MINKOWSKI = 1
    MANHATTAN = 2

class kNN:
    def __init__(self, k=1, distance_metric=DistanceType.EUCLIDEAN):
        self.k = k
        if distance_metric == DistanceType.EUCLIDEAN:
            self.distance = euclidean_distance
        elif distance_metric == DistanceType.MINKOWSKI:
            self.distance = minkowski_distance
        elif distance_metric == DistanceType.MANHATTAN:
            self.distance = manhattan_distance
    
    def fit(self, train_input, train_output):
        self.train_input = train_input
        self.train_output = train_output

    def predict(self, test_input):
        # predict output for each input
        predictions = []
        for i in test_input:
            # calculate distance to each point in the training input set
            # use self.distance

            # select the closest neighbour(s)

            # get the outputs of the nearest neighbours

            # get the most common output across the outputs
            continue

        return predictions