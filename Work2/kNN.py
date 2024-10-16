from enum import Enum

# Skaiste
def euclidean_distance(x, y):
    return x

# Tatevik
def minkowski_distance(x, y):
    pass

# Wiktoria
def manhattan_distance(x, y):
    pass

class DistanceType(Enum):
    EUCLIDEAN = 0
    MINKOWSKI = 1
    MANHATTAN = 2

# Skaiste
def majority_class_vs():
    pass

# Tatevik
def innverse_distance_weighted_vs():
    pass

# Wiktoria
def sheppards_work_vs():
    pass

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

    def predict(self, test_input, fold):
        # predict output for each input
        predictions = []
        for _, x in test_input.iterrows():
            # calculate distance to each point in the training input set
            # use self.distance
            # distances = [self.distance(x, x_train) for _, x_train in self.train_input[fold].iterrows()]

            # sum each row
            # distance_sums = [sum(d) for d in distances]

            # select the closest neighbour(s)



            # get the outputs of the nearest neighbours

            # apply voting schemes

            # apply weigting strategies:
            # - equal - Skaiste
            # - filter - Tatevik
            # - wrapper - Wiktoria

            # get the most common output across the outputs
            continue

        return predictions