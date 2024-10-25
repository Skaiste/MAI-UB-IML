import numpy as np
import pandas as pd
from enum import Enum
from sklearn.feature_selection import mutual_info_classif
from sklearn_relief import ReliefF


# Skaiste
def euclidean_distance(x, y, weights):
    return np.sqrt(np.sum(weights * (x - y) ** 2, axis=1))

# Tatevik
def minkowski_distance(x, y, weights, r=1):
    return ((weights * abs(x-y)**r).sum(axis = 1))**(1/r)

# Wiktoria
def manhattan_distance(x, y, weights):
    return np.sum(weights * np.abs(x - y), axis=1)

class DistanceType(Enum):
    EUCLIDEAN = 'euclidean'
    MINKOWSKI = 'minkowski'
    MANHATTAN = 'manhattan'

# Skaiste
def majority_class_vs(outputs, _):
    return outputs.mode().values[0]

# Tatevik
def innverse_distance_weighted_vs(outputs, distances):
    votes = {}
    for i in range(len(outputs)):
        vote = 1/distances[i]
        label = outputs.iloc[i]
        if label not in votes:
            votes[label] = 0
        votes[label] = vote+votes[label]
    return max(votes, key=votes.get)

# Wiktoria
def sheppards_work_vs(outputs, distances, p=2):
    # add a small epsilon to the distances (to prevent division by 0)
    epsilon = 1e-9
    distances = np.array(distances) + epsilon

    # calculate inverse distance weights
    weights = 1 / (distances ** p)

    # normalize the weights so they sum to 1
    weights = weights / np.sum(weights)
    return weights


class VotingSchemas(Enum):
    MAJORITY_CLASS = 'majority_class'
    INVERSE_DISTANCE = 'inverse_distance'
    SHEPHERDS_WORK = 'shepherds_work'


class WeigthingStrategies(Enum):
    EQUAL = 'equal'
    FILTER = 'filter'
    WRAPPER = 'wrapper'


class KNN:
    def __init__(self, k=1, dm=DistanceType.EUCLIDEAN, vs=VotingSchemas.MAJORITY_CLASS, ws=WeigthingStrategies.EQUAL, r=1):
        self.k = k
        self.r = r
        if dm == DistanceType.EUCLIDEAN:
            self.distance = euclidean_distance
        elif dm == DistanceType.MINKOWSKI:
            self.distance = minkowski_distance
        elif dm == DistanceType.MANHATTAN:
            self.distance = manhattan_distance

        if vs == VotingSchemas.MAJORITY_CLASS:
            self.voting_scheme = majority_class_vs
        elif vs == VotingSchemas.INVERSE_DISTANCE:
            self.voting_scheme = innverse_distance_weighted_vs
        elif vs == VotingSchemas.SHEPHERDS_WORK:
            self.voting_scheme = sheppards_work_vs

        self.weighting_strategy = ws
    
    def fit(self, train_input, train_output):
        self.train_input = train_input
        self.train_output = train_output

        if self.weighting_strategy == WeigthingStrategies.FILTER:
            weights = mutual_info_classif(train_input, train_output)
            self.feature_weights = weights / np.sum(weights)
        elif self.weighting_strategy == WeigthingStrategies.WRAPPER:
            relieff = ReliefF(n_features=train_input.shape[1])
            relieff.fit(train_input.values, train_output.values)
            self.feature_weights = relieff.w_ 
        else:
            self.feature_weights = np.ones((train_input.shape[1],))


    def predict(self, test_input):
        # predict output for each input
        predictions = pd.Series([], dtype=self.train_output.dtype, name=self.train_output.name)
        for idx, x in test_input.iterrows():
            # calculate distance to each point in the training input set
            if self.distance == minkowski_distance:
                distances = self.distance(x, self.train_input, self.feature_weights, self.r)
            else:
                distances = self.distance(x, self.train_input, self.feature_weights)
            distances = distances.sort_values()

            # select the closest neighbour(s)
            neighbours = distances.iloc[:self.k]

            # get the outputs of the nearest neighbours
            outputs = self.train_output.iloc[list(neighbours.keys())]

            # apply voting schemes
            output = self.voting_scheme(outputs, distances)

            predictions.loc[idx] = output

        return predictions