import numpy as np
import pandas as pd
from enum import Enum


# Skaiste
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))

# Tatevik
def minkowski_distance(x, y,r = 3):
    return ((abs(x-y)**r).sum(axis = 1))**(1/r)

# Wiktoria
def manhattan_distance(x, y):
    pass

class DistanceType(Enum):
    EUCLIDEAN = 0
    MINKOWSKI = 1
    MANHATTAN = 2

# Skaiste
def majority_class_vs(outputs):
    return outputs.mode().values[0]

# Tatevik
def innverse_distance_weighted_vs(outputs,distances):
    votes = {}
    for i in range(len(outputs)):
        vote = 1/distances[i]
        label = outputs.iloc[i]
        if label not in votes:
            votes[label] = 0
        votes[label] = vote+votes[label]
    return max(votes, key=votes.get)
    # Wiktoria
def sheppards_work_vs():
    pass


class VotingSchemas(Enum):
    MAJORITY_CLASS = 0
    INVERSE_DISTANCE = 1
    SHEPHERDS_WORK = 2


class WeigthingStrategies(Enum):
    EQUAL = 0
    FILTER = 1
    WRAPPER = 2

class kNN:
    def __init__(self, k=1, dm=DistanceType.EUCLIDEAN, vs=VotingSchemas.MAJORITY_CLASS, ws=WeigthingStrategies.EQUAL):
        self.k = k
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

    def predict(self, test_input, fold):
        # predict output for each input
        predictions = pd.Series([], dtype=self.train_output[fold].dtype, name=self.train_output[fold].name)
        for idx, x in test_input.iterrows():
            # calculate distance to each point in the training input set
            # use self.distance
            distances = self.distance(x, self.train_input[fold]).sort_values()

            # select the closest neighbour(s)
            neighbours = distances.iloc[:self.k]

            # get the outputs of the nearest neighbours
            outputs = self.train_output[fold].iloc[list(neighbours.keys())]

            # apply voting schemes
            #Edit by Tatevik
            if self.distance == minkowski_distance:
                output = self.voting_scheme(outputs,distances)
            else:
                output = self.voting_scheme(outputs)

            # apply weigting strategies:
            # - filter - Tatevik
            if self.weighting_strategy == WeigthingStrategies.FILTER:
                pass
            # - wrapper - Wiktoria
            elif self.weighting_strategy == WeigthingStrategies.WRAPPER:
                pass

            predictions.loc[idx] = output

        return predictions