import numpy as np
import pandas as pd
from sklearn import svm
from enum import Enum
from sklearn.preprocessing import MultiLabelBinarizer
from concurrent.futures import ProcessPoolExecutor
from kNN import ReductionMethod

# Classifier Kernel Types https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py
class KernelType(Enum):
    RBF = 'rbf'                 # radial basis function
    POLYNOMIAL = 'polynomial'

class SVM:
    def __init__(self, kernel, reduction=ReductionMethod.NO_REDUCTION):
        # setup classifiers
        if kernel == KernelType.RBF:
            self.classifier = svm.SVC(kernel='rbf', gamma='scale')
        elif kernel == KernelType.POLYNOMIAL:
            self.classifier = svm.SVC(kernel='poly', degree=3, gamma='scale', coef0=1)
        else:
            raise Exception("Unknown classifier specified")
        
        self.reduction_method = reduction


    def DROP3_reduction(self, train_input, train_output):
        retain_indices = train_input.index.tolist()
        # predict output for each input
        for idx in train_input.index:
            # remove current instance from training set
            temp_X = train_input.drop(index=idx)
            temp_y = train_output.drop(index=idx)

            # calculate distance to each point in the training input set
            weights = np.ones((train_input.shape[1],)) # using equal weights to reduce noise when selecting weights after reduction
            distances = self.distance(train_input.loc[idx], temp_X, weights)
            distances = distances.sort_values()

            # get output using selected knn algorithms
            neighbours = distances.iloc[:min(self.k, distances.size)]
            outputs = temp_y.loc[list(neighbours.keys())]
            output = self.voting_scheme(outputs, distances)

            # checking if predicted value matches and if it does, retain the instance
            if output != train_output.loc[idx]:
                retain_indices.append(idx)

        # reduce the train input and output dataframes to selected instances
        retain_indices = list(set(retain_indices))  # Remove duplicates
        reduced_X = train_input.loc[retain_indices]
        reduced_y = train_output.loc[retain_indices]

        return reduced_X, reduced_y
    
    

    def RENN_reduction(self, max_iterations=100):
        retain_indices = self.train_input.index.tolist()
        iteration = 0
        stable = False

        while not stable and iteration < max_iterations:
            stable = True
            indices_to_remove = []

            for idx in retain_indices:
                temp_X = self.train_input.loc[retain_indices].drop(index=idx)
                temp_y = self.train_output.loc[retain_indices].drop(index=idx)

                # calculate distance to each point in the training input set
                weights = np.ones((self.train_input.shape[1],))  # using equal weights to reduce noise when selecting weights after reduction
                distances = self.distance(self.train_input.loc[idx], temp_X, weights)
                distances = distances.sort_values()

                # get output using the k nearest neighbors
                neighbours = distances.iloc[:self.k]
                valid_keys = [i for i in neighbours.keys() if i in temp_y.index]  # Ensure valid keys
                outputs = temp_y.loc[valid_keys]  # Use .loc with valid keys
                output = self.voting_scheme(outputs, distances)

                # check if the instance is misclassified
                if output != self.train_output.loc[idx]:
                    indices_to_remove.append(idx)
                    stable = False

            # remove misclassified instances for this iteration
            retain_indices = [i for i in retain_indices if i not in indices_to_remove]
            iteration += 1

            # reduce dataset after all iterations 
            reduced_X = self.train_input.loc[retain_indices].reset_index(drop=True)
            reduced_y = self.train_output.loc[retain_indices].reset_index(drop=True)
            print(f"Final reduced dataset size: {len(reduced_X)} instances.")

            return reduced_X, reduced_y
        
    # Tatevik
    def fcnn1(self):
        unique_labels = self.train_output.unique()
        threshold = 0.01 * len(self.train_input)

        centroids = set()
        S = set()
        nearest = {p: None for p in self.train_input.index}
        for class_ in unique_labels:
            all_class = self.train_input[self.train_output == class_]
            all_mean = all_class.mean(axis = 0)
            all_mean = pd.DataFrame([all_mean.values] * all_class.shape[0], columns=all_class.columns)
            distances = self.distance(all_mean,all_class, self.feature_weights)
            #centroids[class_] = self.train_input.iloc[distances.idxmin()].index
            centroids.add(self.train_input.index[distances.idxmin()])
        difference_s = set(centroids)
        i = 0
        while difference_s:
            print(f"Iteration {i}; difference_s ({len(S)}) > threshold ({threshold})" + " " * 10, end="\r")
            i+=1
            S.update(difference_s)

            rep = {p: None for p in S}

            for q in self.train_input.index.difference(S):
                for p in difference_s:
                    if nearest[q]:
                        if (self.distance(
                            self.train_input.iloc[nearest[q]].to_frame().T,
                            self.train_input.iloc[q].to_frame().T,
                            self.feature_weights).iloc[0]
                        > self.distance(
                            self.train_input.iloc[p].to_frame().T,
                            self.train_input.iloc[q].to_frame().T,
                            self.feature_weights).iloc[0]
                        ):
                            nearest[q] = p
                    else:
                        nearest[q] = p

                if rep[nearest[q]]:
                    if ((self.train_output.iloc[q] != self.train_output.iloc[nearest[q]]) and 
                        (self.distance(
                            self.train_input.iloc[nearest[q]].to_frame().T, 
                            self.train_input.iloc[q].to_frame().T,self.feature_weights
                        ).iloc[0] < self.distance(
                            self.train_input.iloc[nearest[q]].to_frame().T,
                            self.train_input.iloc[rep[nearest[q]]].to_frame().T ,self.feature_weights
                        ).iloc[0])):
                        rep[nearest[q]] = q
                elif (self.train_output.iloc[q] != self.train_output.iloc[nearest[q]]):
                    rep[nearest[q]] = q

            difference_s = set()
            for p in S:
                if rep[p]:
                    difference_s.add(rep[p])

            #difference_s = {rep[p] for p in S if rep[p] is not None}
        reduced_X = self.train_input.loc[list(S)].reset_index(drop=True)
        reduced_Y = self.train_output.loc[list(S)].reset_index(drop=True)
        print(f"Final reduced dataset size: {len(reduced_X)} instances.")
        return reduced_X,reduced_Y
    
    def fcnn1_nearest_distance(self, q, difference_s, nearest, rep):
        for p in difference_s:
            if nearest[q]:
                if (self.distance(
                    self.train_input.iloc[nearest[q]].to_frame().T,
                    self.train_input.iloc[q].to_frame().T,
                    self.feature_weights).iloc[0]
                > self.distance(
                    self.train_input.iloc[p].to_frame().T,
                    self.train_input.iloc[q].to_frame().T,
                    self.feature_weights).iloc[0]
                ):
                    nearest[q] = p
            else:
                nearest[q] = p

        if rep[nearest[q]]:
            if ((self.train_output.iloc[q] != self.train_output.iloc[nearest[q]]) and 
                (self.distance(
                    self.train_input.iloc[nearest[q]].to_frame().T, 
                    self.train_input.iloc[q].to_frame().T,self.feature_weights
                ).iloc[0] < self.distance(
                    self.train_input.iloc[nearest[q]].to_frame().T,
                    self.train_input.iloc[rep[nearest[q]]].to_frame().T ,self.feature_weights
                ).iloc[0])):
                rep[nearest[q]] = q
        elif (self.train_output.iloc[q] != self.train_output.iloc[nearest[q]]):
            rep[nearest[q]] = q

        return nearest[q], rep[nearest[q]]

    def fcnn1_threaded(self):
        unique_labels = self.train_output.unique()
        max_iterations = 100

        centroids = set()
        S = set()
        nearest = {p: None for p in self.train_input.index}
        for class_ in unique_labels:
            all_class = self.train_input[self.train_output == class_]
            all_mean = all_class.mean(axis = 0)
            all_mean = pd.DataFrame([all_mean.values] * all_class.shape[0], columns=all_class.columns)
            distances = self.distance(all_mean,all_class, self.feature_weights)
            #centroids[class_] = self.train_input.iloc[distances.idxmin()].index
            centroids.add(self.train_input.index[distances.idxmin()])
        difference_s = set(centroids)
        i = 0
        while difference_s:
            print(f"Iteration {i}; difference_s ({len(S)})" + " " * 10, end="\r")
            if i == max_iterations:
                break
            i+=1
            S.update(difference_s)

            rep = {p: None for p in S}

            # Parallelize finding nearest for each point not in S
            with ProcessPoolExecutor() as executor: #q, difference_s, nearest, rep
                futures = {q: executor.submit(self.fcnn1_nearest_distance, q, difference_s, nearest, rep) for q in self.train_input.index.difference(S)}
                for q, future in futures.items():
                    rtrn = future.result()
                    nearest[q] = rtrn[0]
                    rep[nearest[q]] = rtrn[1]

            difference_s = set()
            for p in S:
                if rep[p]:
                    difference_s.add(rep[p])

            #difference_s = {rep[p] for p in S if rep[p] is not None}
        reduced_X = self.train_input.loc[list(S)].reset_index(drop=True)
        reduced_Y = self.train_output.loc[list(S)].reset_index(drop=True)
        print(f"Final reduced dataset size: {len(reduced_X)} instances.")
        return reduced_X,reduced_Y

    def fit(self, training_input, training_output):
        self.train_input = training_input
        self.train_output = training_output

        if self.reduction_method == ReductionMethod.CONDENSATION:
            self.train_input, self.train_output = self.fcnn1_threaded()

        elif self.reduction_method == ReductionMethod.EDITION:
            self.train_input, self.train_output = self.RENN_reduction()

        elif self.reduction_method == ReductionMethod.HYBRID:
            self.train_input, self.train_output = self.DROP3_reduction(self.train_input, self.train_output)

        # train the classifiers on the data
        self.classifier.fit(self.train_input, self.train_output)

    def predict(self, testing_input):
        # X = np.array(testing_input.values.tolist())
        return self.classifier.predict(testing_input)