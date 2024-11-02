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
            self.create_classifier = lambda: svm.SVC(kernel='rbf', gamma='scale')
        elif kernel == KernelType.POLYNOMIAL:
            self.create_classifier = lambda: svm.SVC(kernel='poly', degree=3, gamma='scale', coef0=1)
        else:
            raise Exception("Unknown classifier specified")
        
        self.reduction_method = reduction


    def DROP3_reduction(self, train_input, train_output):
        retain_indices = train_input.index.tolist()
        total = len(retain_indices)
        # predict output for each input
        for idx in train_input.index:
            if idx % 100 == 0:
                print(f"Index {idx}; num instances ({total})" + " " * 10, end="\r")
            # remove current instance from training set
            temp_X = train_input.drop(index=idx)
            temp_y = train_output.drop(index=idx)

            # train classifier based on the set with removed instance
            classifier = self.create_classifier()
            classifier.fit(temp_X, temp_y)
            # predict using trained classifier and removed instance as input
            output = classifier.predict(train_input.loc[idx].to_frame().T)

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
                if idx % 100 == 0:
                    print(f"Index {idx}; retain_indexes len({len(retain_indices)})" + " " * 10, end="\r")
                temp_X = self.train_input.loc[retain_indices].drop(index=idx)
                temp_y = self.train_output.loc[retain_indices].drop(index=idx)

                # train classifier based on the set with removed instance
                classifier = self.create_classifier()
                classifier.fit(temp_X, temp_y)
                # predict using trained classifier and removed instance as input
                output = classifier.predict(self.train_input.loc[idx].to_frame().T)

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
            centroid_idx = self.train_input.index[(all_class - all_mean).pow(2).sum(axis=1).idxmin()]
            centroids.add(centroid_idx)
        difference_s = set(centroids)
        i = 0
        while difference_s:
            print(f"Iteration {i}; difference_s ({len(S)})" + " " * 10, end="\r")
            i+=1
            S.update(difference_s)

            reduced_X = self.train_input.loc[list(S)]
            reduced_Y = self.train_output.loc[list(S)]
            classifier = self.create_classifier()  # Assume create_classifier returns an SVM model
            classifier.fit(reduced_X, reduced_Y)

            difference_s = set()
            for idx in self.train_input.index.difference(S):
                predicted_label = classifier.predict(self.train_input.loc[idx].to_frame().T)[0]
                actual_label = self.train_output.loc[idx]

                # If SVM misclassifies, add this index to difference_s
                if predicted_label != actual_label:
                    difference_s.add(idx)

            # If the change in S is below the threshold, stop
            if len(difference_s) < threshold:
                break

            #difference_s = {rep[p] for p in S if rep[p] is not None}
        reduced_X = self.train_input.loc[list(S)].reset_index(drop=True)
        reduced_Y = self.train_output.loc[list(S)].reset_index(drop=True)
        print(f"Final reduced dataset size: {len(reduced_X)} instances.")
        return reduced_X,reduced_Y

    def fit(self, training_input, training_output):
        self.train_input = training_input
        self.train_output = training_output

        if self.reduction_method == ReductionMethod.CONDENSATION:
            self.train_input, self.train_output = self.fcnn1()

        elif self.reduction_method == ReductionMethod.EDITION:
            self.train_input, self.train_output = self.RENN_reduction()

        elif self.reduction_method == ReductionMethod.HYBRID:
            self.train_input, self.train_output = self.DROP3_reduction(self.train_input, self.train_output)

        # train the classifiers on the data
        self.classifier = self.create_classifier()
        self.classifier.fit(self.train_input, self.train_output)

    def predict(self, testing_input):
        # X = np.array(testing_input.values.tolist())
        return self.classifier.predict(testing_input)