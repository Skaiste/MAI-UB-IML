import numpy as np
from sklearn import svm
from enum import Enum
from sklearn.preprocessing import MultiLabelBinarizer

# Classifyer Kernel Types https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py
class KernelType(Enum):
    RBF = 'rbf'                 # radial basis function
    POLYNOMIAL = 'polynomial'

class SVM:
    def __init__(self, kernel):
        # setup classifiers
        if kernel == KernelType.RBF:
            self.classifier = svm.SVC(kernel='rbf', gamma='scale')
        elif kernel == KernelType.POLYNOMIAL:
            self.classifier = svm.SVC(kernel='poly', degree=3, gamma='scale', coef0=1)
        else:
            raise Exception("Unknown classifier specified")

    def fit(self, training_input, training_output):
        # train the classifiers on the data
        self.classifier.fit(training_input, training_output)

    def predict(self, testing_input):
        # X = np.array(testing_input.values.tolist())
        return self.classifier.predict(testing_input)