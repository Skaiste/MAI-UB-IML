# MAI-UB-IML

## Work 2, Deadline - November, 3rd
### Datasets:
 1. Adult
    - Number of Cases: 48 842
    - Numeric attributes: 6
    - Nominal attributes: 8
    - Classes: 2
    - Deviation of class distribution: 26.07%
    - Instances belonging to the majority class: 76.07%
    - Instances belonging to the minority class: 23.93%
    - Missing values: 0.95%

2. Mushroom
    - Number of Cases: 8 124
    - Nominal attributes: 22
    - Classes: 2
    - Deviation of class distribution: 1.80%
    - Instances belonging to the majority class: 51.80%
    - Instances belonging to the minority class: 48.20%
    - Missing values: 1.38%

### Steps:
 - [ ] Read the source data for training and testing the algorithms
 - [ ] Read and save the information from a training and their corresponding testing  file in a TrainMatrix and a TestMatrix, respectively. Normalize all the numerical attributes in the range [0..1].
 - [ ] Write a python fn that repeats the previous process for the 10-fold cross-validation files.
 - [ ] Write a Python function for classifying, using a kNN algorithm.
 - [ ] Write a Python function for classifying, using an SVM algorithm.
 - [ ] Modify the kNN algorithm so that it includes a pre-processing for reducing the training set.

### Untidy notes from session 2 Labs:

Use lazy learning

training step: just preprocessing 
interference step:
 - Nearest neighbour
	 - Distance metric - Euclidean distance
	 - Look at **one** neighbour
	 - Weighting is not used
	 - Predict the same output as the nearest neighbour
 - k-Nearest neighbour
	 - Distance metric - Minkowski ($r=1$, $r= 2$) and another one
	 - Look at one or $K$ neighbours
	 - Weighting: analyse weighted and unweighted approaches
	 - Fitting local points: 3 voting rules
	 - Learning solved cases: None

Find the best k-NN algorithm for the multiple data

We will **die** if we don't do the algorithms before the evaluation session!!!

Large datasets require a lot of time for training (depending on the machine).

Decision for algorithm should be reasoned in the report
