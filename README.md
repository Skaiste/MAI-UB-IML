# MAI-UB-IML

## Work 2, Deadline - November, 3rd

### Steps:
 - [x] Read the source data for training and testing the algorithms
 - [x] Read and save the information from a training and their corresponding testing  file in a TrainMatrix and a TestMatrix, respectively. Normalize all the numerical attributes in the range [0..1].
 - [x] Write a python fn that repeats the previous process for the 10-fold cross-validation files.
 - [x] Write a Python function for classifying, using a kNN algorithm.
 - [ ] Write a Python function for classifying, using an SVM algorithm.
 - [ ] Modify the kNN algorithm so that it includes a pre-processing for reducing the training set. Instance reduction techniques!


### Datasets:
1. Sick
    - Number of Cases: 3 772
    - Numeric attributes: 7
    - Nominal attributes: 22
    - Classes: 2
    - Deviation of class distribution: 43.88%
    - Instances belonging to the majority class: 93.88%
    - Instances belonging to the minority class: 6.12%
    - Missing values: 5.54%

2. Mushroom
    - Number of Cases: 8 124
    - Nominal attributes: 22
    - Classes: 2
    - Deviation of class distribution: 1.80%
    - Instances belonging to the majority class: 51.80%
    - Instances belonging to the minority class: 48.20%
    - Missing values: 1.38%

## KNN accuracy results for sick dataset before feature reduction

| Distance | Voting | Weighting | Accuracy | Pred. Time | Accuracy | Pred. Time | Accuracy | Pred. Time | Accuracy | Pred. Time |
|---|---|---|---|---|---|---|---|---|---|---|
|||| K1 | K1 | K3 | K3 | K5 | K5 | K7 | K7 |
| manhattan | shepards_work | filter | 97.13639 | 0.62944 | 97.58732 | 0.63044 | 97.77328 | 0.62951 | 97.56101 | 0.62594 |
| manhattan | shepards_work | equal | 96.55382 | 0.63713 | 97.08439 | 0.62584 | 97.05773 | 0.62980 | 96.97787 | 0.62767 |
| manhattan | shepards_work | wrapper | 96.55382 | 0.62731 | 97.08439 | 0.62772 | 97.05773 | 0.63531 | 96.97787 | 0.63277 |
| manhattan | majority_class | equal | 96.55382 | 0.62826 | 97.11106 | 0.64406 | 97.08404 | 0.64111 | 97.00433 | 0.64050 |
| manhattan | majority_class | filter | 97.13654 | 0.64391 | 97.58746 | 0.63973 | 97.74654 | 0.64256 | 97.58760 | 0.64527 |
| euclidean | shepards_work | filter | 97.13716 | 0.64238 | 97.61469 | 0.65236 | 97.48199 | 0.64946 | 97.64079 | 0.64884 |
| euclidean | shepards_work | equal | 96.73956 | 0.64643 | 96.97843 | 0.64650 | 97.03106 | 0.64516 | 97.19028 | 0.64804 |
| euclidean | shepards_work | wrapper | 96.73956 | 0.64858 | 96.97843 | 0.64689 | 97.03106 | 0.65811 | 97.19028 | 0.65671 |
| euclidean | majority_class | filter | 97.13745 | 0.66457 | 97.42908 | 0.66322 | 97.40228 | 0.65727 | 97.53483 | 0.65246 |
| manhattan | majority_class | wrapper | 96.55382 | 0.66004 | 97.11106 | 0.65006 | 97.08404 | 0.65221 | 97.00433 | 0.66224 |
| euclidean | majority_class | equal | 96.73956 | 0.64885 | 96.87240 | 0.73380 | 97.05759 | 0.69389 | 97.13723 | 0.65933 |
| euclidean | majority_class | wrapper | 96.73956 | 0.67286 | 96.87240 | 0.65169 | 97.05759 | 0.66113 | 97.13723 | 0.65321 |
| minkowski | shepards_work | equal | 96.73956 | 0.66832 | 96.97843 | 0.66521 | 97.03106 | 0.65648 | 97.19028 | 0.66256 |
| minkowski | shepards_work | filter | 97.21688 | 0.67541 | 97.53511 | 0.66498 | 97.53490 | 0.66027 | 97.66746 | 0.67143 |
| euclidean | inverse_distance | equal | 96.73956 | 0.65532 | 96.44800 | 0.66173 | 96.95156 | 0.67015 | 96.97808 | 0.67889 |
| minkowski | majority_class | equal | 96.73956 | 0.66136 | 96.87240 | 0.65984 | 97.05759 | 0.65854 | 97.13723 | 0.66128 |
| manhattan | inverse_distance | filter | 97.24264 | 0.66245 | 97.53476 | 0.66901 | 97.66669 | 0.67902 | 97.58788 | 0.66911 |
| minkowski | shepards_work | wrapper | 96.73956 | 0.67334 | 96.97843 | 0.66484 | 97.03106 | 0.67226 | 97.19028 | 0.66220 |
| euclidean | inverse_distance | filter | 97.05766 | 0.66135 | 97.03134 | 0.67637 | 97.42831 | 0.67490 | 97.18979 | 0.68076 |
| minkowski | majority_class | filter | 97.19022 | 0.66939 | 97.53511 | 0.67083 | 97.42873 | 0.67146 | 97.58753 | 0.66728 |
| manhattan | inverse_distance | equal | 96.55382 | 0.67061 | 96.31530 | 0.67057 | 97.03127 | 0.66421 | 96.84518 | 0.67136 |
| euclidean | inverse_distance | wrapper | 96.73956 | 0.66407 | 96.44800 | 0.67370 | 96.95156 | 0.67087 | 96.97808 | 0.68463 |
| minkowski | shepards_work | filter | 97.11008 | 0.67743 | 97.56094 | 0.67431 | 97.69356 | 0.67247 | 97.66718 | 0.68742 |
| minkowski | majority_class | equal | 96.55382 | 0.67214 | 97.11106 | 0.67511 | 97.08404 | 0.68451 | 97.00433 | 0.66812 |
| minkowski | shepards_work | equal | 96.55382 | 0.67045 | 97.08439 | 0.68198 | 97.05773 | 0.69765 | 96.97787 | 0.67089 |
| minkowski | majority_class | filter | 97.08334 | 0.68026 | 97.77286 | 0.68086 | 97.79952 | 0.67812 | 97.72016 | 0.68406 |
| minkowski | majority_class | wrapper | 96.55382 | 0.67568 | 97.11106 | 0.69071 | 97.08404 | 0.70214 | 97.00433 | 0.70887 |
| minkowski | shepards_work | wrapper | 96.55382 | 0.68864 | 97.08439 | 0.68181 | 97.05773 | 0.68327 | 96.97787 | 0.68056 |
| manhattan | inverse_distance | wrapper | 96.55382 | 0.68706 | 96.31530 | 0.68001 | 97.03127 | 0.68135 | 96.84518 | 0.68255 |
| minkowski | inverse_distance | equal | 96.55382 | 0.69918 | 96.31530 | 0.67645 | 97.03127 | 0.70110 | 96.84518 | 0.69771 |
| minkowski | inverse_distance | equal | 96.73956 | 0.67965 | 96.44800 | 0.68831 | 96.95156 | 0.69358 | 96.97808 | 0.69269 |
| minkowski | inverse_distance | filter | 97.05780 | 0.74470 | 96.97794 | 0.68158 | 97.29568 | 0.68970 | 97.58760 | 0.69029 |
| minkowski | majority_class | wrapper | 96.73956 | 0.70725 | 96.87240 | 0.69014 | 97.05759 | 0.68222 | 97.13723 | 0.69773 |
| minkowski | inverse_distance | filter | 97.24278 | 0.71239 | 97.53455 | 0.70152 | 97.61371 | 0.69123 | 97.53511 | 0.71060 |
| minkowski | inverse_distance | wrapper | 96.73956 | 0.71139 | 96.44800 | 0.69087 | 96.95156 | 0.70057 | 96.97808 | 0.70897 |
| minkowski | inverse_distance | wrapper | 96.55382 | 0.72754 | 96.31530 | 0.70987 | 97.03127 | 0.71994 | 96.84518 | 0.70085 |


### Notes from loading and parsing arrf files:
The testing set contains more abnormal entries where:
 - an entry is incomplete
 - two incomplete entries are joined together

To resolve abnormal entries the joined entries were seperated and then filled the attributes that don't fit the structure with '?' (for nominal attributes) or nans (for numerical attributes). The incomplete entries were filled with '?' and nans the same way.

### Notes on normalisation
 - For nominal values, where the value isn't specified, i.e set as '?', it is normalised as a 0 and other values are set as their index in the list of possible values
 - The numerical values are normalised to a range of [0..1]


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
