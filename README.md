# MAI-UB-IML

## Work 3, Deadline - December, 8th

Edit report [Here](https://docs.google.com/document/d/1-Ye6cBojwXEHfyYGivocpvwlN2uZXfafMD3xGi19afU/edit?tab=t.0)

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

3. cmc
    - Number of Cases: 1 473
    - Numeric attributes: 2
    - Nominal attributes: 7
    - Classes: 3
    - Deviation of class distribution: 8.26%
    - Instances belonging to the majority class: 42.70%
    - Instances belonging to the minority class: 22.61%



## Work 2, Deadline - November, 3rd

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

| Distance | Voting | Weighting | Accuracy | Pred. Time | Accuracy | Pred. Time | Accuracy | Pred. Time | Accuracy | Pred. Time | Best k |
|---|---|---|---|---|---|---|---|---|---|---|---|
|||| K1 | K1 | K3 | K3 | K5 | K5 | K7 | K7 ||
| euclidean | majority_class | equal | 95.89160 | 0.48737 | 95.86430 | 0.48270 | 96.20878 | 0.46841 | 96.26190 | 0.46183 | 7 |
| manhattan | majority_class | filter | 97.58739 | 0.64161 | 97.72030 | 0.89665 | 97.58739 | 0.95843 | 97.45533 | 0.91822 | 1 |
| manhattan | shepards_work | filter | 97.58739 | 0.66061 | 97.69363 | 0.66532 | 97.71995 | 0.65684 | 97.69363 | 0.65779 | 5 |
| manhattan | majority_class | equal | 95.62606 | 0.66470 | 95.81076 | 0.67922 | 96.26204 | 0.66267 | 96.12935 | 0.65826 | 7 |
| manhattan | shepards_work | wrapper | 96.36870 | 0.66557 | 96.63339 | 0.66295 | 96.50098 | 0.66635 | 96.71318 | 0.67527 | 3 |
| manhattan | shepards_work | equal | 95.62606 | 0.66312 | 95.70571 | 0.67209 | 95.83806 | 0.66730 | 95.86451 | 0.68108 | 1 |
| euclidean | majority_class | filter | 96.92510 | 0.68479 | 96.87156 | 0.68939 | 96.87219 | 0.73314 | 97.03134 | 0.79163 | 1 |
| euclidean | shepards_work | equal | 95.89160 | 0.68234 | 95.91784 | 0.68611 | 95.91749 | 0.67804 | 96.07650 | 0.67928 | 5 |
| manhattan | inverse_distance | equal | 95.62606 | 0.67807 | 95.70501 | 0.68082 | 95.54754 | 0.69004 | 95.38734 | 0.68990 | 1 |
| euclidean | shepards_work | filter | 96.92510 | 0.70519 | 97.13674 | 0.69206 | 97.05752 | 0.69324 | 97.24334 | 0.74089 | 3 |
| manhattan | inverse_distance | wrapper | 96.36870 | 0.68665 | 96.10233 | 0.71627 | 96.26267 | 0.71238 | 96.31558 | 0.70863 | 1 |
| euclidean | shepards_work | wrapper | 96.20997 | 0.69586 | 96.44821 | 0.70556 | 96.42168 | 0.74770 | 96.60736 | 0.72980 | 1 |
| euclidean | majority_class | wrapper | 96.20997 | 0.70259 | 96.42126 | 0.69772 | 96.47431 | 0.71287 | 96.63318 | 0.71387 | 3 |
| manhattan | inverse_distance | filter | 97.58739 | 0.94687 | 97.21709 | 1.00387 | 97.11050 | 1.00152 | 97.29632 | 0.70729 | 7 |
| euclidean | inverse_distance | wrapper | 96.20997 | 0.70085 | 96.02416 | 0.71392 | 96.07686 | 0.71748 | 95.99735 | 0.72451 | 1 |
| euclidean | inverse_distance | filter | 96.92510 | 0.74238 | 96.34183 | 0.71620 | 96.66013 | 0.73943 | 96.55389 | 0.72490 | 3 |
| euclidean | inverse_distance | equal | 95.89160 | 0.73641 | 95.89076 | 0.72615 | 96.12913 | 0.78553 | 95.62578 | 0.79529 | 3 |
| manhattan | majority_class | wrapper | 96.36870 | 0.93362 | 96.65943 | 0.94236 | 96.60701 | 0.95883 | 96.71297 | 0.90647 | 7 |
| minkowski | shepards_work | filter | 96.60687 | 1.06628 | 96.71290 | 1.05283 | 96.76609 | 1.03741 | 96.87233 | 0.96712 | 7 |
| minkowski | majority_class | wrapper | 96.20990 | 0.96304 | 96.26197 | 1.04454 | 96.39460 | 0.98180 | 96.55382 | 0.98179 | 1 |
| minkowski | majority_class | equal | 95.75918 | 0.96675 | 95.78445 | 0.98657 | 95.99679 | 1.03284 | 96.15601 | 1.02612 | 1 |
| minkowski | majority_class | filter | 96.34162 | 0.98412 | 96.52687 | 0.99887 | 96.39467 | 1.00154 | 96.71325 | 0.97773 | 7 |
| minkowski | shepards_work | filter | 96.34162 | 1.03372 | 96.50042 | 1.02019 | 96.55382 | 0.97979 | 96.71325 | 1.05410 | 5 |
| minkowski | shepards_work | equal | 95.75918 | 0.98232 | 95.91756 | 0.98151 | 95.91763 | 0.98254 | 96.10317 | 0.97942 | 7 |
| minkowski | majority_class | filter | 96.60687 | 1.00688 | 96.65999 | 1.00479 | 96.60694 | 0.98519 | 96.76623 | 1.03306 | 5 |
| minkowski | majority_class | wrapper | 96.26267 | 0.98561 | 96.42119 | 1.03454 | 96.55389 | 1.03911 | 96.52708 | 1.00222 | 1 |
| minkowski | shepards_work | equal | 95.83848 | 0.98650 | 95.94409 | 1.01299 | 96.02366 | 0.99330 | 96.07664 | 0.98984 | 1 |
| minkowski | majority_class | equal | 95.83848 | 0.98962 | 95.89055 | 1.00098 | 96.10275 | 0.99171 | 96.26197 | 1.00043 | 5 |
| minkowski | inverse_distance | wrapper | 96.20990 | 0.99539 | 96.23544 | 1.01723 | 96.15594 | 1.02846 | 95.86451 | 1.07648 | 1 |
| minkowski | inverse_distance | filter | 96.34162 | 1.00296 | 96.36835 | 1.04291 | 96.20906 | 1.01985 | 96.36884 | 1.01910 | 1 |
| minkowski | inverse_distance | equal | 95.75918 | 1.02949 | 95.78445 | 1.02214 | 95.99679 | 1.03853 | 95.62621 | 1.04451 | 3 |
| minkowski | inverse_distance | wrapper | 96.26267 | 1.02925 | 96.34190 | 1.02819 | 96.10331 | 1.04314 | 95.75848 | 1.15012 | 3 |
| minkowski | inverse_distance | filter | 96.60687 | 1.07368 | 96.44835 | 1.11725 | 96.31537 | 1.03973 | 96.44779 | 1.03102 | 7 |
| minkowski | shepards_work | wrapper | 96.20990 | 1.06229 | 96.26253 | 1.05203 | 96.42168 | 1.04890 | 96.58069 | 1.04549 | 7 |
| minkowski | inverse_distance | equal | 95.83848 | 1.04174 | 95.91700 | 1.05210 | 96.12920 | 1.06004 | 95.46691 | 1.05832 | 1 |
| minkowski | shepards_work | wrapper | 96.26267 | 1.08242 | 96.36863 | 1.06983 | 96.50133 | 1.08450 | 96.63382 | 1.06248 | 7 |

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
