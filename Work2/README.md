### Explanation on how to execute the code
```
    usage: main.py [-h] [-k K] [-d DATASET] [-p PATH] [-l LIMIT] [-r RESULT_DIRECTORY] [-t {euclidean,minkowski,manhattan}]
                   [-v {majority_class,inverse_distance,shepherds_work}] [-w {equal,filter,wrapper}]

    optional arguments:
    -h, --help            show this help message and exit
    -k K                  k number of neighbours to use in k-NN algorithm
    -d DATASET, --dataset DATASET
                            Name of the dataset to process, adult on default.
    -p PATH, --path PATH  Path to the data directory.
    -l LIMIT, --limit LIMIT
                            Limit to how many datasets to load
    -r RESULT_DIRECTORY, --result-directory RESULT_DIRECTORY
                            Path to the results directory.
    -t {euclidean,minkowski,manhattan}, --distance-type {euclidean,minkowski,manhattan}
                            Function for calculating distance, default: euclidean.
    -v {majority_class,inverse_distance,shepherds_work}, --voting-schema {majority_class,inverse_distance,shepherds_work}
                            Voting schema for selecting neighbors, default: majority_class.
    -w {equal,filter,wrapper}, --weighting-strategy {equal,filter,wrapper}
                            Weighting scheme for scaling neighbors, default: equal.
```