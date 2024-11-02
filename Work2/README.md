### Team
- Skaiste Butkute
- Tatevik Davtyan
- Wiktoria Pejs

### Running script for the first time
These sections show how to create virtual environment for
our script and how to install dependencies
1. Open folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env
```bash
python3 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running next
command,it should print list with installed dependencies
```bash
pip list
```

### Run the script for a selected model
1. Open virtual env
```bash
source venv/bin/activate
```
2. Running the script.
```bash
python3 main.py
```
3. Close virtual env
```bash
deactivate
```

The `main.py` script can run any model, here are different parameters (flags) to select different algorithms:
```
  -k K                  k number of neighbours to use in k-NN algorithm
  -d DATASET, --dataset DATASET
                        Name of the dataset to process, sick on default.
  -p PATH, --path PATH  Path to the data directory.
  -l LIMIT, --limit LIMIT
                        Limit to how many datasets to load
  -r RESULT_DIRECTORY, --result-directory RESULT_DIRECTORY
                        Path to the results directory.
  -c CACHE_DIRECTORY, --cache-directory CACHE_DIRECTORY
                        Path to the cache directory for normalised values.
  --disable-cache       Disable saving to and loading normalised data from cache
  -t, --distance-type {euclidean,minkowski,manhattan}
                        KNN: Function for calculating distance, default: euclidean.
  -v, --voting-schema {majority_class,inverse_distance,shepards_work}
                        KNN: Voting schema for selecting neighbors, default: majority_class.
  -R MINKOWSKI_R, --minkowski-r MINKOWSKI_R
                        KNN: r value for minkowski algorithm, default: 3
  -w, --weighting-strategy {equal,filter,wrapper}
                        KNN: Weighting scheme for scaling neighbors, default: equal.
  -i, --instance-reduction {no_reduction,condensation,edition,hybrid}
                        Instance reduction method, default: no_reduction.
  -m, --model {KNN,SVM}
                        Choose model to run, KNN set on default
  -K, --kernel {rbf,polynomial}
                        SVM: Choose classifier kernel, 'polynomial' set on default
```

The script will output the results file into the 'results/{dataset name}' directory next to the `main.py` script.

### Run the script for all models to retrieve results for all algorithm combinations
1. Open virtual env
```bash
source venv/bin/activate
```
2. Running the script for both datasets
```bash
python3 runner.py -d sick
python3 runner.py -d mushroom
```
3. Running the script for the best algorithms in both datasets using reduction methods
```bash
python3 runner.py -d sick -R
python3 runner.py -d mushroom -R
```
4. Close virtual env
```bash
deactivate
```

### Running the analysis script
1. Open virtual env
```bash
source venv/bin/activate
```
1. Running the script evaluating using accuracy and prediction time for sick dataset
```bash
python3 result_analysis.py -d sick -m 'Accuracy'
python3 result_analysis.py -d sick -m 'Pred. Time'
```
2. Running the script evaluating using accuracy and prediction time for mushroom dataset
```bash
python3 result_analysis.py -d mushroom -m 'Accuracy'
python3 result_analysis.py -d mushroom -m 'Pred. Time'
```
3. Running the script evaluating instance reduction models using accuracy, prediction and storage time for sick dataset
```bash
python3 result_analysis.py -d sick -m 'Accuracy' -r ./results_reduced
python3 result_analysis.py -d sick -m 'Pred. Time' -r ./results_reduced
python3 result_analysis.py -d sick -m 'Storage' -r ./results_reduced
```
4. Running the script evaluating instance reduction models using accuracy, prediction and storage time for mushroom dataset
```bash
python3 result_analysis.py -d mushroom -m 'Accuracy' -r ./results_reduced
python3 result_analysis.py -d mushroom -m 'Pred. Time' -r ./results_reduced
python3 result_analysis.py -d mushroom -m 'Storage' -r ./results_reduced
```
5. Close virtual env
```bash
deactivate
```