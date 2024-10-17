import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from loadarff import loadarff

class MinMaxNormalisation:
    def __init__(self):
        self.minval = {}
        self.maxval = {}

    def normalise(self, fold, name, values, train=True):
        if train:
            if fold not in self.minval:
                self.minval[fold] = {}
            if fold not in self.maxval:
                self.maxval[fold] = {}
            self.minval[fold][name] = values.min()
            self.maxval[fold][name] = values.max()
        return (values - self.minval[fold][name]) / (self.maxval[fold][name] - self.minval[fold][name])
            

def normalise(dataframes, num_norm, train=True):
    if len(dataframes) == 0:
        return []
    le = LabelEncoder()
    for i,df in enumerate(dataframes):
        for name, dt in df.dtypes.items():
            if dt.name == 'float64':
                df[name] = num_norm.normalise(i, name, df[name], train)
            else:
                df[name] = le.fit_transform(df[name])
    return


def replace_missing_data(dataframes, numerical_means, common_nominals):
    # deal with missing data:
    #  - if 25% of the entry is missing -> remove the entry
    #  - if a nominal attribute is missing -> use the most common value in the category
    #  - if a numerical attribute is missing -> use the mean of the attribute across all entries
    for i, df in enumerate(dataframes):
        for j, row in df.iterrows():
            count = np.sum(row.isna().values)
            if count > len(df.dtypes) * 0.25:
                df.drop(j, inplace=True)

            for k in range(len(row)):
                if not isinstance(row.iloc[k], bytes) and np.isnan(row.iloc[k]):
                    if df.dtypes.iloc[k].name == 'float64':
                        # print(f"numerical in fold {i}, attr {df.dtypes.keys()[k]} in row {j} to {numerical_means[i][df.dtypes.keys()[k]]}")
                        row.iloc[k] = numerical_means[i][df.dtypes.keys()[k]]
                    else:
                        # print(f"nominal in fold {i}, attr {df.dtypes.keys()[k]} in row {j} to {common_nominals[i][df.dtypes.keys()[k]]}")
                        row.iloc[k] = common_nominals[i][df.dtypes.keys()[k]]


def get_data(training_fns, testing_fns):
    training_ds = []
    for fn in training_fns:
        traw = loadarff(fn)
        training_ds.append(pd.DataFrame(traw[0], columns=traw[1]._attributes.keys()))

    testing_ds = []
    for fn in testing_fns:
        traw = loadarff(fn)
        testing_ds.append(pd.DataFrame(traw[0], columns=traw[1]._attributes.keys()))

    # retrieving means for numerical and most frequent category for nominal data
    allow_to_remove = 'TBG'
    numerical_means = {}
    common_nominals = {}
    for i, df in enumerate(training_ds):
        numerical_means[i] = {}
        common_nominals[i] = {}
        for name, dt in df.dtypes.items():
            if dt.name == 'float64':
                # if the whole column has missing data -> remove the whole column in both
                #       training and testing dataframes of the same fold
                if len(df[name].unique()) == 1 and name == allow_to_remove:
                    df.drop(name, axis='columns', inplace=True)
                    # remove the same column in the test dataframe
                    testing_ds[i].drop(name, axis='columns', inplace=True)
                else:
                    numerical_means[i][name] = np.nanmean(df[name])
            else:
                common_nominals[i][name] = df[name].mode().values[0].decode('utf-8')

    replace_missing_data(training_ds, numerical_means, common_nominals)
    replace_missing_data(testing_ds, numerical_means, common_nominals)

    # normalise both datasets
    numerical_normaliser = MinMaxNormalisation()
    print("Normalising training set")
    normalise(training_ds, numerical_normaliser)

    print("Normalising testing set")
    normalise(testing_ds, numerical_normaliser, train=False)

    input_columns = list(training_ds[0].columns)
    if 'Class' in list(training_ds[0].columns):
        input_columns.remove('Class')
        output_column = 'Class'
    else:
        input_columns.remove('class')
        output_column = 'class'

    train_input = [tids[input_columns] for tids in training_ds]
    train_output = [tods[output_column] for tods in training_ds]
    test_input = [tids[input_columns] for tids in testing_ds]
    test_output = [tods[output_column] for tods in testing_ds]

    return train_input, train_output, test_input, test_output
