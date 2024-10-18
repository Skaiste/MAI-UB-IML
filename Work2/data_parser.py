import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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


            
#Edit by Tatevik
class Normalization:
    def __init__(self):
        self.encoders = {}
    def normalise(self,dataframes, num_norm, train=True):
        if len(dataframes) == 0:
            return []

        for i in  range(len(dataframes)):

            df = dataframes[i]

            for name, dt in df.dtypes.items():
                if dt.name == 'float64':

                    dataframes[i][name] = num_norm.normalise(i, name, df[name], train)
                else:

                    #Edit by Tatevik
                    if name != "class" and name != "Class":

                        column_reshaped = dataframes[i][name].values.reshape(-1, 1)
                        if train:

                            if i not in self.encoders:
                                self.encoders[i] = {}
                            if name not in self.encoders[i]:
                                self.encoders[i][name]= OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
                                self.encoders[i][name].fit(column_reshaped)

                        encoded = self.encoders[i][name].transform(column_reshaped)
                        encoded = pd.DataFrame(
                            encoded,
                            columns= self.encoders[i][name].get_feature_names_out([name])
                        )

                        dataframes[i] = pd.concat([dataframes[i].drop(name, axis=1), encoded], axis=1)

        return


def replace_missing_data(dataframes, numerical_means, common_nominals):
    # deal with missing data:
    #  - if 25% of the entry is missing -> remove the entry
    #  - else if a nominal attribute is missing -> use the most common value in the category
    #  - else if a numerical attribute is missing -> use the mean of the attribute across all entries
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
                common_nominals[i][name] = df[name].dropna().mode().values[0].decode('utf-8')

    replace_missing_data(training_ds, numerical_means, common_nominals)
    replace_missing_data(testing_ds, numerical_means, common_nominals)

    # normalise both datasets
    numerical_normaliser = MinMaxNormalisation()

    #Edit by  Tatevik
    general_normalizer = Normalization()

    print("Normalising training set")
    general_normalizer.normalise(training_ds, numerical_normaliser)

    print("Normalising testing set")
    general_normalizer.normalise(testing_ds, numerical_normaliser, train=False)

    #Edit by Tatevik
    input_columns = {}
    output_column = {}
    for i in range(len(training_ds)):
        input_columns_i = list(training_ds[i].columns)
        if 'Class' in list(input_columns_i):
            input_columns_i.remove('Class')
            input_columns[i] = input_columns_i
            output_column[i] = 'Class'

        else:
            input_columns_i.remove('class')
            input_columns[i] = input_columns_i
            output_column[i] = 'class'


    train_input = [training_ds[i][input_columns[i]] for i in range(len(training_ds))]
    train_output = [training_ds[i][output_column[i]] for i in range(len(training_ds))]
    test_input = [testing_ds[i][input_columns[i]] for i in range(len(testing_ds))]
    test_output = [testing_ds[i][output_column[i]] for i in range(len(testing_ds))]

    return train_input, train_output, test_input, test_output
