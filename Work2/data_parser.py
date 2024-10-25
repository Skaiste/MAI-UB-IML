import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from loadarff import loadarff

class MinMaxNormalisation:
    def __init__(self):
        self.minval = {}
        self.maxval = {}

    def get_min_max(self, name, values):
        self.minval[name] = values.min()
        self.maxval[name] = values.max()

    def normalise(self, name, values, train=True):
        if train:
            self.get_min_max(name, values)

        return (values - self.minval[name]) / (self.maxval[name] - self.minval[name])

            
#Edit by Tatevik
class Normalization:
    def __init__(self, nominal_values):
        self.encoders = {}
        self.num_norm = MinMaxNormalisation()
        self.nom_vals = nominal_values

    def normalise(self, df, train=True):
        for name, dt in df.dtypes.items():
            if dt.name == 'float64':
                df[name] = self.num_norm.normalise(name, df[name], train)
            else:
                #Edit by Tatevik
                if name.lower() != "class":
                    column_reshaped = df[name].values.reshape(-1, 1)
                    columns = [f"{name}_{a}" for a in self.nom_vals[name]]
                    if train:
                        if name not in self.encoders:
                            self.encoders[name] = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[self.nom_vals[name]])
                            self.encoders[name].fit(column_reshaped)

                    encoded = self.encoders[name].transform(column_reshaped)
                    encoded = pd.DataFrame(
                        encoded,
                        columns=self.encoders[name].get_feature_names_out([name])
                    )

                    df = pd.concat([df.drop(name, axis=1), encoded], axis=1)
        return df


def replace_missing_data(df, numerical_means, common_nominals):
    # deal with missing data:
    #  - if 25% of the entry is missing -> remove the entry
    #  - else if a nominal attribute is missing -> use the most common value in the category
    #  - else if a numerical attribute is missing -> use the mean of the attribute across all entries
    for j, row in df.iterrows():
        count = np.sum(row.isna().values)
        if count > len(df.dtypes) * 0.25:
            df.drop(j, inplace=True)

        for k in range(len(row)):
            if not isinstance(row.iloc[k], bytes) and np.isnan(row.iloc[k]):
                if df.dtypes.iloc[k].name == 'float64':
                    df.iloc[j, k] = numerical_means[df.dtypes.keys()[k]]
                else:
                    df.iloc[j, k] = common_nominals[df.dtypes.keys()[k]]


def get_data(data_fns, cache=True, cache_dir=None):
    train_input = []
    train_output = []
    test_input = []
    test_output = []

    for files in data_fns.values():
        training_df = None
        testing_df = None
        if cache:
            training_cache_fn = cache_dir / (files['training'].stem + '.pkl')
            if training_cache_fn.is_file():
                training_df = pd.read_pickle(training_cache_fn)
            testing_cache_fn = cache_dir / (files['testing'].stem + '.pkl')
            if testing_cache_fn.is_file():
                testing_df = pd.read_pickle(testing_cache_fn)

        if training_df is None or testing_df is None:
            traw = loadarff(files['training'])
            training_df = pd.DataFrame(traw[0], columns=traw[1]._attributes.keys())
            traw = loadarff(files['testing'])
            testing_df = pd.DataFrame(traw[0], columns=traw[1]._attributes.keys())

            # retrieving means for numerical and most frequent category for nominal data
            allow_to_remove = 'TBG'
            numerical_means = {}
            common_nominals = {}
            for name, dt in training_df.dtypes.items():
                if dt.name == 'float64':
                    # if the whole column has missing data -> remove the whole column in both
                    #       training and testing dataframes of the same fold
                    if len(training_df[name].unique()) == 1 and name == allow_to_remove:
                        training_df.drop(name, axis='columns', inplace=True)
                        # remove the same column in the test dataframe
                        testing_df.drop(name, axis='columns', inplace=True)
                    else:
                        numerical_means[name] = np.nanmean(training_df[name])
                else:
                    common_nominals[name] = training_df[name].dropna().mode().values[0].decode('utf-8')

            replace_missing_data(training_df, numerical_means, common_nominals)
            replace_missing_data(testing_df, numerical_means, common_nominals)

            #Edit by  Tatevik
            nominal_values = {attr:list(traw[1][attr][1]) for attr in traw[1].names() if traw[1][attr][0] == 'nominal'}
            general_normalizer = Normalization(nominal_values)
            training_df = general_normalizer.normalise(training_df)
            testing_df = general_normalizer.normalise(testing_df, train=False)

            # cache normalised data
            if cache and cache_dir is not None:
                training_df.to_pickle(training_cache_fn)
                testing_df.to_pickle(testing_cache_fn)

        #Edit by Tatevik
        input_columns = {}
        output_column = {}
        input_columns_i = list(training_df.columns)
        if 'Class' in list(input_columns_i):
            input_columns_i.remove('Class')
            input_columns = input_columns_i
            output_column = 'Class'
        else:
            input_columns_i.remove('class')
            input_columns = input_columns_i
            output_column = 'class'

        train_input.append(training_df[input_columns])
        training_df[output_column] = training_df[output_column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        train_output.append(training_df[output_column])
        test_input.append(testing_df[input_columns])
        testing_df[output_column] = testing_df[output_column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        test_output.append(testing_df[output_column])

    return train_input, train_output, test_input, test_output
