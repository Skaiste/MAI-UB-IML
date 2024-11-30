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
    def __init__(self, nominal_values, normalise_nominal=True):
        self.encoders = {}
        self.num_norm = MinMaxNormalisation()
        self.nom_vals = nominal_values
        self.normalise_nominal = normalise_nominal

    def normalise(self, df, train=True):
        for name, dt in df.dtypes.items():
            if dt.name == 'float64':
                df[name] = self.num_norm.normalise(name, df[name], train)
            else:
                #Edit by Tatevik
                if name.lower() != "class":
                    if self.normalise_nominal:
                        column_reshaped = df[name].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x).values.reshape(-1, 1)
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
                    else:
                        df[name] = df[name].apply(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else x)
                        df[name] = self.num_norm.normalise(name, df[name], train)
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


def get_data(data_fn, cache=True, cache_dir=None, normalise_nominal=True):
    df = None
    if cache:
        cache_fn = cache_dir / (data_fn.stem + '.pkl')
        if cache_fn.is_file():
            df = pd.read_pickle(cache_fn)

    if df is None:
        data = loadarff(data_fn)
        df = pd.DataFrame(data[0], columns=data[1]._attributes.keys())

        # retrieving means for numerical and most frequent category for nominal data
        allow_to_remove = 'TBG'
        numerical_means = {}
        common_nominals = {}
        for name, dt in df.dtypes.items():
            if dt.name == 'float64':
                # if the whole column has missing data -> remove the whole column in both
                #       training and testing dataframes of the same fold
                if len(df[name].unique()) == 1 and name == allow_to_remove:
                    df.drop(name, axis='columns', inplace=True)

                else:
                    numerical_means[name] = np.nanmean(df[name])
            else:
                common_nominals[name] = df[name].dropna().mode().values[0].decode('utf-8')

        replace_missing_data(df, numerical_means, common_nominals)

        #Edit by  Tatevik
        nominal_values = {attr:list(data[1][attr][1]) for attr in data[1].names() if data[1][attr][0] == 'nominal'}
        general_normalizer = Normalization(nominal_values, normalise_nominal=normalise_nominal)
        df = general_normalizer.normalise(df)

        # cache normalised data
        if cache and cache_dir is not None:
            df.to_pickle(cache_fn)

    #Edit by Tatevik
    input_columns = {}
    output_column = {}
    input_columns_i = list(df.columns)
    if 'Class' in list(input_columns_i):
        input_columns_i.remove('Class')
        input_columns = input_columns_i
        output_column = 'Class'
    else:
        input_columns_i.remove('class')
        input_columns = input_columns_i
        output_column = 'class'

    input = df[input_columns]
    df[output_column] = df[output_column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    output = df[output_column]

    return input, output
