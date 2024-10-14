import pandas as pd
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


def get_data(training_fns, testing_fns):
    training_ds = []
    for fn in training_fns:
        traw = loadarff(fn)
        training_ds.append(pd.DataFrame(traw[0], columns=traw[1]._attributes.keys()))

    testing_ds = []
    for fn in testing_fns:
        traw = loadarff(fn)
        testing_ds.append(pd.DataFrame(traw[0], columns=traw[1]._attributes.keys()))

    # normalise both datasets
    numerical_normaliser = MinMaxNormalisation()
    print("Normalising training set")
    normalise(training_ds, numerical_normaliser)

    print("Normalising testing set")
    normalise(testing_ds, numerical_normaliser, train=False)

    input_columns = list(training_ds[0].columns)
    input_columns.remove('class')
    output_column = 'class'

    train_input = [tids[input_columns] for tids in training_ds]
    train_output = [tods[output_column] for tods in training_ds]
    test_input = [tids[input_columns] for tids in testing_ds]
    test_output = [tods[output_column] for tods in training_ds]

    return train_input, train_output, test_input, test_output
