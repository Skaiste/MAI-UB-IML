import math
import numpy as np
from loadarff import loadarff

def get_normalising_minmax(data_sets):
    print("Preparing for normalising data")
    minmax = {}
    for data in data_sets:
        for batch in data:
            for entry in batch[0].tolist():
                for idx, val in enumerate(entry):
                    if batch[1].types()[idx] == "numeric":
                        attr_name = batch[1].names()[idx]
                        if attr_name not in minmax:
                            minmax[attr_name] = {"min": val, "max": val}
                        elif not math.isnan(val) and minmax[attr_name]["min"] > val or math.isnan(minmax[attr_name]["min"]):
                            minmax[attr_name]["min"] = val
                        elif not math.isnan(val) and minmax[attr_name]["max"] < val or math.isnan(minmax[attr_name]["max"]):
                            minmax[attr_name]["max"] = val
    return minmax

def normalise_numeric(val, minmax):
    if math.isnan(val):
        return np.nan
    return (val - minmax["min"]) / (minmax["max"] - minmax["min"])

def normalise_nominal(val, choices):
    if val not in choices:
        for ch in choices:
            if set(val).issubset(set(ch)):
                return choices.index(ch) + 1
        if val == '?':
            return 0
    return choices.index(val) + 1

def normalise(dataset, minmax):
    if len(dataset) == 0:
        return []

    choices = {name:attr.values for name, attr in dataset[0][1]._attributes.items() if attr.type_name == "nominal"}
    normalised_data = []
    norm_class_data = []
    for raw_data in dataset:
        batch = []
        class_batch = []
        for entry in raw_data[0].tolist():
            norm_entry = [0] * len(entry)
            for idx, val in enumerate(entry):
                # get attr type
                attr_name = raw_data[1].names()[idx]
                if raw_data[1].types()[idx] == "nominal":
                    norm_val = normalise_nominal(val.decode("utf-8"), choices[attr_name])
                    if attr_name == "class":
                        class_batch.append(norm_val)
                    else:
                        norm_entry[idx] = norm_val
                if raw_data[1].types()[idx] == "numeric":
                    norm_val = normalise_numeric(val, minmax[attr_name])
                    if attr_name == "class":
                        class_batch.append(norm_val)
                    else:
                        norm_entry[idx] = norm_val
            batch.append(norm_entry)
        normalised_data.append(np.array(batch))
        norm_class_data.append(np.array(class_batch))
    return normalised_data, norm_class_data

def get_data(training_fns, testing_fns):
    training_raw = [loadarff(fn) for fn in training_fns]
    testing_raw = [loadarff(fn) for fn in testing_fns]

    # retrieve numerical variable min and max for value normalisation
    norm_minmax = get_normalising_minmax([training_raw, testing_raw])

    # normalise both datasets
    print("Normalising training set")
    training = normalise(training_raw, norm_minmax)
    print("Normalising testing set")
    testing = normalise(testing_raw, norm_minmax)

    return training, testing
