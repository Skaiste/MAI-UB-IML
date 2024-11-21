import os
import sys
import math
import random
import pathlib
import numpy as np
import pandas as pd

try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
sys.path.append(str(curr_dir))

from data_parser import get_data

data_dir = curr_dir / "datasets"
dataset_name = "cmc"

cache_dir = curr_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

dataset = data_dir / f"{dataset_name}.arff"
if not dataset.is_file():
    raise Exception(f"Dataset {dataset} could not be found.")

print("Loading data")
normalise_nominal = True if dataset_name != "cmc" else False
train_input, train_output, test_input, test_output = get_data(dataset, cache_dir=cache_dir, cache=False, normalise_nominal=normalise_nominal)

# %%

def get_cluster_centroid(cluster):
    return np.mean(cluster, axis = 0)

def random_partition(X, k):
    assert k > 0, "k must be greater than 0"
    if k == 1: get_cluster_centroid(X)

    # randomise indexes
    rand_indexes = list(range(X.shape[0]))
    random.shuffle(rand_indexes)
    # split indexes into clusters
    cluster_size = math.ceil(X.shape[0] / k)
    clusters_idx = [rand_indexes[c * cluster_size:min((c+1) * cluster_size, X.shape[0])] for c in range(k)]
    # set up clusters based on the random index splitting
    clusters = [X.iloc[c] for c in clusters_idx]
    return [get_cluster_centroid(c) for c in clusters]

def euclidean_distance(x, y, weights):
    return np.sqrt(np.sum(weights * (x - y) ** 2, axis=1))

# %%
def kmeans_fit(X, k):
    # Firstly, randomly initialise centroids
    centroids = random_partition(X, k)

    # Loop until convergence
    converged = False
    while not converged:
        # Assign each point to the "closest" centroid
        cluster_indexes = [[] for _ in range(k)]
        for idx, x in X.iterrows():
            distances = euclidean_distance(x, pd.DataFrame(centroids), np.ones(X.shape[1]))
            cluster_idx = np.argmin(distances)
            cluster_indexes[cluster_idx].append(idx)

        clusters = [X.loc[c] for c in cluster_indexes]
        new_centroids = [get_cluster_centroid(c) for c in clusters]
        converged = True if np.array_equal(centroids, new_centroids) else False
        centroids = new_centroids
        if converged:
            return [pd.DataFrame(c) for c in clusters]

input_clusters = kmeans_fit(train_input, 3)

# %%
def determine_class(clusters, y):
    cluster_outputs = [y.loc[[i.name for _,i in c.iterrows()]] for c in clusters]
    cluster_outputs = [c.value_counts() for c in cluster_outputs]
    class_values = {cls:[c[cls] for c in cluster_outputs] for cls in cluster_outputs[0].keys()}
    cluster_classes = {counts.index(max(counts)):cls for cls, counts in class_values.items()}

    return list(cluster_classes.values())

output_clusters = determine_class(input_clusters, train_output)

clusters = [{
    'input':input_clusters[i],
    'output':output_clusters[i],
    'centroids': get_cluster_centroid(input_clusters[i])}
    for i in range(len(input_clusters))]

# %%
def kmeans_predict(clstr, x):
    inputs = pd.DataFrame([c['centroids'] for c in clstr])
    outputs = [c['output'] for c in clstr]
    distances = euclidean_distance(x, inputs, np.ones(x.shape[0]))

    return outputs[distances.idxmin()]

# %%
def kmeans_predict_all(clstr, X):
    outputs = []
    for idx, x in X.iterrows():
        outputs.append(kmeans_predict(clstr, x))
    return pd.Series(outputs)

predictions = kmeans_predict_all(clusters, test_input)
matches = test_output == predictions.reindex(test_output.index)
result_counts = matches.value_counts()
accuracy = result_counts[True] / matches.count() * 100
print(f"Accuracy: {accuracy:.2f}%")