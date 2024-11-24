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
dataset_name = "mushroom"

cache_dir = curr_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

dataset = data_dir / f"{dataset_name}.arff"
if not dataset.is_file():
    raise Exception(f"Dataset {dataset} could not be found.")

print("Loading data")
normalise_nominal = True if dataset_name != "cmc" else False
input, output = get_data(dataset, cache_dir=cache_dir, cache=False, normalise_nominal=normalise_nominal)

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


# %%

def silhouette_score(clusters):
    scores = []
    for ci in clusters:
        scores.append([])

        if len(ci) == 1:
            scores[-1].append(0)
            continue

        for c_idx in ci.index:
            ci_without_c = ci.drop(index=c_idx)
            c = ci.loc[c_idx]
            distances = euclidean_distance(c, ci_without_c, np.ones(c.shape[0]))
            a = np.mean(distances)
            b = min(distances)
            scores[-1].append((b - a) / max(a, b))

    return [np.array(s).mean() for s in scores]

# %%

# search for cluster combination with the best mean of the silhouette score across all clusters
def explore_seed(k=3, threshold=0.5):
    iterations = 100

    best = None
    while iterations > 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        input_clusters = kmeans_fit(input, k)
        scores = silhouette_score(input_clusters)
        if best is None or np.array(scores).mean() > np.array(best['scores']).mean():
            best = {
                'clusters' : input_clusters,
                'scores': scores,
                'seed': seed,
            }

        if np.array(scores).mean() > threshold:
            break

        # if iterations % 10 == 0:
        print(100 - iterations, seed, scores)

        iterations -= 1
    return best

# random.seed(226)
random.seed(42)
best_clusters = explore_seed()

# for k = 3, best clustering seed = 7931

# %%

def explore_k_and_seed(threshold=0.5, max_iter=2):
    best = None
    for k in range(2, 11):
        iterations = max_iter
        print("Running kmeans with k =", k)
        while iterations > 0:
            seed = random.randint(1, 10000)
            random.seed(seed)
            input_clusters = kmeans_fit(input, k)
            scores = silhouette_score(input_clusters)
            if best is None or np.array(scores).mean() > np.array(best['scores']).mean():
                best = {
                    'clusters' : input_clusters,
                    'scores': scores,
                    'seed': seed,
                    'k': k
                }

            if np.array(scores).mean() > threshold:
                break

            # if iterations % 10 == 0:
            print(max_iter - iterations, seed, scores)

            iterations -= 1
    return best

# random.seed(7931)
random.seed(42)
best_clusters = explore_k_and_seed()

