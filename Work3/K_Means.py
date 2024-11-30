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

def load_data(data_dir, dataset_name, cache, cache_dir):
    dataset = data_dir / f"{dataset_name}.arff"
    if not dataset.is_file():
        raise Exception(f"Dataset {dataset} could not be found.")

    print("Loading data")
    normalise_nominal = True if dataset_name != "cmc" else False
    return get_data(dataset, cache_dir=cache_dir, cache=cache, normalise_nominal=normalise_nominal)


# %%

def get_cluster_centroid(cluster):
    return np.mean(cluster, axis = 0)

def minkowski_distance(x, y, r=2):
    x = np.expand_dims(x, axis=0)
    return ((abs(x-y)**r).sum(axis = 1))**(1/r)

def cosine_distance(x, y):
    dot_product = np.dot(y, x)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y, axis=1)
    cosine_similarity = dot_product / (norm_x * norm_y)
    return 1 - cosine_similarity


def kmeans_fit(X, k, distance_fn=minkowski_distance, max_iter=1000, centroids=None):
    # Firstly, randomly initialise centroids if they're not provided
    if centroids is None:
        centroids = X.sample(n=k).to_numpy()

    # Loop until convergence
    converged = False
    i = 0
    while not converged or i < max_iter:
        print(f"K-Means Iteration {i}", end="\r")
        # Assign each point to the "closest" centroid
        cluster_indexes = [[] for _ in range(k)]
        for idx, x in X.iterrows():
            distances = distance_fn(x, pd.DataFrame(centroids))
            cluster_idx = np.argmin(distances)
            cluster_indexes[cluster_idx].append(idx)

        clusters = [X.loc[c] for c in cluster_indexes]
        new_centroids = [get_cluster_centroid(c) for c in clusters]
        converged = np.array_equal(centroids, new_centroids)
        centroids = new_centroids
        if converged or i == max_iter:
            return [pd.DataFrame(c) for c in clusters]
        i += 1

# %%

def silhouette_score(clusters, distance_fn=minkowski_distance):
    scores = []
    for ci in clusters:
        scores.append([])

        if len(ci) == 1:
            scores[-1].append(0)
            continue

        for c_idx in ci.index:
            ci_without_c = ci.drop(index=c_idx)
            c = ci.loc[c_idx]
            distances = distance_fn(c, ci_without_c)
            a = np.mean(distances)
            b = min(distances)
            scores[-1].append((b - a) / max(a, b))

    return [np.array(s).mean() for s in scores]

def sum_of_squared_distances_score(clusters, distance_fn=minkowski_distance):
    scores = []
    for ci in clusters:
        centroid = get_cluster_centroid(ci)
        distances = distance_fn(centroid, ci)
        scores.append(np.sum(distances**2))

    return scores

# %%

def explore_k_and_seed(input, threshold=0.5, max_iter=100, init_seed=42, distance_fn=minkowski_distance):
    best_for_k = {}
    for k in range(2,11):
        best = None
        iterations = max_iter
        print("Running kmeans with k =", k)
        random.seed(init_seed)
        while iterations > 0:
            seed = random.randint(1, 100000000000)
            random.seed(seed)
            input_clusters = kmeans_fit(input, k, distance_fn)
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

            print(max_iter - iterations, best['seed'], np.array(best['scores']).mean(), seed, np.array(scores).mean())

            iterations -= 1
        best_for_k[k] = best
    return best_for_k

if __name__ == "__main__":
    data_dir = curr_dir / "datasets"
    dataset_name = "mushroom"

    cache_dir = curr_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    input, output = load_data(data_dir, dataset_name, cache=False, cache_dir=cache_dir)

    best_clusters = explore_k_and_seed(input)
