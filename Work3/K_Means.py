import os
import sys
import random
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score, homogeneity_completeness_v_measure

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

def sum_of_squared_error(cluster):
    # Calculate the centroid of the cluster
    centroid = get_cluster_centroid(cluster)
    # Compute the squared differences from the centroid
    squared_diffs = ((cluster - centroid) ** 2).sum(axis=1)
    # Calculate SSE by summing the squared differences
    sse = np.sum(squared_diffs)

    return sse

# %%

def explore_k_and_seed(input, true_labels, seeds=(42,), max_k=10):
    results = {}
    distances = {
        "L1": lambda x, y: minkowski_distance(x, y, r=1),
        "L2": lambda x, y: minkowski_distance(x, y, r=2),
        "cosine": cosine_distance,
    }
    for k in range(2, max_k):
        for distance_name, distance_fn in distances.items():
            print("Running kmeans with k =", k, "and distance =", distance_name)
            for idx, seed in enumerate(seeds):
                random.seed(seed)
                res_name = f"k{k}_{distance_name}_seed{seed}"
                input_clusters = kmeans_fit(input, k, distance_fn)

                labels = pd.DataFrame(columns=['cluster'])
                for i, cl in enumerate(input_clusters):
                    l = pd.DataFrame({'cluster': [i] * cl.shape[0]}, index=cl.index)
                    labels = pd.concat([labels, l]).sort_index()

                sli_scores = sk_silhouette_score(input, labels.squeeze())
                db_score = davies_bouldin_score(input, labels.squeeze())
                ari_score = adjusted_rand_score(true_labels.squeeze(), labels.squeeze())
                hmv_score = homogeneity_completeness_v_measure(true_labels.squeeze(), labels.squeeze())
                sse_score = np.array([sum_of_squared_error(cl) for cl in input_clusters]).mean()

                results[res_name] = {
                    'k': k,
                    'distance': distance_name,
                    'seed': seed,
                    "silhouette": sli_scores,
                    "davies_bouldin": db_score,
                    "adjusted_rand_score": ari_score,
                    "homogeneity_completeness_v_measure": hmv_score,
                    "sum_of_squared_error": sse_score,
                }

                print(idx, seed, np.array(sli_scores).mean(), np.array(db_score).mean())
    return results

if __name__ == "__main__":
    data_dir = curr_dir / "datasets"

    cache_dir = curr_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = curr_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(20, 70, 5))

    datasets = ["cmc", "sick", "mushroom"]
    for dataset_name in datasets:
        print("For dataset", dataset_name)
        input, true_labels = load_data(data_dir, dataset_name, cache=False, cache_dir=cache_dir)

        results = explore_k_and_seed(input, true_labels, seeds=seeds)
        pd.DataFrame(results).T.to_csv(output_dir / f"{dataset_name}_kmeans_results.csv")
