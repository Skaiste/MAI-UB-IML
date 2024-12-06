import os
import sys
import pathlib
import time
import numpy as np
import pandas as pd
from scipy.stats import anderson
from sklearn.decomposition import PCA

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


# Add the current script's directory to sys.path
try:
    curr_dir = pathlib.Path(__file__).parent.resolve()
except NameError:
    curr_dir = pathlib.Path(os.getcwd()).resolve()
    if curr_dir.name != "Work3":
        curr_dir = curr_dir / "Work3"
sys.path.append(str(curr_dir))

from K_Means import *

# %%

def initialise_clusters(data, k, distance_fn=minkowski_distance):
    return [data] if k == 1 else kmeans_fit(input, k, distance_fn)

def split_cluster(cluster, max_k_iter=100, distance_fn=minkowski_distance):
    # splits cluster randomly
    old_centroid = get_cluster_centroid(cluster)
    random_centroid = cluster.loc[np.random.choice(cluster.shape[0])]
    new_centroid = old_centroid - (random_centroid - old_centroid)
    centroids = np.array([random_centroid, new_centroid])

    print("run kmeans on new centroids to split the cluster")
    return kmeans_fit(cluster, 2, distance_fn, centroids=centroids, max_iter=max_k_iter)

def gaussian_check(data, strictness=4):
    statistic = anderson(data, "norm")
    if statistic[0] <= statistic[1][strictness]:
        return True
    else:
        return False

#%%
def split_with_pca(cluster):
    pca = PCA(n_components=2)
    split_cluster_data = pca.fit_transform(cluster)
    split_mask = split_cluster_data[:, 0] > 0
    return [cluster[split_mask], cluster[~split_mask]]

def gmeans_fit(data, k_min=1, k_max=20, max_k_iter=100, distance_fn=minkowski_distance):
    k = k_min
    clusters = initialise_clusters(data, k, distance_fn)
    while k < k_max:
        print("Current k =", k)
        k_old = k
        _clusters = []

        with ThreadPoolExecutor() as executor:
            split_futures = {executor.submit(split_with_pca, cl): c_idx for c_idx, cl in enumerate(clusters) if cl.shape[0] > 1}
            for future in concurrent.futures.as_completed(split_futures):
                c_idx = split_futures[future]
                try:
                    new_clusters = future.result()
                except Exception as exc:
                    print(f'Cluster splitting generated an exception for cluster {c_idx}: {exc}')
                    continue

                # print(f"calculate centroids of the split clusters for cluster {c_idx}")
                new_centroids = [get_cluster_centroid(c) for c in new_clusters]

                # project clusters along connection axis
                # print(f"projecting clusters along the connection axis for cluster {c_idx}")
                v = new_centroids[1] - new_centroids[0]
                projected = np.dot(clusters[c_idx], v) / np.linalg.norm(v)
                # print(f"gaussian check for cluster {c_idx}")
                accept_null_hypothesis = gaussian_check(projected)

                if not accept_null_hypothesis:
                    # if the hypothesis is rejected -> replace the current cluster
                    # with the split clusters
                    _clusters += new_clusters
                    k += 1
                    print(f"adding split cluster {c_idx} into clusters")
                else:
                    _clusters.append(clusters[c_idx])

        # if k doesn't change -> g-means converges
        if k_old == k:
            clusters = _clusters
            break

        print("re-evaluating clusters based on the new centroids")
        clusters = kmeans_fit(data, k, distance_fn, centroids=[get_cluster_centroid(c) for c in _clusters], max_iter=max_k_iter)

    return clusters

# %%
def explore(input, true_labels, seeds=(42,)):
    results = {}
    distances = {
        "L1": lambda x, y: minkowski_distance(x, y, r=1),
        "L2": lambda x, y: minkowski_distance(x, y, r=2),
        "cosine": cosine_distance,
    }
    for distance_name, distance_fn in distances.items():
        # for idx, seed in enumerate(seeds):
        #     random.seed(seed)
        print("Running gmeans with distance =", distance_name)#, "seed =", seed, "iter =", idx)
        res_name = f"{distance_name}"#_seed{seed}"
        input_clusters = gmeans_fit(input, distance_fn=distance_fn)

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
            'k': len(input_clusters),
            'distance': distance_name,
            # 'seed': seed,
            "silhouette": sli_scores,
            "davies_bouldin": db_score,
            "adjusted_rand_score": ari_score,
            "homogeneity_completeness_v_measure": hmv_score,
            "sum_of_squared_error": sse_score
        }
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

        results = explore(input, true_labels, seeds=seeds)
        pd.DataFrame(results).T.to_csv(output_dir / f"{dataset_name}_gmeans_results.csv")
