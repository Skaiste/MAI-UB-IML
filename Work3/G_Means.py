import os
import sys
import pathlib
import time
from scipy.stats import anderson

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

def gmeans_fit(data, k_min=1, k_max=10, max_k_iter=100, distance_fn=minkowski_distance):
    k = k_min
    clusters = initialise_clusters(data, k, distance_fn)
    while k < k_max:
        print("Current k =", k)
        k_old = k
        _clusters = []
        for c_idx, cl in enumerate(clusters):
            if cl.shape[0] <= 1:
                continue    # skip if there aren't enough data in a cluster

            # split the cluster into two
            print("splitting clusters")
            new_clusters = split_cluster(cl, max_k_iter, distance_fn)
            print("calculate centroids of the split clusters")
            new_centroids = [get_cluster_centroid(c) for c in new_clusters]

            # project clusters along connection axis
            print("projecting clusters along the connection axis")
            v = new_centroids[1] - new_centroids[0]
            projected = np.dot(cl, v) / np.linalg.norm(v)
            print("gaussian check")
            accept_null_hypothesis = gaussian_check(projected)

            if not accept_null_hypothesis:
                # if the hypothesis is rejected -> replace the current cluster
                # with the split clusters
                _clusters += new_clusters
                k += 1
                print(f"adding split cluster {c_idx} into clusters")
            else:
                _clusters.append(cl)

        # if k doesn't change -> g-means converges
        if k_old == k:
            clusters = _clusters
            break

        print("re-evaluating clusters based on the new centroids")
        clusters = kmeans_fit(data, k, distance_fn, centroids=[get_cluster_centroid(c) for c in _clusters], max_iter=max_k_iter)

    return clusters


if __name__ == "__main__":
    data_dir = curr_dir / "datasets"
    dataset_name = "cmc"

    cache_dir = curr_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    input, output = load_data(data_dir, dataset_name, cache=False, cache_dir=cache_dir)

    start = time.time()
    best_clusters = gmeans_fit(input, k_max=100)
    end = time.time()
    print(f"G-Means took {end - start} seconds for dataset {dataset_name}")
