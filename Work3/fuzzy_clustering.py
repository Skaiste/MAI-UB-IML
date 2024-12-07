import os
import sys
import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, homogeneity_completeness_v_measure

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

def matrix(n_samples, n_clusters):
    membership_matrix = np.random.rand(n_samples, n_clusters)
    membership_matrix = membership_matrix / np.sum(membership_matrix, axis=1, keepdims=True)  # normalize rows
    return membership_matrix

def cluster_centers(X, U, m):
    U_m = U ** m  
    cluster_centers = (U_m.T @ X) / np.sum(U_m.T, axis=1, keepdims=True)
    return cluster_centers

def update_matrix(X, centers, m):
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)  # distances to cluster centers
    distances = np.fmax(distances, np.finfo(np.float64).eps)  # avoid division by zero
    reciprocal_distances = 1.0 / distances
    U_new = reciprocal_distances / np.sum(reciprocal_distances, axis=1, keepdims=True)  # normalize
    return U_new

def fuzzy_c_means(X, n_clusters=3, m=2.0, max_iter=150, error=1e-5):
    # convert input to numpy array if it's a dataframe
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    n_samples, n_features = X.shape
    U = matrix(n_samples, n_clusters)  
    centers = cluster_centers(X, U, m)  

    for iteration in range(max_iter):
        U_new = update_matrix(X, centers, m)
        new_centers = cluster_centers(X, U_new, m)

        if np.linalg.norm(new_centers - centers) < error:  # convergence based on centroids
            return new_centers, U_new  

        centers = new_centers
        U = U_new

    return centers, U

def explore_k_fuzzy(input_data, true_labels, max_k=10, m=2.0):
    results = {}

    # convert input to NumPy array if necessary
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.to_numpy()

    for k in range(2, max_k):
        print(f"Running Fuzzy Clustering with k = {k}")
        res_name = f"fuzzy_k{k}"

        centers, membership_matrix = fuzzy_c_means(input_data, n_clusters=k, m=m)

        # Make the clustering crisp by assigning each point to the cluster with the highest membership
        labels = np.argmax(membership_matrix, axis=1)

        # External metrics evaluation
        ari_score = adjusted_rand_score(true_labels, labels)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, labels)

        # evaluation metrics
        sli_scores = silhouette_score(input_data, labels)
        db_score = davies_bouldin_score(input_data, labels)

        results[res_name] = {
            'k': k,
            "silhouette": sli_scores,
            "davies_bouldin": db_score,
            "adjusted_rand_score": ari_score,
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_measure": v_measure
        }
        print(f"Completed: {res_name}, Silhouette: {sli_scores}, Davies-Bouldin: {db_score}, ARI: {ari_score}, Homogeneity: {homogeneity}, Completeness: {completeness}, V-Measure: {v_measure}")
    
    return results

if __name__ == "__main__":
    data_dir = curr_dir / "datasets"

    cache_dir = curr_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = curr_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["cmc", "sick", "mushroom"]
    for dataset_name in datasets:
        print(f"For dataset {dataset_name}")
        input_data, true_labels = load_data(data_dir, dataset_name, cache=False, cache_dir=cache_dir)

        # clustering for different k
        fuzzy_results = explore_k_fuzzy(input_data, true_labels)
        pd.DataFrame(fuzzy_results).T.to_csv(output_dir / f"{dataset_name}_fuzzy_results.csv")
