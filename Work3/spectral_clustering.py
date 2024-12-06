import os
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
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


def spectral_clustering_fit(X, n_clusters, affinity='nearest_neighbors', gamma=None):

    # clustering_model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=42, n_neighbors=10 if affinity == 'nearest_neighbors' else None, gamma=gamma if affinity == 'rbf' else None)
    if affinity == 'rbf':
        clustering_model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=42,
            gamma=gamma
        )
    else:
        clustering_model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=42,
            n_neighbors=10
        )
    cluster_labels = clustering_model.fit_predict(X)
    return cluster_labels


def explore_k_spectral(input, true_labels, max_k=10, gamma_values=None):
    results = {}
    affinities = ['nearest_neighbors', 'rbf'] 
    gamma_values = gamma_values or [0.1, 1, 10]

    for k in range(2, max_k):
        for affinity in affinities:
            if affinity == 'rbf':
                for gamma in gamma_values:
                    print(f"Running Spectral Clustering with k = {k} and affinity = {affinity}, gamma = {gamma} ")
                    res_name =  f"sc_k{k}_{affinity}_gamma{gamma}"
                
             # evaluation metrics
                    sli_scores = silhouette_score(input, labels)
                    db_score = davies_bouldin_score(input, labels)
                    ari_score = adjusted_rand_score(true_labels.squeeze(), labels)
                    hmv_score = homogeneity_completeness_v_measure(true_labels.squeeze(), labels)

                    results[res_name] = {
                        'k': k,
                        'affinity': affinity,
                        'gamma': gamma,
                        "silhouette": sli_scores,
                        "davies_bouldin": db_score,
                        "adjusted_rand_score": ari_score,
                        "homogeneity_completeness_v_measure": hmv_score,
                    }
                    print(np.array(sli_scores).mean(), np.array(db_score).mean())
            else:
                print(f"Running Spectral Clustering with k = {k} and affinity = {affinity}")
                res_name = f"sc_k{k}_{affinity}"
                
                labels = spectral_clustering_fit(input, n_clusters=k, affinity=affinity, gamma=None)
                
                # evaluation metrics
            sli_scores = silhouette_score(input, labels)
            db_score = davies_bouldin_score(input, labels)
            ari_score = adjusted_rand_score(true_labels.squeeze(), labels)
            hmv_score = homogeneity_completeness_v_measure(true_labels.squeeze(), labels)

            results[res_name] = {
                    'k': k,
                    'affinity': affinity,
                    "silhouette": sli_scores,
                    "davies_bouldin": db_score,
                    "adjusted_rand_score": ari_score,
                    "homogeneity_completeness_v_measure": hmv_score,
            }
            print(np.array(sli_scores).mean(), np.array(db_score).mean())
    return results

if __name__ == "__main__":
    data_dir = curr_dir / "datasets"

    cache_dir = curr_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = curr_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

  
    datasets = ["cmc", "sick", "mushroom"]
    for dataset_name in datasets:
        print("For dataset", dataset_name)
        input, true_labels = load_data(data_dir, dataset_name, cache=False, cache_dir=cache_dir)

        spectral_results = explore_k_spectral(input, true_labels)
        pd.DataFrame(spectral_results).T.to_csv(output_dir / f"{dataset_name}_spectral_results.csv")