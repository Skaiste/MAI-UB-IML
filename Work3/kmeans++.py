import os
import sys
import pathlib
import numpy as np
import pandas as pd
import argparse
import random
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score, homogeneity_completeness_v_measure
from K_Means import load_data,  kmeans_fit, minkowski_distance, cosine_distance, sum_of_squared_error
try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
sys.path.append(str(curr_dir))
def create_centroids(data,k,distance):
    indices = np.array([np.random.choice(len(data))])
    centroids = pd.DataFrame(data.iloc[indices])
    while len(indices)<k:

        distances = np.array([distance(row, centroids).min()  for index, row  in data.iterrows()])
        probabilities = distances**2/(np.sum(distances**2))
        index = np.random.choice(data.index,p = probabilities)

        if index not in indices:
            indices = np.append(indices, index)

            centroids = pd.DataFrame(data.iloc[indices])

    return np.array(centroids)

def kmeans_plus_plus_fit(X, k,distance,max_iter = 100):

    centroids = create_centroids(X, k,distance)
    return kmeans_fit(X, k,distance,max_iter, centroids)

def explore_k(input,labels, distance,csv_file_name, max_k=10):
    seeds = np.arange(20,70, 5)
    results = {}
    for k in range(2,max_k):
        best_error = np.inf
        best_seed = seeds[0]
        for seed in seeds:
            random.seed(seed)

            clusters = kmeans_plus_plus_fit(input, k, distance)
            output = pd.DataFrame(columns=['cluster'])
            for i, cl in enumerate(clusters):
                l = pd.DataFrame({'cluster': [i] * cl.shape[0]}, index=cl.index)
                output = pd.concat([output, l]).sort_index()

            sli_scores = sk_silhouette_score(input, output.squeeze())
            db_score = davies_bouldin_score(input, output.squeeze())
            ari_score = adjusted_rand_score(labels.squeeze(), output.squeeze())
            hmv_score = homogeneity_completeness_v_measure(labels.squeeze(), output.squeeze())
            sse_score = np.array([sum_of_squared_error(cl) for cl in clusters]).mean()
            if sse_score < best_error:
                best_error = sse_score
                best_seed = seed
                results[k] = {
                'seed': best_seed,
                 "silhouette": sli_scores,
                 "davies_bouldin": db_score,
                 "adjusted_rand_score": ari_score,
                 "homogeneity_completeness_v_measure": hmv_score,
                 "sum_of_squared_error": best_error,
                 }

    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.rename(columns={'index': 'k'}, inplace=True)
    df.to_csv("results/"+csv_file_name+'_kmeans++_results.csv', index=False)
    return results
if __name__ == "__main__":

    data_dir = curr_dir / "datasets"

    cache_dir = curr_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = curr_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [ "sick", "mushroom"]
    for dataset_name in datasets:
        print("For dataset", dataset_name)
        input, output = load_data(data_dir, dataset_name, cache=False, cache_dir=cache_dir)
        # the best distances based on kmeans
        if dataset_name == "cmc":
            distance = lambda x, y: minkowski_distance(x, y, 2)
        elif dataset_name == "sick":
            distance = lambda x, y: minkowski_distance(x, y, 1)
        else:
            distance = cosine_distance
        explore_k(input,output, distance, dataset_name)




