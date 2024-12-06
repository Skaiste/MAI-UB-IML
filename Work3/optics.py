import os
import sys
import pandas as pd
import pathlib
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score, homogeneity_completeness_v_measure

try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
sys.path.append(str(curr_dir))

from K_Means import load_data


def fit_optics(data, min_sample, distance, algorithm_,max_eps_):
    model = OPTICS(min_samples = min_sample,max_eps = max_eps_,metric=distance, algorithm=algorithm_)
    model.fit(data)
    return model


def main(dataset_name, input, labels_):


    distances = ['euclidean', 'cosine', 'l1']
    algorithms = ['auto', 'brute']
    min_samples_list = [15,20,25]
    if dataset_name == "mushroom":
        max_eps_list = [7, 9, 10, 15]
    else:
        max_eps_list = [0.5, 1, 1.5, 3, 5, 7, 9]
    for max_eps in max_eps_list:
        for min_sample in min_samples_list:
            for distance in distances:
                for algorithm in algorithms:
                    print(f" OPTICS for {dataset_name} with {distance} and {algorithm} and  min samples = {min_sample} and max eps = {max_eps}")
                    model = fit_optics(input, min_sample, distance, algorithm,max_eps)
                    clusters = model.labels_
                    noise = np.sum(clusters == -1)
                    labels = labels_[clusters != -1]
                    input_  = input[clusters != -1]

                    labels = labels.to_numpy()

                    output = clusters[clusters != -1]

                    sli_scores = sk_silhouette_score(input_, output)
                    db_score = davies_bouldin_score(input_, output)
                    ari_score = adjusted_rand_score(labels, output)
                    hmv_score = homogeneity_completeness_v_measure(labels, output)


                    results[distance+"_"+algorithm+"_"+str(min_sample)+"_"+str(max_eps)] = {
                            "silhouette": sli_scores,
                            "davies_bouldin": db_score,
                            "adjusted_rand_score": ari_score,
                            "homogeneity_completeness_v_measure": hmv_score,
                            'noise':noise
                    }
    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.rename(columns={'index': 'Algorithm'}, inplace=True)
    df.to_csv("results/" + dataset_name + '_optics_results.csv', index=False,mode='w')
    return results


if __name__ == "__main__":
    data_dir = curr_dir / "datasets"

    cache_dir = curr_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = curr_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [ "mushroom"] #"sick", "cmc",
    results = {}
    for dataset_name in datasets:
        print("For dataset", dataset_name)
        input, labels_ = load_data(data_dir, dataset_name, cache=False, cache_dir=cache_dir)

        main(dataset_name, input, labels_)
