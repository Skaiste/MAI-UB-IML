import os
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, homogeneity_completeness_v_measure
from statsmodels.stats.diagnostic import compare_j

try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()).resolve()
    if curr_dir.name != "Work3":
        curr_dir = curr_dir / "Work3"
sys.path.append(str(curr_dir))

from data_parser import get_data
from K_Means import minkowski_distance, cosine_distance

distances = {
    "L1 (Manhattan)": lambda x, y: minkowski_distance(x, y, r=1),
    "L2 (Euclidean)": lambda x, y: minkowski_distance(x, y, r=2),
    "cosine": cosine_distance
}

input_path = input("Input the path for the dataset (default ./datasets/cmc.arff): ") or (curr_dir/"datasets"/"cmc.arff")
input_path = pathlib.Path(input_path)
if not input_path.exists():
    raise FileNotFoundError(f"File {input_path} does not exist. Check the path to the dataset or use an absolute path. The current working directory is {os.getcwd()}.")
dataset_name = pathlib.Path(input_path).stem

normalise_nominal = True if "cmc" not in input_path.name else False
input_data, true_labels = get_data(input_path, cache_dir=input_path.parent, cache=False, normalise_nominal=normalise_nominal)

def get_choice(name, choices, default=None):
    if default is None:
        default = choices[0]
    while True:
        print(f"Select {name} (0 to exit):")
        for i, choice in enumerate(choices, start=1):
            print(f"{i}. {choice}")
        selected = int(input(f"Enter the number corresponding to the {name} (default {default}): ") or (choices.index(default)+1))
        if selected == 0:
            sys.exit("Exiting the program.")
        if selected <= len(choices):
            break
        print(f"Chosen {name} doesn't exist, try again.")
    return choices[selected - 1]

def get_choice_short(name, choices, choices_str, val_type, default=None):
    if default is None:
        default = choices[0]
    while True:
        selected = val_type(input(f"Enter the {name} {choices_str} (default {default}, 0 to exit): ") or default)
        if selected == 0:
            sys.exit("Exiting the program.")
        if selected in choices:
            break
        print(f"Chosen value {name} is incorrect, try again.")
    return selected

algorithms = ["optics", "spectral", "k-means", "k-means++", "g-means", "fuzzy"]
default_alg = "optics" if "mushroom" != dataset_name else "k-means++"
algorithm = get_choice("algorithm", algorithms, default_alg)

if algorithm == "optics":
    distance_names = ["L1 (Manhattan)", "L2 (Euclidean)", "cosine"]
    default_dist = "L2 (Euclidean)" if "cmc" not in input_path.name else "L1 (Manhattan)"
    distance_choice = get_choice("distance function", distance_names, default_dist)
    distance_function = distances[distance_choice]

    algorithms_optics = ["auto", "brute"]
    default_optics_alg = "brute" if "cmc" not in input_path.name else "auto"
    algorithm_optics = get_choice("algorithm", algorithms_optics, default_optics_alg)

    min_samples = [15, 20, 25]
    default_min_samples = 25 if "cmc" not in input_path.name else 15
    min_sample = get_choice_short("number of minimum samples", min_samples, str(min_samples), int, default_min_samples)

    max_epsilons = [7, 9, 10, 15] if "mushroom" == dataset_name else [0.5, 1, 1.5, 3, 5, 7, 9]
    default_max_eps = 9 if "mushroom" == dataset_name else 1
    max_eps = get_choice_short("maximum epsilon value", max_epsilons, str(max_epsilons), float, default_max_eps)

    from optics import fit_optics
    print(f"Running OPTICS model with {distance_choice} distance function, {algorithm_optics} algorithm, min samples = {min_sample} and max eps = {max_eps}")
    model = fit_optics(input_data, min_sample, distance_function, algorithm_optics, max_eps)
    clusters = model.labels_
    noise = np.sum(clusters == -1)
    labels = true_labels[clusters != -1]
    input_  = input_data[clusters != -1]

    labels = labels.to_numpy()
    output = clusters[clusters != -1]

    sli_scores = silhouette_score(input_, output)
    db_score = davies_bouldin_score(input_, output)
    ari_score = adjusted_rand_score(labels, output)
    hmv_score = homogeneity_completeness_v_measure(labels, output)


elif algorithm == "spectral":
    cluster_range = list(range(2, 10))
    default_clusters = 2 if "sick" == dataset_name else 8
    num_clusters = get_choice_short("number of clusters", cluster_range, "from 2 to 9", int, default_clusters)

    affinities = ["rbf", "nearest neighbours"]
    affinity = get_choice("affinity", affinities, "rbf")

    gamma = None
    if affinity == "rbf":
        gamma_range = [0.1, 1, 10]
        gamma = get_choice_short("gamma value", gamma_range, str(gamma_range), float, 0.1)

    from spectral_clustering import spectral_clustering_fit
    print(f"Running spectral clustering with {num_clusters} clusters, {affinity} affinity" + (f" and gamma = {gamma}" if affinity == "rbf" else ""))
    labels = spectral_clustering_fit(input_data, n_clusters=num_clusters, affinity=affinity, gamma=gamma, eigen_solver='lobpcg')
    sli_scores = silhouette_score(input_data, labels)
    db_score = davies_bouldin_score(input_data, labels)
    ari_score = adjusted_rand_score(true_labels.squeeze(), labels)
    hmv_score = homogeneity_completeness_v_measure(true_labels.squeeze(), labels)


elif algorithm == "k-means":
    seed_choices = list(range(20, 70, 5))
    default_seed = 40 if "mushroom" == dataset_name else 60
    seed = get_choice_short("seed value", seed_choices, str(seed_choices), int, default_seed)

    k_range = list(range(2, 10))
    default_k = 5 if "mushroom" == dataset_name else 4
    k_value = get_choice_short("k value", k_range, "from 2 to 9", int, default_k)

    distance_names = ["L1 (Manhattan)", "L2 (Euclidean)", "cosine"]
    default_dist = "cosine" if "mushroom" == dataset_name else ("L1 (Manhattan)" if "sick" == dataset_name else "L2 (Euclidean)")
    distance_choice = get_choice("distance function", distance_names, default_dist)
    distance_function = distances[distance_choice]

    from K_Means import kmeans_fit
    print(f"Running K-Means algorithm with {k_value} clusters, {distance_choice} distance function and seed = {seed}")
    input_clusters = kmeans_fit(input_data, k_value, distance_function, seed)

    labels = pd.DataFrame(columns=['cluster'])
    for i, cl in enumerate(input_clusters):
        l = pd.DataFrame({'cluster': [i] * cl.shape[0]}, index=cl.index)
        labels = pd.concat([labels, l]).sort_index()

    sli_scores = silhouette_score(input_data, labels.squeeze())
    db_score = davies_bouldin_score(input_data, labels.squeeze())
    ari_score = adjusted_rand_score(true_labels.squeeze(), labels.squeeze())
    hmv_score = homogeneity_completeness_v_measure(true_labels.squeeze(), labels.squeeze())


elif algorithm == "k-means++":
    k_range = list(range(2, 10))
    default_k = 3 if "mushroom" == dataset_name else (8 if "sick" == dataset_name else 6)
    k_value = get_choice_short("k value", k_range, "from 2 to 9", int, default_k)

    distance_names = ["L1 (Manhattan)", "L2 (Euclidean)", "cosine"]
    default_dist = "cosine" if "mushroom" == dataset_name else ("L1 (Manhattan)" if "sick" == dataset_name else "L2 (Euclidean)")
    distance_choice = get_choice("distance function", distance_names, default_dist)
    distance_function = distances[distance_choice]

    from kmeans_pp import kmeans_plus_plus_fit
    print(f"Running K-Means++ algorithm with {k_value} clusters and {distance_choice} distance function")
    input_clusters = kmeans_plus_plus_fit(input_data, k_value, distance_function)

    labels = pd.DataFrame(columns=['cluster'])
    for i, cl in enumerate(input_clusters):
        l = pd.DataFrame({'cluster': [i] * cl.shape[0]}, index=cl.index)
        labels = pd.concat([labels, l]).sort_index()

    sli_scores = silhouette_score(input_data, labels.squeeze())
    db_score = davies_bouldin_score(input_data, labels.squeeze())
    ari_score = adjusted_rand_score(true_labels.squeeze(), labels.squeeze())
    hmv_score = homogeneity_completeness_v_measure(true_labels.squeeze(), labels.squeeze())


elif algorithm == "g-means":
    distance_names = ["L1 (Manhattan)", "L2 (Euclidean)", "cosine"]
    default_dist = "L1 (Manhattan)" if "mushroom" == dataset_name else ("cosine" if "sick" == dataset_name else "L2 (Euclidean)")
    distance_choice = get_choice("distance function", distance_names, default_dist)
    distance_function = distances[distance_choice]

    from G_Means import gmeans_fit
    print(f"Running G-Means algorithm with {distance_choice} distance function")
    input_clusters = gmeans_fit(input_data, distance_fn=distance_function)

    labels = pd.DataFrame(columns=['cluster'])
    for i, cl in enumerate(input_clusters):
        l = pd.DataFrame({'cluster': [i] * cl.shape[0]}, index=cl.index)
        labels = pd.concat([labels, l]).sort_index()

    sli_scores = silhouette_score(input_data, labels.squeeze())
    db_score = davies_bouldin_score(input_data, labels.squeeze())
    ari_score = adjusted_rand_score(true_labels.squeeze(), labels.squeeze())
    hmv_score = homogeneity_completeness_v_measure(true_labels.squeeze(), labels.squeeze())


elif algorithm == "fuzzy":
    cluster_range = list(range(2, 10))
    num_clusters = get_choice_short("number of clusters", cluster_range, "from 2 to 9", int, 2)
    fuzzyness_range = [1.5, 2.0, 2.5]
    default_fuzzyness = 2.5 if "mushroom" == dataset_name else 1.5
    fuzzyness_level = get_choice_short("fuzzyness level", fuzzyness_range, str(fuzzyness_range), float, default_fuzzyness)

    from fuzzy_clustering import fuzzy_c_means

    centers, membership_matrix = fuzzy_c_means(input_data, n_clusters=num_clusters, m=fuzzyness_level)

    # clustering crisp by assigning each point to the cluster with the highest membership
    labels = np.argmax(membership_matrix, axis=1)

    # external metrics evaluation
    ari_score = adjusted_rand_score(true_labels, labels)
    hmv_score = homogeneity_completeness_v_measure(true_labels, labels)

    # evaluation metrics
    sli_scores = silhouette_score(input_data, labels)
    db_score = davies_bouldin_score(input_data, labels)

else:
    sys.exit("Exiting the program.")

print(f"Silhouette score: {sli_scores}")
print(f"Davies-Bouldin score: {db_score}")
print(f"Adjusted Rand Index (ARI) score: {ari_score}")
print(f"Homogeneity-Completeness-V-Measure score: {hmv_score}")
