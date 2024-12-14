import os
import sys
import random
import pathlib
import argparse
import matplotlib.pyplot as plt
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, homogeneity_completeness_v_measure
import umap

try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work4"
sys.path.append(str(curr_dir))

from data_parser import get_data
from K_Means import minkowski_distance, cosine_distance
from kmeans_pp import kmeans_plus_plus_fit
from optics import fit_optics


# %%
def pca_fit(X, n_components=None, print_info=True):
    # Standardisation
    # Removing invalid data to allow eigenvector extraction
    X = X.fillna(X.mean())
    z_score = (X - X.mean()) / X.std().replace(0, np.nan)
    z_score = z_score.dropna(axis=1, how='any')

    # covariance matrix
    covar = z_score.cov()

    if np.isnan(covar).any().any():
        raise Exception("NaN values found in covar")
    if np.isinf(covar).any().any():
        raise Exception("Inf values found in covar")

    # eigenvectors and eigenvalues
    eigen_vals, eigen_vecs = np.linalg.eig(covar)
    eigen_vecs = eigen_vecs.real
    eigen_vals = eigen_vals.real
    if print_info:
        print("Eigenvalues: ", eigen_vals)
        print("Eigenvectors: \n", eigen_vecs)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigen_vals)[::-1]
    eigen_vecs = eigen_vecs[:, sorted_indices]

    # feature vectors
    n_components = n_components if n_components is not None else len(X.columns)
    top_eigen_vecs = eigen_vecs[:, :n_components]

    # projecting the data for dimensionality reduction
    transformed = z_score.dot(top_eigen_vecs)
    return transformed, top_eigen_vecs

# %%
def plot_features(X, y, title, columns=None, result_dir=None, show=True):
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if columns is not None:
        columns = X.columns[:2]

    plt.figure(figsize=(10, 10))

    classes = labels.unique()
    for cl in classes:
        cl_idx = y[y == cl].index
        plt.scatter(X[columns[0]].loc[cl_idx], X[columns[1]].loc[cl_idx], label=f"Class {cl}")
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title(title)
    plt.legend()
    if show:
        plt.show()
    else:
        if result_dir is None:
            result_dir = curr_dir / "results"
        filename = title.lower().replace(" ", "_") + ".png"
        plt.savefig(result_dir / filename)

def plot_eigenvectors(eigenvecs, title, result_dir=None, show=True):
    plt.figure(figsize=(10, 10))
    plt.scatter(eigenvecs[:, 0], eigenvecs[:, 1])
    plt.xlabel("First Eigenvector")
    plt.ylabel("Second Eigenvector")
    plt.title(title)
    if show:
        plt.show()
    else:
        if result_dir is None:
            result_dir = curr_dir / "results"
        filename = title.lower().replace(" ", "_") + ".png"
        plt.savefig(result_dir / filename)

def plot_clusters(clusters, title, result_dir=None, show=True):
    plt.figure(figsize=(10, 10))
    for cluster in clusters:
        plt.scatter(cluster[0], cluster[1])
    plt.title(title)
    if show:
        plt.show()
    else:
        if result_dir is None:
            result_dir = curr_dir / "results"
        filename = title.lower().replace(" ", "_") + ".png"
        plt.savefig(result_dir / filename)


# %%


def parse_arguments():
    parser = argparse.ArgumentParser(description="PCA Arguments Parser")
    parser.add_argument(
        "-d", "--dataset_path",
        type=pathlib.Path,
        default=curr_dir / "datasets",
        help="Path to the dataset file (type: pathlib.Path), if it is a directory will use all dataset files",
    )
    parser.add_argument(
        "-n", "--n_components",
        type=int,
        default=4,
        help="Number of principal components for PCA (default: 4)",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Flag indicating whether to display plots (default: False)",
    )

    args = parser.parse_args()
    # Check if the dataset path exists
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"The provided dataset path `{args.dataset_path}` does not exist.")

    # If it's a directory, get all '.arff' files
    if args.dataset_path.is_dir():
        arff_files = list(args.dataset_path.glob("*.arff"))
        if not arff_files:
            raise FileNotFoundError(f"No '.arff' files found in the directory `{args.dataset_path}`.")
        args.dataset_path = arff_files  # Replace with list of '.arff' files
    else:
        # If it's a file, check if it has '.arff' suffix
        if not args.dataset_path.suffix == ".arff":
            raise ValueError(f"The provided dataset path `{args.dataset_path}` is not an '.arff' file.")
        args.dataset_path = [args.dataset_path]

    return args
    

# %%
if __name__ == "__main__":
    args = parse_arguments()
    # args = lambda : None
    # args.dataset_path = list((curr_dir / "datasets").glob("*.arff"))
    # args.n_components = 4
    # args.show_plots = True

    columns_to_plot = {
        "cmc": ['wage', 'weducation'],
        "sick": ['age', 'TSH'],
        "mushroom": ['cap-shape_b', 'cap-shape_c']
    }
    k_values = {"cmc": 6, "sick": 8, "mushroom": 3}
    distance = {
        "cmc": lambda x, y: minkowski_distance(x, y, r=2),
        "sick": lambda x, y: minkowski_distance(x, y, r=1),
        "mushroom": cosine_distance
    }
    optics_params = {
        "cmc": {"distance": "euclidean", "algorithm": "brute", "min_samples": 25, "eps": 1},
        "sick": {"distance": "l1", "algorithm": "auto", "min_samples": 15, "eps": 1},
        "mushroom": {"distance": "l1", "algorithm": "auto", "min_samples": 15, "eps": 9}
    }
    previous_results = {
        "cmc": {
            "kmeans++": {"silhouette": 0.297408, "davies_bouldin": 1.225529, "adjusted_rand_score": 0.018344, "v_measure": 0.028068},
            "optics": {"silhouette": 0.603298, "davies_bouldin": 0.560854, "adjusted_rand_score": 0.08041, "v_measure": 0.078456},
        },
        "sick": {
            "kmeans++": {"silhouette": 0.326836, "davies_bouldin": 1.233292, "adjusted_rand_score": 0.005036, "v_measure": 0.031435},
            "optics": {"silhouette": 0.834781, "davies_bouldin": 0.202779, "adjusted_rand_score": -0.001405, "v_measure": 0.029909},
        },
        "mushroom": {
            "kmeans++": {"silhouette": 0.208595, "davies_bouldin": 1.675602, "adjusted_rand_score": 0.4819, "v_measure": 0.484288},
            "optics": {"silhouette": 0.295235, "davies_bouldin": 1.226707, "adjusted_rand_score": 0.276198, "v_measure": 0.460663},
        }
    }
    for fn in args.dataset_path:
        dataset_name = fn.stem
        plot_result_dir = curr_dir / "results" / dataset_name
        plot_result_dir.mkdir(parents=True, exist_ok=True)
        print("For dataset", dataset_name)
        normalise_nominal = True if dataset_name != "cmc" else False
        input, labels = get_data(fn, cache_dir=None, cache=False, normalise_nominal=normalise_nominal)

        plot_features(input, labels,
                      title=f'Initial Input of {dataset_name} dataset',
                      columns=columns_to_plot[dataset_name],
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        print("Running PCA on input data...")
        results, eigenvectors = pca_fit(input, n_components=args.n_components)
        plot_eigenvectors(eigenvectors,
                          title=f'Eigenvectors of {dataset_name} dataset',
                          show=args.show_plots,
                          result_dir=plot_result_dir)
        plot_features(results, labels,
                      title=f'PCA of {dataset_name} dataset',
                      columns=columns_to_plot[dataset_name],
                      show=args.show_plots,
                      result_dir=plot_result_dir)

        # compare with sklearn PCA results
        pca = PCA(n_components=args.n_components)
        pca_results = pca.fit_transform(input)
        inc_pca = IncrementalPCA(n_components=args.n_components)
        inc_pca_results = inc_pca.fit_transform(input)
        plot_features(pca_results, labels,
                      title=f'Sklearn PCA of {dataset_name} dataset',
                      columns=columns_to_plot[dataset_name],
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        plot_features(inc_pca_results, labels,
                      title=f'Sklearn Incremental PCA of {dataset_name} dataset',
                      columns=columns_to_plot[dataset_name],
                      show=args.show_plots,
                      result_dir=plot_result_dir)

        # cluster with kmeans++
        print("Running K-Means++ clustering on PCA results...")
        kmeanspp_result = kmeans_plus_plus_fit(results, k_values[dataset_name], distance[dataset_name])
        plot_clusters(kmeanspp_result,
                      title=f'Kmeans++ Clustering of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)

        print("K-Means++ metric scores: ")
        kmeanspp_labels = pd.DataFrame(columns=['cluster'])
        for i, cl in enumerate(kmeanspp_result):
            l = pd.DataFrame({'cluster': [i] * cl.shape[0]}, index=cl.index)
            kmeanspp_labels = pd.concat([kmeanspp_labels, l]).sort_index()
        pprint({
            "silhouette": silhouette_score(results, kmeanspp_labels.squeeze()),
            "davies_bouldin": davies_bouldin_score(results, kmeanspp_labels.squeeze()),
            "adjusted_rand_score": adjusted_rand_score(labels.squeeze(), kmeanspp_labels.squeeze()),
            "v_measure": homogeneity_completeness_v_measure(labels.squeeze(), kmeanspp_labels.squeeze())[2],
        })
        print("K-Means++ metric scores before PCA:")
        pprint(previous_results[dataset_name]["kmeans++"])

        # cluster with OPTICS
        print("Running OPTICS clustering on PCA results...")
        params = optics_params[dataset_name]
        optics_result = fit_optics(results, params['min_samples'], params['distance'], params['algorithm'], params['eps'])
        optics_cluster_labels = optics_result.labels_
        optics_labels = optics_cluster_labels[optics_cluster_labels != -1]
        optics_clusters = [pd.DataFrame([results.loc[l_idx]
                                         for l_idx, l in enumerate(optics_cluster_labels)
                                         if l == label])
                           for label in np.unique(optics_labels)]
        plot_clusters(optics_clusters,
                      title=f'OPTICS Clustering of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        print("OPTICS metric scores: ")
        input_ = results[optics_cluster_labels != -1]
        labels_ = labels[optics_cluster_labels != -1].to_numpy()
        pprint({
            "silhouette": silhouette_score(input_, optics_labels.squeeze()),
            "davies_bouldin": davies_bouldin_score(input_, optics_labels.squeeze()),
            "adjusted_rand_score": adjusted_rand_score(labels_.squeeze(), optics_labels.squeeze()),
            "v_measure": homogeneity_completeness_v_measure(labels_.squeeze(), optics_labels.squeeze())[2],
        })
        print("OPTICS metric scores before PCA:")
        pprint(previous_results[dataset_name]["optics"])

        # Visualising everything in a 2-dimensional space
        orig_visualisation = pca_fit(input, n_components=2)
        orig_visualisation_clustered = [pd.DataFrame([input.loc[l_idx]
                                                      for l_idx in labels.index
                                                      if labels.loc[l_idx] == label])
                                        for label in labels.unique()]
        print("Running PCA for visualisation of original dataset...")
        orig_visualisation_clustered_PCA = [pca_fit(cl, n_components=2)[0] for cl in orig_visualisation_clustered]
        plot_clusters(orig_visualisation_clustered_PCA,
                      title=f'2-D Visualisation of original labeled clusters using PCA of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        print("Running PCA for visualisation of K-Means++ output...")
        kmeanspp_visualisation_PCA = [pca_fit(cl, n_components=2)[0] for cl in kmeanspp_result]
        plot_clusters(kmeanspp_visualisation_PCA,
                      title=f'K-Means 2-D Visualisation using PCA of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        print("Running PCA for visualisation of OPTICS output...")
        optics_visualisation_PCA = [pca_fit(cl, n_components=2)[0] for cl in optics_clusters]
        plot_clusters(optics_visualisation_PCA,
                      title=f'OPTICS 2-D Visualisation using PCA of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)

        orig_visualisation_clustered_UMAP = [umap.UMAP().fit_transform(cl).T for cl in orig_visualisation_clustered]
        plot_clusters(orig_visualisation_clustered_UMAP,
                      title=f'2-D Visualisation of original labeled clusters using UMAP of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        kmeanspp_visualisation_UMAP = [umap.UMAP().fit_transform(cl).T for cl in kmeanspp_result]
        plot_clusters(kmeanspp_visualisation_UMAP,
                      title=f'K-Means 2-D Visualisation using UMAP of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        optics_visualisation_UMAP = [umap.UMAP().fit_transform(cl).T for cl in optics_clusters]
        plot_clusters(optics_visualisation_UMAP,
                      title=f'OPTICS 2-D Visualisation using UMAP of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)