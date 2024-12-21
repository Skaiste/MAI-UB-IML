import os
import sys
import time
import pathlib
import argparse
import matplotlib.pyplot as plt
from pprint import pprint
import seaborn as sns

import numpy as np
import pandas as pd
from matplotlib import lines
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
def pca_fit(X, n_components=None, print_info=False):
    # Standardisation
    # Removing invalid data to allow eigenvector extraction
    X = X.fillna(X.mean())
    z_score = (X - X.mean()) / X.std().replace(0, np.nan)
    z_score = z_score.dropna(axis=1, how='any')

    # covariance matrix
    covar = z_score.cov()

    # Print covariance matrix in a readable format
    if print_info:
        print("Covariance Matrix:")
        # Round to 3 digits and ensure the full matrix is printed
        # with np.printoptions(threshold=np.inf, linewidth=np.inf, precision=3, suppress=True):
        print(covar.round(3).to_numpy())

    if np.isnan(covar).any().any():
        raise Exception("NaN values found in covar")
    if np.isinf(covar).any().any():
        raise Exception("Inf values found in covar")

    # eigenvectors and eigenvalues
    eigen_vals, eigen_vecs = np.linalg.eig(covar)
    eigen_vecs = eigen_vecs.real
    eigen_vals = eigen_vals.real
    if print_info:
        print("Eigenvalues: ", eigen_vals, "\nEigenvectors:")
        # with np.printoptions(threshold=np.inf, linewidth=np.inf, precision=3, suppress=True):
        print(eigen_vecs)

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

    if columns is None:
        columns = X.columns[:2]

    plt.figure(figsize=(10, 10))

    classes = y.unique()
    palette = sns.color_palette("Set2", len(classes))
    for idx, cl in enumerate(classes):
        cl_idx = y[y == cl].index
        sns.scatterplot(data=X[columns].loc[cl_idx], x=columns[0], y=columns[1], alpha=0.75, label=str(cl), color=palette[idx], s=200)
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

    plt.close()

    plt.figure(figsize=(10, 10))
    for idx, cl in enumerate(classes):
        cl_idx = y[y == cl].index
        sns.kdeplot(data=X[columns].loc[cl_idx], x=columns[0], y=columns[1], fill=True, alpha=0.5, label=str(cl), color=palette[idx])
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title(title)

    handles = [lines.Line2D([0], [0], color=color) for color in palette[:len(classes)]]
    plt.legend(handles=handles, labels=[str(cl) for cl in classes])
    if show:
        plt.show()
    else:
        if result_dir is None:
            result_dir = curr_dir / "results"
        filename = title.lower().replace(" ", "_") + "_sns.png"
        plt.savefig(result_dir / filename)

    plt.close()

def plot_eigenvectors(eigenvecs, title, result_dir=None, show=True):
    plt.figure(figsize=(10, 10))
    eigenvecs = pd.DataFrame(eigenvecs)
    palette = sns.color_palette("Set2", 1)
    sns.scatterplot(data=eigenvecs, x=0, y=1, alpha=0.75, color=palette[0], s=200)
    # plt.scatter(eigenvecs[:, 0], eigenvecs[:, 1])
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

    plt.close()

def plot_clusters(clusters, title, columns=None, result_dir=None, show=True):
    plt.figure(figsize=(10, 10))
    palette = sns.color_palette("Set2", len(clusters))
    for idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        if columns is None:
            columns = cluster.columns[:2]
        sns.scatterplot(data=cluster[columns], x=columns[0], y=columns[1], alpha=0.75, color=palette[idx], s=200)
    plt.title(title)
    if show:
        plt.show()
    else:
        if result_dir is None:
            result_dir = curr_dir / "results"
        filename = title.lower().replace(" ", "_") + ".png"
        plt.savefig(result_dir / filename)

    plt.close()


# %%

def kmeanspp_with_labels(data, k, distance):
    result = kmeans_plus_plus_fit(data, k, distance)
    labels = pd.DataFrame(columns=['cluster'])
    for i, cl in enumerate(result):
        l = pd.DataFrame({'cluster': [i] * cl.shape[0]}, index=cl.index)
        labels = pd.concat([labels, l]).sort_index()
    return result, labels

def optics_with_labels(data, min_samples, distance, algorithm, eps):
    result = fit_optics(data, min_samples, distance, algorithm, eps)
    cluster_labels = result.labels_
    labels = cluster_labels[cluster_labels != -1]
    return result, labels, cluster_labels

def optics_to_clusters(data, labels, cluster_labels):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    clusters = [pd.DataFrame([data.loc[l_idx]
                              for l_idx, l in enumerate(cluster_labels)
                              if l == label and l_idx in list(data.index)])
                for label in np.unique(labels)]
    return clusters

def df_to_clusters(data, labels):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)
    cluster = [pd.DataFrame([data.loc[l_idx]
                             for l_idx in labels.index
                             if labels.loc[l_idx] == label and l_idx in list(data.index)])
               for label in np.unique(labels)]
    return cluster

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
    columns_to_plot = {
        "cmc": ['children', 'wage'],
        "sick": ['T4U', 'FTI'],
        "mushroom": ['spore-print-color', 'gill-color']
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
    for fn in args.dataset_path:
        dataset_name = fn.stem
        if dataset_name == "cmc": continue
        plot_result_dir = curr_dir / "results" / dataset_name
        plot_result_dir.mkdir(parents=True, exist_ok=True)
        print("For dataset", dataset_name)
        normalise_nominal = True if dataset_name != "cmc" else False
        input, labels, plot_input = get_data(fn, cache_dir=None, cache=False, normalise_nominal=normalise_nominal)

        plot_features(plot_input, labels,
                      title=f'Initial Input of {dataset_name} dataset',
                      columns=columns_to_plot[dataset_name],
                      show=args.show_plots,
                      result_dir=plot_result_dir)



        print("Running PCA on input data...")
        results, eigenvectors = pca_fit(input, n_components=args.n_components, print_info=True)
        plot_eigenvectors(eigenvectors,
                          title=f'Eigenvectors of {dataset_name} dataset',
                          show=args.show_plots,
                          result_dir=plot_result_dir)
        plot_features(results, labels,
                      title=f'PCA of {dataset_name} dataset',
                      columns=results.columns[:2],
                      show=args.show_plots,
                      result_dir=plot_result_dir)

        # compare with sklearn PCA results
        pca = PCA(n_components=args.n_components)
        pca_results = pca.fit_transform(input)
        inc_pca = IncrementalPCA(n_components=args.n_components)
        inc_pca_results = inc_pca.fit_transform(input)
        plot_features(pca_results, labels,
                      title=f'Sklearn PCA of {dataset_name} dataset',
                      columns=[0, 1],
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        plot_features(inc_pca_results, labels,
                      title=f'Sklearn Incremental PCA of {dataset_name} dataset',
                      columns=[0, 1],
                      show=args.show_plots,
                      result_dir=plot_result_dir)

        # cluster with kmeans++
        print("Running K-Means++ clustering on original data...")
        orig_kmeanspp_tick = time.time()
        orig_kmeanspp_result, orig_kmeanspp_labels = kmeanspp_with_labels(input, k_values[dataset_name], distance[dataset_name])
        orig_kmeanspp_tock = time.time()
        print(f">>> Time taken for K-Means++ clustering on original data for {dataset_name} dataset: ", orig_kmeanspp_tock - orig_kmeanspp_tick, "seconds")
        print("Running K-Means++ clustering on PCA results...")
        kmeanspp_tick = time.time()
        kmeanspp_result, kmeanspp_labels = kmeanspp_with_labels(results, k_values[dataset_name], distance[dataset_name])
        kmeanspp_tock = time.time()
        print(f">>> Time taken for K-Means++ clustering on PCA results for {dataset_name} dataset: ", kmeanspp_tock - kmeanspp_tick, "seconds")
        print("K-Means++ metric scores: ")
        pprint({
            "silhouette": silhouette_score(results, kmeanspp_labels.squeeze()),
            "davies_bouldin": davies_bouldin_score(results, kmeanspp_labels.squeeze()),
            "adjusted_rand_score": adjusted_rand_score(labels.squeeze(), kmeanspp_labels.squeeze()),
            "v_measure": homogeneity_completeness_v_measure(labels.squeeze(), kmeanspp_labels.squeeze())[2],
        })
        print("K-Means++ metric scores before PCA:")
        pprint({
            "silhouette": silhouette_score(results, orig_kmeanspp_labels.squeeze()),
            "davies_bouldin": davies_bouldin_score(results, orig_kmeanspp_labels.squeeze()),
            "adjusted_rand_score": adjusted_rand_score(labels.squeeze(), orig_kmeanspp_labels.squeeze()),
            "v_measure": homogeneity_completeness_v_measure(labels.squeeze(), orig_kmeanspp_labels.squeeze())[2],
        })

        # cluster with OPTICS
        print("Running OPTICS clustering on original data...")
        params = optics_params[dataset_name]
        orig_optics_tick = time.time()
        orig_optics = optics_with_labels(input, params['min_samples'], params['distance'], params['algorithm'], params['eps'])
        orig_optics_tock = time.time()
        print(f">>> Time taken for OPTICS clustering on original data for {dataset_name} dataset: ", orig_optics_tock - orig_optics_tick, "seconds")
        orig_optics_result, orig_optics_labels, orig_optics_cluster_labels = orig_optics

        print("Running OPTICS clustering on PCA results...")
        optics_tick = time.time()
        _optics = optics_with_labels(results, params['min_samples'], params['distance'], params['algorithm'], params['eps'])
        optics_tock = time.time()
        print(f">>> Time taken for OPTICS clustering on PCA results for {dataset_name} dataset: ", optics_tock - optics_tick, "seconds")
        optics_result, optics_labels, optics_cluster_labels = _optics
        optics_clusters = optics_to_clusters(results, optics_labels, optics_cluster_labels)

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
        input_ = results[orig_optics_cluster_labels != -1]
        labels_ = labels[orig_optics_cluster_labels != -1].to_numpy()
        pprint({
            "silhouette": silhouette_score(input_, orig_optics_labels.squeeze()),
            "davies_bouldin": davies_bouldin_score(input_, orig_optics_labels.squeeze()),
            "adjusted_rand_score": adjusted_rand_score(labels_.squeeze(), orig_optics_labels.squeeze()),
            "v_measure": homogeneity_completeness_v_measure(labels_.squeeze(), orig_optics_labels.squeeze())[2],
        })

        # Visualising everything in a 2-dimensional space
        orig_visualisation_PCA = pca_fit(input, n_components=2)[0]
        orig_visualisation_clustered_PCA = df_to_clusters(orig_visualisation_PCA, labels)
        print("Running PCA for visualisation of original dataset...")
        plot_clusters(orig_visualisation_clustered_PCA,
                      title=f'2-D Visualisation of original labeled clusters using PCA of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)

        print("Running PCA for visualisation of K-Means++ output...")
        orig_kmeanspp_visualisation_PCA = df_to_clusters(orig_visualisation_PCA, orig_kmeanspp_labels["cluster"])
        plot_clusters(orig_kmeanspp_visualisation_PCA,
                      title=f'K-Means 2-D Visualisation using PCA of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        print("Running PCA for visualisation of reduced K-Means++ output...")
        kmeanspp_visualisation_PCA = df_to_clusters(orig_visualisation_PCA, kmeanspp_labels["cluster"])
        plot_clusters(kmeanspp_visualisation_PCA,
                      title=f'Reduced K-Means 2-D Visualisation using PCA of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)

        print("Running PCA for visualisation of OPTICS output...")
        orig_optics_visualisation_PCA = df_to_clusters(orig_visualisation_PCA, orig_optics_labels)
        plot_clusters(orig_optics_visualisation_PCA,
                      title=f'OPTICS 2-D Visualisation using PCA of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        print("Running PCA for visualisation of reduced OPTICS output...")
        optics_visualisation_PCA = df_to_clusters(orig_visualisation_PCA, optics_labels)
        plot_clusters(optics_visualisation_PCA,
                      title=f'Reduced OPTICS 2-D Visualisation using PCA of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)

        orig_visualisation_UMAP = umap.UMAP().fit_transform(input)
        orig_visualisation_clustered_UMAP = df_to_clusters(orig_visualisation_UMAP, labels)
        plot_clusters(orig_visualisation_clustered_UMAP,
                      title=f'2-D Visualisation of original labeled clusters using UMAP of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        orig_kmeanspp_visualisation_UMAP = df_to_clusters(orig_visualisation_UMAP, orig_kmeanspp_labels["cluster"])
        plot_clusters(orig_kmeanspp_visualisation_UMAP,
                      title=f'K-Means 2-D Visualisation using UMAP of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        kmeanspp_visualisation_UMAP = df_to_clusters(orig_visualisation_UMAP, kmeanspp_labels["cluster"])
        plot_clusters(kmeanspp_visualisation_UMAP,
                      title=f'Reduced K-Means 2-D Visualisation using UMAP of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        orig_optics_visualisation_UMAP = df_to_clusters(orig_visualisation_UMAP, orig_optics_labels)
        plot_clusters(orig_optics_visualisation_UMAP,
                      title=f'OPTICS 2-D Visualisation using UMAP of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)
        optics_visualisation_UMAP = df_to_clusters(orig_visualisation_UMAP, optics_labels)
        plot_clusters(optics_visualisation_UMAP,
                      title=f'Reduced OPTICS 2-D Visualisation using UMAP of {dataset_name} dataset',
                      show=args.show_plots,
                      result_dir=plot_result_dir)