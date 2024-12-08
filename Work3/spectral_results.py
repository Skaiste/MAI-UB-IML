import os
import re
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
sys.path.append(str(curr_dir))

float_pattern = re.compile(r"gamma([\d\.]+)")  # extracting gamma
results_dir = curr_dir / "results"
results_spectral = {}

# Load spectral clustering results
for fn in results_dir.iterdir():
    if "spectral" in fn.name:
        results_spectral[fn.name.split("_")[0]] = pd.read_csv(fn)

datasets = ["cmc", "sick", "mushroom"]
metrics = ["silhouette", "davies_bouldin", "adjusted_rand_score", "hcv_average"]
metric_labels = {
    "silhouette": "Silhouette",
    "davies_bouldin": "Davies-Bouldin",
    "adjusted_rand_score": "Adjusted Rand",
    "hcv_average": "HC Score"
}

def plot_scatter_subplots(data, x_values, x_label, title_prefix, y_labels, metric_list):
    n_metrics = len(metric_list)
    n_datasets = len(data)

    fig, axs = plt.subplots(n_metrics, n_datasets, figsize=(n_datasets * 5, n_metrics * 3))
    for row, metric in enumerate(metric_list):
        for col, (dataset, df) in enumerate(data.items()):
            axs[row, col].scatter(x_values[dataset], df[metric], marker="o")
            axs[row, col].set_title(f"{dataset}: {metric_labels[metric]}")
            axs[row, col].set_xlabel(x_label)
            axs[row, col].set_ylabel(y_labels[row])
            axs[row, col].grid(True)

    plt.tight_layout()
    plt.show()


# best gamma
best_gamma_for_datasets = {}
gamma_results = {}

for dataset in datasets:
    if dataset not in results_spectral:
        print(f"No results found for dataset: {dataset}")
        continue

    df = results_spectral[dataset]
    rbf_results = df[df["affinity"].str.contains("rbf_gamma")].copy()
    rbf_results["gamma"] = rbf_results["affinity"].str.extract(float_pattern).astype(float)

    # parse and separate `homogeneity_completeness_v_measure`
    rbf_results[["homogeneity", "completeness", "v_measure"]] = pd.DataFrame(
        rbf_results["homogeneity_completeness_v_measure"]
        .apply(lambda x: list(eval(x)) if isinstance(x, str) else list(x))
        .tolist(),
        index=rbf_results.index
    )
    rbf_results["hcv_average"] = rbf_results[["homogeneity", "completeness", "v_measure"]].mean(axis=1)

    # aggregate metrics by gamma
    gamma_summary = rbf_results.groupby("gamma").mean(numeric_only=True)[metrics]
    gamma_results[dataset] = gamma_summary

    # the best gamma for each metric
    best_gamma = {metric: gamma_summary[metric].idxmax() for metric in metrics}
    best_gamma_for_datasets[dataset] = best_gamma

print("\nBest Gamma for Each Dataset and Metric:")
for dataset, gamma_info in best_gamma_for_datasets.items():
    print(f"{dataset}:")
    for metric, gamma in gamma_info.items():
        print(f"  - Best Gamma for {metric_labels[metric]}: {gamma}")

# scatter plot for gamma results
plot_scatter_subplots(
    data=gamma_results,
    x_values={ds: gamma_results[ds].index for ds in gamma_results},
    x_label="Gamma",
    title_prefix="Gamma Performance",
    y_labels=[metric_labels[m] for m in metrics],
    metric_list=metrics,
)


#  affinities
affinity_results = {}
best_affinity_for_datasets = {}

for dataset in datasets:
    if dataset not in results_spectral:
        print(f"No results found for dataset: {dataset}")
        continue

    df = results_spectral[dataset]
    relevant_results = df[df["affinity"].str.contains("rbf_gamma|nearest_neighbors")].copy()

    relevant_results[["homogeneity", "completeness", "v_measure"]] = pd.DataFrame(
        relevant_results["homogeneity_completeness_v_measure"]
        .apply(lambda x: list(eval(x)) if isinstance(x, str) else list(x))
        .tolist(),
        index=relevant_results.index
    )
    relevant_results["hcv_average"] = relevant_results[["homogeneity", "completeness", "v_measure"]].mean(axis=1)

    # add a column for affinity type
    relevant_results["affinity_type"] = relevant_results["affinity"].apply(
        lambda x: "rbf" if "rbf" in x else "nearest_neighbors"
    )

    affinity_summary = relevant_results.groupby("affinity_type").mean(numeric_only=True)[metrics]
    affinity_results[dataset] = affinity_summary

    # best affinity for each metric
    best_affinity = {metric: affinity_summary[metric].idxmax() for metric in metrics}
    best_affinity_for_datasets[dataset] = best_affinity

print("\nBest Affinity for Each Dataset and Metric:")
for dataset, affinity_info in best_affinity_for_datasets.items():
    print(f"{dataset}:")
    for metric, affinity in affinity_info.items():
        print(f"  - Best Affinity for {metric_labels[metric]}: {affinity}")

# scatter plot for affinity results
plot_scatter_subplots(
    data=affinity_results,
    x_values={ds: affinity_results[ds].index for ds in affinity_results},
    x_label="Affinity Type",
    title_prefix="Affinity Comparison",
    y_labels=[metric_labels[m] for m in metrics],
    metric_list=metrics,
)


# finding the best cluster number
cluster_results = {}
best_cluster_numbers = {}

for dataset in datasets:
    if dataset not in results_spectral:
        print(f"No results found for dataset: {dataset}")
        continue

    df = results_spectral[dataset]
    df.rename(columns={"k": "cluster_number"}, inplace=True)
    filtered_df = df.copy()

    filtered_df[["homogeneity", "completeness", "v_measure"]] = pd.DataFrame(
        filtered_df["homogeneity_completeness_v_measure"]
        .apply(lambda x: list(eval(x)) if isinstance(x, str) else list(x))
        .tolist(),
        index=filtered_df.index
    )
    filtered_df["hcv_average"] = filtered_df[["homogeneity", "completeness", "v_measure"]].mean(axis=1)

    # aggregate metrics by cluster number
    cluster_summary = filtered_df.groupby("cluster_number").mean(numeric_only=True)[metrics]
    cluster_results[dataset] = cluster_summary

    best_clusters = {metric: cluster_summary[metric].idxmax() for metric in metrics}
    best_cluster_numbers[dataset] = best_clusters

# best cluster number
print("\nBest cluster Number for Each Dataset and Metric:")
for dataset, cluster_info in best_cluster_numbers.items():
    print(f"{dataset}:")
    for metric, cluster in cluster_info.items():
        print(f"  - Best Cluster Number for {metric_labels[metric]}: {cluster}")

# cluster results
plot_scatter_subplots(
    data=cluster_results,
    x_values={ds: cluster_results[ds].index for ds in cluster_results},
    x_label="Cluster Number",
    title_prefix="Cluster Number Performance",
    y_labels=[metric_labels[m] for m in metrics],
    metric_list=metrics,
)

# cluster number elimination
def eliminate_clusters(data, metrics, metric_priority):
    eliminated_clusters = {}
    
    for dataset, cluster_data in data.items():
        remaining_clusters = cluster_data.index.tolist()
        all_steps = {}  

        print(f"\nStarting elimination for {dataset}:")
        for metric in metric_priority:
            metric_values = cluster_data.loc[remaining_clusters, metric]
            best_value = metric_values.max()
            remaining_clusters = metric_values[metric_values == best_value].index.tolist()
            all_steps[metric] = remaining_clusters.copy()  
            
            print(f"  After {metric_labels[metric]}: Remaining clusters = {remaining_clusters}")
        
        # if multiple clusters remain, choose the first one for simplicity
        best_cluster = remaining_clusters[0]
        eliminated_clusters[dataset] = {
            "best_cluster": best_cluster,
            "elimination_steps": all_steps 
        }
        print(f"Best cluster for {dataset}: {best_cluster}")
    
    return eliminated_clusters


# cluster elimination
metric_priority = ["silhouette", "davies_bouldin", "adjusted_rand_score", "hcv_average"]
elimination_results = eliminate_clusters(cluster_results, metrics, metric_priority)

# final results
print("\nBest Clusters After Elimination Process:")
for dataset, result in elimination_results.items():
    print(f"  {dataset}: Best Cluster Number = {result['best_cluster']}")


def plot_elimination(data, elimination_results, metric_priority):

    n_datasets = len(data)

    # subplots, one for each dataset
    fig, axs = plt.subplots(1, n_datasets, figsize=(n_datasets * 6, 6), sharey=True)

    for i, (dataset_name, cluster_data) in enumerate(data.items()):
        ax = axs[i]
        elimination_steps = elimination_results[dataset_name]["elimination_steps"]

        # Plot elimination process for the current dataset
        for j, metric in enumerate(metric_priority):
            clusters = elimination_steps.get(metric, [])
            ax.plot([j] * len(clusters), clusters, 'o-', label=f"After {metric_labels[metric]}")
        
        # Configure plot appearance
        ax.set_title(f"{dataset_name} Elimination Process")
        ax.set_xlabel("Elimination Step")
        ax.set_xticks(range(len(metric_priority)))
        ax.set_xticklabels([metric_labels[metric] for metric in metric_priority], rotation=45)
        if i == 0:  # Add shared y-axis label only to the first subplot
            ax.set_ylabel("Remaining Clusters")
        ax.legend()

    # ajjust layout and show plot
    plt.tight_layout()
    plt.show()


# combined plot for all datasets
plot_elimination(cluster_results, elimination_results, metric_priority)