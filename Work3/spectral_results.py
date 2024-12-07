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

# load spectral clustering results
for fn in results_dir.iterdir():
    if "spectral" in fn.name:
        results_spectral[fn.name.split("_")[0]] = pd.read_csv(fn)


datasets = ["cmc", "sick", "mushroom"]
fig, axs = plt.subplots(1, len(datasets), figsize=(len(datasets) * 6, 6))
best_gamma_for_datasets = {}

for i, dataset in enumerate(datasets):
    if dataset not in results_spectral:
        print(f"No results found for dataset: {dataset}")
        continue

    df = results_spectral[dataset]
    rbf_results = df[df["affinity"].str.contains("rbf_gamma")].copy()
    rbf_results["gamma"] = rbf_results["affinity"].str.extract(float_pattern).astype(float)

    # parse and separate `homogeneity_completeness_v_measure`
    if "homogeneity_completeness_v_measure" in rbf_results.columns:
        rbf_results[["homogeneity", "completeness", "v_measure"]] = pd.DataFrame(
            rbf_results["homogeneity_completeness_v_measure"]
            .apply(lambda x: list(eval(x)) if isinstance(x, str) else list(x))
            .tolist(),
            index=rbf_results.index
        )
        rbf_results["hcv_average"] = rbf_results[["homogeneity", "completeness", "v_measure"]].mean(axis=1)
    else:
        raise KeyError("Column 'homogeneity_completeness_v_measure' is missing in the data.")

    # aggregate metrics by gamma
    gamma_summary = rbf_results.groupby("gamma").mean(numeric_only=True)[[
        "silhouette", "davies_bouldin", "adjusted_rand_score", "hcv_average"
    ]]

    # best gamma based on the average of all metrics
    gamma_summary["overall_score"] = gamma_summary[["silhouette", "adjusted_rand_score", "hcv_average"]].mean(axis=1)
    best_gamma = gamma_summary["overall_score"].idxmax()
    best_gamma_for_datasets[dataset] = best_gamma
    axs[i].plot(gamma_summary.index, gamma_summary["silhouette"], label="Silhouette", marker="o")
    axs[i].plot(gamma_summary.index, gamma_summary["davies_bouldin"], label="Davies-Bouldin (lower is better)", marker="o")
    axs[i].plot(gamma_summary.index, gamma_summary["adjusted_rand_score"], label="Adjusted Rand", marker="o")
    axs[i].plot(gamma_summary.index, gamma_summary["hcv_average"], label="HCV Average", marker="o")
    axs[i].plot(gamma_summary.index, gamma_summary["overall_score"], label="Overall Score", linestyle="--", marker="o")

    axs[i].set_title(f"{dataset} Dataset: Gamma Performance")
    axs[i].set_xlabel("Gamma")
    axs[i].set_ylabel("Score")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()

print("\nBest Gamma for Each Dataset:")
for dataset, gamma in best_gamma_for_datasets.items():
    print(f"Dataset: {dataset}, Best Gamma: {gamma}")




# Compare Affinities
fig, axs = plt.subplots(1, len(datasets), figsize=(len(datasets) * 6, 6))

for i, dataset in enumerate(datasets):
    if dataset not in results_spectral:
        print(f"No results found for dataset: {dataset}")
        continue

    df = results_spectral[dataset]
    best_gamma = best_gamma_for_datasets[dataset]

    # 'rbf' with best gamma and 'nearest_neighbors'
    relevant_results = df[df["affinity"].str.contains(f"rbf_gamma{best_gamma}|nearest_neighbors")].copy()

    if "homogeneity_completeness_v_measure" in relevant_results.columns:
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

    #  overall score
    relevant_results["overall_score"] = (
        relevant_results["silhouette"] +
        relevant_results["adjusted_rand_score"] +
        relevant_results["hcv_average"]
    ) - relevant_results["davies_bouldin"]

    # group by affinity type and calculate mean metrics
    affinity_means = relevant_results.groupby("affinity_type")[[
        "silhouette", "davies_bouldin", "adjusted_rand_score", "hcv_average", "overall_score"
    ]].mean()

    print(f"\n{dataset.capitalize()} Dataset Affinity Comparison:\n", affinity_means)

    # plot for affinity comparison
    affinity_means.T.plot(
        kind="bar",
        ax=axs[i],
        title=f"{dataset.capitalize()} Dataset: Affinity Comparison",
        legend=True,
        grid=True,
    )
    axs[i].set_ylabel("Score")
    axs[i].set_xlabel("Metrics")
    axs[i].legend(title="Affinity", loc="best")

plt.tight_layout()
plt.show()

# best number of clusters for each dataset
fig, axs = plt.subplots(1, len(datasets), figsize=(len(datasets) * 6, 6))
best_cluster_numbers = {}

for i, dataset in enumerate(datasets):
    if dataset not in results_spectral:
        print(f"No results found for dataset: {dataset}")
        continue

    df = results_spectral[dataset]
    df.rename(columns={"k": "cluster_number"}, inplace=True)

    best_affinity = "rbf" if dataset == "sick" else "nearest_neighbors"
    filtered_df = df[df["affinity"].str.contains(best_affinity)].copy()

    if "homogeneity_completeness_v_measure" in filtered_df.columns:
        filtered_df[["homogeneity", "completeness", "v_measure"]] = pd.DataFrame(
            filtered_df["homogeneity_completeness_v_measure"]
            .apply(lambda x: list(eval(x)) if isinstance(x, str) else list(x))
            .tolist(),
            index=filtered_df.index
        )
        filtered_df["hcv_average"] = filtered_df[["homogeneity", "completeness", "v_measure"]].mean(axis=1)

    cluster_summary = filtered_df.groupby("cluster_number").mean(numeric_only=True)[[
        "silhouette", "davies_bouldin", "adjusted_rand_score", "hcv_average"
    ]]

    cluster_summary["overall_score"] = cluster_summary[["silhouette", "adjusted_rand_score", "hcv_average"]].mean(axis=1)

    # best cluster number
    best_cluster_number = cluster_summary["overall_score"].idxmax()
    best_cluster_numbers[dataset] = best_cluster_number

    axs[i].plot(cluster_summary.index, cluster_summary["silhouette"], label="Silhouette", marker="o")
    axs[i].plot(cluster_summary.index, cluster_summary["davies_bouldin"], label="Davies-Bouldin (lower is better)", marker="o")
    axs[i].plot(cluster_summary.index, cluster_summary["adjusted_rand_score"], label="Adjusted Rand", marker="o")
    axs[i].plot(cluster_summary.index, cluster_summary["hcv_average"], label="HCV Average", marker="o")
    axs[i].plot(cluster_summary.index, cluster_summary["overall_score"], label="Overall Score", linestyle="--", marker="o")

    axs[i].set_title(f"{dataset} Dataset: Cluster Number Performance")
    axs[i].set_xlabel("Cluster Number")
    axs[i].set_ylabel("Score")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()

print("\nBest Number of Clusters for Each Dataset:")
for dataset, cluster_number in best_cluster_numbers.items():
    print(f"Dataset: {dataset}, Best Cluster Number: {cluster_number}")
