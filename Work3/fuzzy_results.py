import os
import pathlib
import re
import pandas as pd
import matplotlib.pyplot as plt

# paths
try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
results_dir = curr_dir / "results"

# define datasets and metrics
datasets = ["cmc", "sick", "mushroom"]
metrics = ["silhouette", "davies_bouldin", "adjusted_rand_score", "v_measure"]
metric_labels = {
    "silhouette": "Silhouette",
    "davies_bouldin": "Davies-Bouldin",
    "adjusted_rand_score": "Adjusted Rand",
    "v_measure": "V-Measure",
}

# extract information from file names
float_pattern = re.compile(r"m([\d\.]+)_k([\d\.]+)")  

# fuzzy clustering results
fuzzy_results = {}
for fn in results_dir.iterdir():
    if "_fuzzy_results.csv" in fn.name:
        dataset_name = fn.name.split("_fuzzy_results.csv")[0]
        fuzzy_results[dataset_name] = pd.read_csv(fn)

# ensure all datasets are loaded
for dataset in datasets:
    if dataset not in fuzzy_results:
        print(f"Warning: No results found for dataset {dataset}. Skipping...")
        continue

# 1:best number of clusters
fig_cluster, axs_cluster = plt.subplots(len(metrics), len(datasets), figsize=(len(datasets) * 6, len(metrics) * 6))

cluster_results = {}
for row, metric in enumerate(metrics):
    for col, dataset in enumerate(datasets):
        if dataset not in fuzzy_results:
            continue
        df = fuzzy_results[dataset]
        df.rename(columns={"k": "cluster_number"}, inplace=True)  # ensure 'k' is labeled consistently

        if dataset not in cluster_results:
            cluster_results[dataset] = df

        x = df['cluster_number']
        y = df[metric]

        ax = axs_cluster[row, col]
        ax.scatter(x, y, label=f'{metric}', color='blue')
        ax.set_title(f'{dataset}: Cluster Number  ')
        ax.set_ylabel(metric_labels[metric])
        ax.grid(True)

plt.tight_layout()
plt.show()

# best degree of fuzziness (m)
fig_fuzziness, axs_fuzziness = plt.subplots(len(metrics), len(datasets), figsize=(len(datasets) * 6, len(metrics) * 6))

fuzziness_results = {}
for row, metric in enumerate(metrics):
    for col, dataset in enumerate(datasets):
        if dataset not in fuzzy_results:
            continue
        df = fuzzy_results[dataset]
        if 'm' not in df.columns:
            print(f"Warning: 'm' column missing for {dataset}. Skipping...")
            continue

        if dataset not in fuzziness_results:
            fuzziness_results[dataset] = df

        x = df['m']
        y = df[metric]

        ax = axs_fuzziness[row, col]
        ax.scatter(x, y, label=f'{metric}', color='orange')
        ax.set_title(f'{dataset}: Degree of Fuzziness (m)')
        ax.set_ylabel(metric_labels[metric])
        ax.grid(True)

plt.tight_layout()
plt.show()

# 3: elimination process
def eliminate_best_parameters(cluster_data, fuzziness_data, metrics, metric_priority):
    eliminated_params = {}
    for dataset in cluster_data.keys():
        if dataset not in fuzziness_data:
            print(f"Warning: No fuzziness data found for {dataset}. Skipping elimination...")
            continue

        remaining_clusters = cluster_data[dataset]['cluster_number'].unique().tolist()
        remaining_fuzziness = fuzziness_data[dataset]['m'].unique().tolist()

        print(f"\nStarting elimination for {dataset}...")

        for metric in metric_priority:
            if metric not in cluster_data[dataset].columns or metric not in fuzziness_data[dataset].columns:
                print(f"  Warning: Metric {metric_labels[metric]} is missing for {dataset}. Skipping...")
                continue

            # eliminate for cluster number
            cluster_scores = {
                c: cluster_data[dataset][cluster_data[dataset]['cluster_number'] == c][metric].mean() for c in remaining_clusters
            }
            best_cluster = max(cluster_scores, key=cluster_scores.get)
            remaining_clusters = [best_cluster]

            # eliminate for fuzziness
            fuzziness_scores = {
                m: fuzziness_data[dataset][fuzziness_data[dataset]['m'] == m][metric].mean() for m in remaining_fuzziness
            }
            best_fuzziness = max(fuzziness_scores, key=fuzziness_scores.get)
            remaining_fuzziness = [best_fuzziness]

            print(f"  After {metric_labels[metric]}: Remaining clusters = {remaining_clusters}, Fuzziness = {remaining_fuzziness}")

        eliminated_params[dataset] = {
            "best_cluster_number": remaining_clusters[0],
            "best_fuzziness": remaining_fuzziness[0],
        }
        print(f"Best parameters for {dataset}: {eliminated_params[dataset]}")
    return eliminated_params


metric_priority = metrics
eliminated_params = eliminate_best_parameters(cluster_results, fuzziness_results, metrics, metric_priority)

print("\nBest Parameters After Elimination Process:")
for dataset, params in eliminated_params.items():
    print(f"{dataset}: Best Cluster Number = {params['best_cluster_number']}, Best Fuzziness = {params['best_fuzziness']}")

# visualization
def plot_best_parameters_only(eliminated_params):
  
    fig, axs = plt.subplots(1, len(eliminated_params), figsize=(len(eliminated_params) * 6, 6), sharey=True)
    
    for i, (dataset, params) in enumerate(eliminated_params.items()):
        ax = axs[i]
        
        # the best parameters
        best_cluster = params["best_cluster_number"]
        best_fuzziness = params["best_fuzziness"]
        
        # bar plot for the best parameters
        ax.bar(["Best Cluster Number", "Best Fuzziness"], [best_cluster, best_fuzziness], color=["blue", "orange"])
        
        # add annotations
        ax.text(0, best_cluster + 0.1, f"{best_cluster:.1f}", ha="center", va="bottom", fontsize=10, color="blue")
        ax.text(1, best_fuzziness + 0.1, f"{best_fuzziness:.1f}", ha="center", va="bottom", fontsize=10, color="orange")
        
        # plot appearance
        ax.set_title(f"Best Parameters for {dataset}")
        ax.set_ylabel("Value")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


plot_best_parameters_only(eliminated_params)