import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# Setup paths
try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
results_dir = curr_dir / "results"

datasets = ["cmc", "sick", "mushroom"]
metrics = ["silhouette", "davies_bouldin", "adjusted_rand_score", "homogeneity", "completeness", "v_measure"]
metric_labels = {
    "silhouette": "Silhouette",
    "davies_bouldin": "Davies-Bouldin",
    "adjusted_rand_score": "Adjusted Rand",
    "homogeneity": "Homogeneity",
    "completeness": "Completeness",
    "v_measure": "V-Measure",
}

# Step 1: Best Number of Clusters
print("Step 1: Plotting Best Number of Clusters...")

# Combined Plot for Best Number of Clusters
fig_cluster, axs_cluster = plt.subplots(len(metrics), len(datasets), figsize=(len(datasets) * 6, len(metrics) * 6))

cluster_results = {}
for row, metric in enumerate(metrics):
    for col, dataset in enumerate(datasets):
        try:
            file_path = results_dir / f"{dataset}_fuzzy_results.csv"
            df = pd.read_csv(file_path)

            # Rename 'k' to 'cluster_number' internally for consistency
            df.rename(columns={"k": "cluster_number"}, inplace=True)

            # Save results for elimination
            if dataset not in cluster_results:
                cluster_results[dataset] = df

            # Plot metric vs cluster number
            x = df['cluster_number']
            y = df[metric]

            ax = axs_cluster[row, col]
            ax.scatter(x, y, label=f'{metric}', color='blue')
            ax.set_title(f'{dataset}: {metric} for Cluster Numbers')
            ax.set_xlabel('Cluster Number')
            ax.set_ylabel(metric_labels[metric])
            ax.grid(True)
        except Exception as e:
            print(f"Error processing {dataset} for {metric}: {e}")

plt.tight_layout()
plt.show()

# Step 2: Best Degree of Fuzziness (m)
print("Step 2: Plotting Best Degree of Fuzziness (m)...")
m_values = [1.5, 2.0, 2.5]  # Example values of m

# Combined Plot for Best Degree of Fuzziness
fig_fuzziness, axs_fuzziness = plt.subplots(len(metrics), len(datasets), figsize=(len(datasets) * 6, len(metrics) * 6))

fuzziness_results = {}
for row, metric in enumerate(metrics):
    for col, dataset in enumerate(datasets):
        try:
            file_path = results_dir / f"{dataset}_fuzzy_results.csv"
            df = pd.read_csv(file_path)

            # Filter only rows matching m_values
            df_fuzziness = df[df['m'].isin(m_values)]

            if dataset not in fuzziness_results:
                fuzziness_results[dataset] = df_fuzziness

            x = df_fuzziness["m"]
            y = df_fuzziness[metric]

            ax = axs_fuzziness[row, col]
            ax.scatter(x, y, label=f'{metric}', color='orange')
            ax.set_title(f'{dataset}: {metric} for Fuzziness Degrees')
            ax.set_xlabel('Degree of Fuzziness (m)')
            ax.set_ylabel(metric_labels[metric])
            ax.grid(True)
        except Exception as e:
            print(f"Error processing {dataset} for {metric}: {e}")

plt.tight_layout()
plt.show()

# Step 3: Elimination Process for Both `Cluster Number` and `m`
def eliminate_best_parameters(cluster_data, fuzziness_data, metrics, metric_priority):
    """
    Eliminate to find the best overall parameters (cluster_number and m) for each dataset.
    """
    eliminated_params = {}
    for dataset in cluster_data.keys():
        remaining_clusters = cluster_data[dataset]['cluster_number'].unique().tolist()  # Cluster numbers
        remaining_fuzziness = fuzziness_data[dataset]['m'].unique().tolist()  # Fuzziness values

        print(f"\nStarting elimination for {dataset}...")
        print(f"Initial Clusters: {remaining_clusters}")
        print(f"Initial Fuzziness Values: {remaining_fuzziness}")

        for metric in metric_priority:
            # Check if metric exists in data
            if metric not in cluster_data[dataset].columns or metric not in fuzziness_data[dataset].columns:
                print(f"  Warning: Metric {metric_labels[metric]} is missing for {dataset}. Skipping...")
                continue

            # Eliminate for cluster number
            cluster_scores = {
                c: cluster_data[dataset][cluster_data[dataset]['cluster_number'] == c][metric].mean() for c in remaining_clusters
            }
            best_cluster = max(cluster_scores, key=cluster_scores.get)
            remaining_clusters = [best_cluster]

            # Eliminate for fuzziness
            fuzziness_scores = {
                m: fuzziness_data[dataset][fuzziness_data[dataset]['m'] == m][metric].mean() for m in remaining_fuzziness
            }
            best_fuzziness = max(fuzziness_scores, key=fuzziness_scores.get)
            remaining_fuzziness = [best_fuzziness]

            print(f"  After {metric_labels[metric]}: Remaining clusters = {remaining_clusters}, Fuzziness = {remaining_fuzziness}")

        # Final parameters
        eliminated_params[dataset] = {
            "best_cluster_number": remaining_clusters[0],
            "best_fuzziness": remaining_fuzziness[0],
        }
        print(f"Best parameters for {dataset}: {eliminated_params[dataset]}")
    return eliminated_params


# Perform elimination
metric_priority = metrics
eliminated_params = eliminate_best_parameters(cluster_results, fuzziness_results, metrics, metric_priority)

# Print final results
print("\nBest Parameters After Elimination Process:")
for dataset, params in eliminated_params.items():
    print(f"{dataset}: Best Cluster Number = {params['best_cluster_number']}, Best Fuzziness = {params['best_fuzziness']}")
