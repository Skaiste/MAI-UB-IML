import pandas as pd
import matplotlib.pyplot as plt
import re

# File paths for the datasets
file_paths = [
    "results/cmc_kmeans++_results.csv",
    "results/sick_kmeans++_results.csv",
    "results/mushroom_kmeans++_results.csv"
]

# Dataset names (derived from file names for display)
dataset_names = ['CMC', 'SICK', 'Mushroom']

# Metrics to plot
metrics = ['silhouette', 'davies_bouldin', 'adjusted_rand_score', 'v_measure']

# Friendly names for metrics
metric_titles = {
    'silhouette': "Silhouette",
    'davies_bouldin': "Davies Bouldin Score",
    'adjusted_rand_score': "Adjusted Rand Score",
    'v_measure': "V-Measure (from Homogeneity Completeness)"
}

# Initialize a list to hold all datasets
datasets = []

# Load each dataset and preprocess
for file_path in file_paths:
    data = pd.read_csv(file_path)

    # Preprocess 'homogeneity_completeness_v_measure' to extract the third value (v-measure)
    if 'homogeneity_completeness_v_measure' in data.columns:
        float_pattern = re.compile(r"0\.\d+")  # Regex to match floating-point numbers
        data['v_measure'] = data['homogeneity_completeness_v_measure'].apply(
            lambda x: float(float_pattern.findall(x)[2]) if isinstance(x, str) else x
        )

    datasets.append(data)

# Set up the subplot grid
num_metrics = len(metrics)
num_datasets = len(datasets)
fig, axes = plt.subplots(num_metrics, num_datasets, figsize=(4 * num_datasets, 4 * num_metrics), sharex=True)

# Ensure axes is always a 2D array
if num_metrics == 1:
    axes = [axes]
if num_datasets == 1:
    axes = [[ax] for ax in axes]

# Plot metrics for each dataset
for col_idx, (dataset_name, dataset_data) in enumerate(zip(dataset_names, datasets)):
    for row_idx, metric in enumerate(metrics):
        if metric not in dataset_data.columns:
            continue  # Skip if the column is not in the data

        ax = axes[row_idx][col_idx]
        ax.plot(dataset_data['k'], dataset_data[metric], marker='o', linestyle='-', label=f"{metric_titles[metric]}")
        if row_idx == 0:
            ax.set_title(f'Dataset: {dataset_name}', fontsize=8)
        ax.set_ylabel(metric_titles[metric], fontsize=6)

        ax.grid(True)
        ax.legend(fontsize=5)
        ax.tick_params(axis='x', labelsize=6)  # Ensure tick labels (numbers) are visible

# Adjust spacing to prevent overlaps and reduce size
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.savefig("plots/kmeans++.png", dpi=300)
plt.close(fig)
