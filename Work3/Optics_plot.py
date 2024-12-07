import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse


parser = argparse.ArgumentParser(description="Runs an algorithm once on provided data")
parser.add_argument(
        "-dataset",
        type=str,
        default='mushroom',
        help="dataset for plotting"
    )
parser.add_argument(
        "-max_eps",
        type=float,
        default=9,
        help="max epsilon to use for plotting"
    )
parser.add_argument(
        "-min_samples",
        type=int,
        default=15,
        help="min samples to use for plotting"
    )

parser.add_argument(
        "-distance_metric",
        type=str,
        default='l1',
        help="distance metric to use for plotting"
    )

args = parser.parse_args()

#dataset = "mushroom"
dataset = args.dataset
file_path = "results/"+dataset+"_optics_results.csv"
data = pd.read_csv(file_path)


data[['distance_metric', 'search_type', 'min_samples', 'max_eps']] = data['Algorithm'].str.split(
    "_", expand=True
)
data['min_samples'] = data['min_samples'].astype(int)
data['max_eps'] = data['max_eps'].astype(float)


metrics = ['silhouette', 'davies_bouldin', 'adjusted_rand_score', 'homogeneity_completeness_v_measure', 'noise']

float_pattern = re.compile(r"0\.\d+")
data['homogeneity_completeness_v_measure'] = data['homogeneity_completeness_v_measure'].apply(
    lambda x: float(float_pattern.findall(x)[2]) if isinstance(x, str) else x
)

aggregated_data = data.groupby('max_eps')[metrics].mean().reset_index()


fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.plot(aggregated_data['max_eps'], aggregated_data[metric], marker='o', color='skyblue')

    ax.set_title(f"Mean {metric.capitalize()}", fontsize=10)
    ax.set_xlabel("Max Epsilon", fontsize=8)

    ax.set_ylabel(f"Mean Value", fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(aggregated_data['max_eps'])
    ax.set_xticklabels(aggregated_data['max_eps'], rotation=45, fontsize=8)


plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.suptitle("Comparison of Metrics by Max Epsilon", fontsize=12)
fig.savefig("plots/"+ dataset+"_optics_comparison_by_max_eps.png", dpi=300)
plt.close(fig)


data = data[data["max_eps"] == (args.max_eps)]

aggregated_data = data.groupby('min_samples')[metrics].mean().reset_index()


fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)
for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.plot(aggregated_data['min_samples'], aggregated_data[metric], marker='o', color='skyblue')

    ax.set_title(f"Mean {metric.capitalize()}", fontsize=10)
    ax.set_xlabel("Min samples", fontsize=8)

    ax.set_ylabel(f"Mean Value", fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(aggregated_data['min_samples'])
    ax.set_xticklabels(aggregated_data['min_samples'], rotation=45, fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
fig.savefig("plots/"+ dataset+"_optics_comparison_by_min_samples.png", dpi=300)
plt.close(fig)


data = data[data["min_samples"] == (args.min_samples)]


aggregated_data = data.groupby('distance_metric')[metrics].mean().reset_index()

fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)

for i, metric in enumerate(metrics):
    ax = axes[i]

    ax.plot(aggregated_data['distance_metric'], aggregated_data[metric], marker='o', color='skyblue')

    ax.set_title(f"Mean {metric.capitalize()}", fontsize=10)
    ax.set_xlabel("Distance metric samples", fontsize=8)

    ax.set_ylabel(f"Mean Value", fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(aggregated_data['distance_metric'])
    ax.set_xticklabels(aggregated_data['distance_metric'], rotation=45, fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.suptitle("Comparison of Metrics by Distance Metric", fontsize=12)
fig.savefig("plots/"+ dataset+"_optics_comparison_by_distance_metric.png", dpi=300)
plt.close(fig)


data = data[data["distance_metric"] == args.distance_metric]

fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)

for i, metric in enumerate(metrics):
    ax = axes[i]


    ax.plot(data['search_type'], data[metric], marker='o', color='skyblue')
    ax.set_title(f"Mean {metric.capitalize()}", fontsize=10)
    ax.set_xlabel("Algorithm", fontsize=8)
    ax.set_ylabel(f"Mean Value", fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(data['search_type'])
    ax.set_xticklabels(data['search_type'], rotation=45, fontsize=8)


plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.suptitle("Comparison of Metrics by Algorithm", fontsize=12)
fig.savefig("plots/"+ dataset+"_optics_comparison_by_algorihm.png", dpi=300)
plt.close(fig)
