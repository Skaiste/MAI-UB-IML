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

float_pattern = re.compile(r"0\.\d+")

results_dir = curr_dir / "results"
results_k_means = {}
results_g_means = {}
for fn in results_dir.iterdir():
    if "kmeans" in fn.name:
        results_k_means[fn.name.split("_")[0]] = pd.read_csv(fn)
    if "gmeans" in fn.name:
        results_g_means[fn.name.split("_")[0]] = pd.read_csv(fn)

# %%
best_seed_for_k = {dataset:{} for dataset in results_k_means.keys()}
# Determine the best seed for different k
fig, axs = plt.subplots(1, len(results_k_means), figsize=(len(results_k_means)*6, 3))
for i, (dataset, df) in enumerate(results_k_means.items()):
    df = df.drop("Unnamed: 0", axis=1)
    # Group by 'k' and compute averages for 'distance'
    table_values = []
    for k_val, k_group in df.groupby('k'):
        numerical_cols = k_group.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = numerical_cols.drop("seed")
        distance_averages = []
        for s, s_group in k_group.groupby('seed'):
            means = s_group[numerical_cols].mean()
            means['seed'] = s
            means['k'] = k_val
            distance_averages.append(means)
        distance_averages = pd.DataFrame(distance_averages)

        # Find the best seed for each 'k'
        best_seed_row = distance_averages.loc[distance_averages['sum_of_squared_error'].idxmin()]
        best_seed = best_seed_row['seed']
        best_seed_for_k[dataset][k_val] = best_seed
        best_sse = best_seed_row['sum_of_squared_error']

        # Plot SSE results in the table
        seed_sse_values = np.round(distance_averages['sum_of_squared_error'].values, 0).astype(int)
        table_values.append(seed_sse_values)

    axs[i].set_title(f"{dataset} dataset: SSE per Seed for K value")

    axs[i].axis('off')
    table = axs[i].table(cellText=table_values,
                         colLabels=df['seed'].unique(),
                         rowLabels=df['k'].unique(),
                         loc='top',
                         cellLoc='center',
                         bbox=[0.0, 0.0, 1.1, 0.9])

    for idx,k in enumerate(best_seed_for_k[dataset]):
        highlighted_cell = (idx+1, df['seed'].unique().tolist().index(best_seed_for_k[dataset][k]))
        table[highlighted_cell].set_text_props(weight='bold', color='black')

    table.auto_set_font_size(False)
    table.set_fontsize(7)

plt.show()

# %%
# Plotting K means k selection based on elbow method with best seeds
best_k = {"cmc": 4, "sick": 4, "mushroom": 5}
fig, axs = plt.subplots(1, len(results_k_means), figsize=(len(results_k_means)*6, 6))
for i, (dataset, df) in enumerate(results_k_means.items()):
    grouped = df.groupby('distance')
    highlight_x = best_k[dataset]
    highlight_y = 0
    for distance, group in grouped:
        sse = {k_val:k_group[k_group['seed'] == best_seed_for_k[dataset][k_val]]['sum_of_squared_error'].values for k_val, k_group in group.groupby('k')}
        axs[i].plot(sse.keys(), sse.values(), label=f"{distance} distance")
        highlight_y += sse[highlight_x]

    # highlight selected K
    highlight_y = highlight_y // len(results_k_means)
    axs[i].scatter(highlight_x, highlight_y, s=600, edgecolors='r', facecolors='none', linewidths=2)

    axs[i].set_xlabel("K")
    axs[i].set_ylabel("Sum of Squared Error")
    axs[i].set_title(f"{dataset} dataset: SSE per Distance for K")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()

# %%
# Plotting K means k selection based on elbow method with all seeds ----- UNUSED
fig, axs = plt.subplots(1, len(results_k_means), figsize=(len(results_k_means)*6, 6))
for i, (dataset, df) in enumerate(results_k_means.items()):
    grouped = df.groupby('seed')
    for seed, group in grouped:
        axs[i].plot(group['k'].values, group['sum_of_squared_error'].values, label=f"Seed {seed}")
    axs[i].set_xlabel("k")
    axs[i].set_ylabel("Sum of Squared Error")
    axs[i].set_title(f"{dataset} dataset: K vs Sum of Squared Error for Different Seeds")
    axs[i].legend()
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# %%
# Plotting K-means distance selection based on selected k value
best_k = {"cmc": 4, "sick": 4, "mushroom": 5}
fig, axs = plt.subplots(4, len(results_k_means), figsize=(len(results_k_means)*3.5, 7))
for i, (dataset, df) in enumerate(results_k_means.items()):
    selected_rows = df[df['k'] == best_k[dataset]]
    values = pd.DataFrame(columns=df.columns)
    grouped = selected_rows.groupby('distance')
    for distance, group in grouped:
        for k_val, k_group in group.groupby('k'):
            values = pd.concat([values,k_group[k_group['seed'] == best_seed_for_k[dataset][k_val]]])

    axs[0,i].scatter(values['distance'].values, values['silhouette'].values)
    axs[1,i].scatter(values['distance'].values, values['davies_bouldin'].values)
    axs[2,i].scatter(values['distance'].values, values['adjusted_rand_score'].values)
    values['v_measure'] = [[float(f) for f in float_pattern.findall(v)][2] for v in values['homogeneity_completeness_v_measure'].values]
    axs[3,i].scatter(values['distance'].values, values['v_measure'].values)
    # print(values[['distance', 'silhouette', 'davies_bouldin', 'adjusted_rand_score', 'v_measure']])

    axs[0,i].set_ylabel("Silhouette")
    axs[0,i].set_title(f"{dataset}: Silhouette score for distances")
    axs[1,i].set_ylabel("Davies Bouldin score")
    axs[1,i].set_title(f"{dataset}: Davies Bouldin score for distances")
    axs[2,i].set_ylabel("Adjusted Rand score")
    axs[2,i].set_title(f"{dataset}: Adjusted Rand score for distances")
    axs[3,i].set_ylabel("V-Measure")
    axs[3,i].set_title(f"{dataset}: V-Measure for distances")
    for j in range(4):
        axs[j,i].set_xlabel("Distance")
        axs[j,i].grid(True)
plt.tight_layout()
plt.show()

# Best distance:
# - cmc - L2 (euclidean)
# - sick - cosine
# - mushroom - L1 (manhattan)

# %%
# Plotting K-means distance selection based on selected k value ----- UNUSED
best_k = {"cmc": 4, "sick": 4, "mushroom": 5}
fig, axs = plt.subplots(2, len(results_k_means), figsize=(len(results_k_means)*5, 5))
for i, (dataset, df) in enumerate(results_k_means.items()):
    selected_rows = df[df['k'] == best_k[dataset]]
    grouped = selected_rows.groupby('distance')
    for distance, group in grouped:
        mean_silhouette = group['silhouette'].mean()
        std_silhouette = group['silhouette'].std()
        axs[0,i].errorbar(distance, mean_silhouette, yerr=std_silhouette, fmt='o', label=f"{distance} Silhouette")
        mean_davies = group['davies_bouldin'].mean()
        std_davies = group['davies_bouldin'].std()
        axs[1,i].errorbar(distance, mean_davies, yerr=std_davies, fmt='o', label=f"{distance} Davies Bouldin")

    axs[0,i].set_ylabel("Silhouette score")
    axs[0,i].set_title(f"{dataset} dataset: Mean Silhouette by Distance")
    axs[1,i].set_ylabel("Davies Bouldin score")
    axs[1,i].set_title(f"{dataset} dataset: Mean Davies Bouldin by Distance")
    for j in range(2):
        axs[j,i].set_xlabel("Distance")
        axs[j,i].grid(True)
plt.tight_layout()
plt.show()

# %%
# Plotting K-means seed selection based on SSE ----- UNUSED
best_distance = {"cmc":"cosine", "sick": "L1", "mushroom": "cosine"}
fig, axs = plt.subplots(1, len(results_k_means), figsize=(len(results_k_means)*5, 5))
for i, (dataset, df) in enumerate(results_k_means.items()):
    rows = df[df['k'] == best_k[dataset]]
    grouped = rows.groupby('distance')
    for distance, group in grouped:
        axs[i].plot(group['seed'].values, group['sum_of_squared_error'].values, label=f"{distance} distance")
    axs[i].set_xlabel("Seed")
    axs[i].set_ylabel("Sum of Squared Error")
    axs[i].set_title(f"{dataset} dataset: SSE for different Seeds and Distances")
    axs[i].legend()
    axs[i].grid(True)

    table_values = [np.round(group['sum_of_squared_error'].values, 0).astype(int) for d, group in grouped]
    table = axs[i].table(cellText=table_values,
                         colLabels=rows['seed'].unique(),
                         rowLabels=rows["distance"].unique(),
                         loc='bottom',
                         cellLoc='center',
                         bbox=[0.0, -0.35, 1.0, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(7)

plt.tight_layout()
plt.show()


# %%
# Plotting G means
fig, axs = plt.subplots(4, len(results_k_means), figsize=(len(results_k_means)*4.5, 7))
for i, (dataset, df) in enumerate(results_g_means.items()):
    # remove the seed, since no matter the seed, g-means produces the same result
    df = df[df["seed"] == 20.0].drop("seed", axis=1)
    axs[0,i].scatter(df['distance'].values, df['silhouette'].values)
    axs[1,i].scatter(df['distance'].values, df['davies_bouldin'].values)
    axs[2,i].scatter(df['distance'].values, df['adjusted_rand_score'].values)
    df['v_measure'] = [[float(f) for f in float_pattern.findall(v)][2] for v in df['homogeneity_completeness_v_measure'].values]
    axs[3,i].scatter(df['distance'].values, df['v_measure'].values)
    print(df[['distance', 'silhouette', 'davies_bouldin', 'adjusted_rand_score', 'v_measure']])

    axs[0,i].set_ylabel("Silhouette")
    axs[0,i].set_title(f"{dataset}: Silhouette score for distances")
    axs[1,i].set_ylabel("Davies Bouldin score")
    axs[1,i].set_title(f"{dataset}: Davies Bouldin score for distances")
    axs[2,i].set_ylabel("Adjusted Rand score")
    axs[2,i].set_title(f"{dataset}: Adjusted Rand score for distances")
    axs[3,i].set_ylabel("V-Measure")
    axs[3,i].set_title(f"{dataset}: V-Measure for distances")
    for j in range(4):
        axs[j,i].set_xlabel("Distance")
        axs[j,i].grid(True)

plt.tight_layout()
plt.show()

# Best algorithm:
# - cmc:
#   - for best clustering: Euclidean distance
#   - for best label matching: Cosine distance
# - sick - cosine
# - mushroom - L1 (manhattan)

# %%
for i, (dataset, df) in enumerate(results_g_means.items()):
    df = df[df["seed"] == 20.0].drop("seed", axis=1)
    plt.scatter(df['distance'].values, df['k'].values, label=dataset)

plt.legend()
plt.grid(True)
plt.xlabel("Distance")
plt.ylabel("K")
plt.title(f"K vs Distance for G-Means across all datasets")
plt.show()


# %%

"""
For K-means, best algorithms are:
- cmc - k4_L2_seed60
- sick - k4_L1_seed60
- mushroom - k5_cosine_seed40

For G-means, best algorithms are:
- cmc - L2_seed20
- sick - cosine_seed20
- mushroom - L1_seed20
"""