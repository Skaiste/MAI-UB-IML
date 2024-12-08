import os
import re
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame

try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
sys.path.append(str(curr_dir))

results_dir = curr_dir / "results"

float_pattern = re.compile(r"0\.\d+")

results_names = {
    "mushroom":{
        "kmeans": "k5_cosine_seed40",
        "gmeans": "L1",
        "spectral": {"k": 8, "affinity": "rbf_gamma0.1"},
        "optics": "l1_auto_15_9",
        "kmeans++": 3,
        "fuzzy": "fuzzy_k2_m2.5"
    },
    "sick": {
        "kmeans": "k4_L1_seed60",
        "gmeans": "cosine",
        "spectral": {"k": 2, "affinity": "rbf_gamma0.1"},
        "optics": "l1_auto_15_1",
        "kmeans++": 8,
        "fuzzy": "fuzzy_k2_m1.5"
    },
    "cmc": {
        "kmeans": "k4_L2_seed60",
        "gmeans": "L2",
        "spectral": {"k": 8, "affinity": "rbf_gamma0.1"},
        "optics": "euclidean_brute_25_1",
        "kmeans++": 6,
        "fuzzy": "fuzzy_k2_m1.5"
    }
}

# %%
results_dfs = {dn:{} for dn in results_names}
for dn, algorithms in results_names.items():
    for alg in algorithms:
        if algorithms[alg] == "" or not (results_dir / f"{dn}_{alg}_results.csv").exists():
            continue
        results_dfs[dn][alg] = pd.read_csv(results_dir / f"{dn}_{alg}_results.csv")

# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
results = {dn:pd.DataFrame() for dn in results_names}
for dn, algorithms in results_dfs.items():
    for alg in algorithms:
        col_name = "k" if alg == "kmeans++" else ("Algorithm" if alg == "optics" else "Unnamed: 0")
        if not isinstance(results_names[dn][alg], dict):
            row = algorithms[alg][algorithms[alg][col_name] == results_names[dn][alg]]
        else:
            row = algorithms[alg]
            for k,v in results_names[dn][alg].items():
                row = row[row[k] == v]
        row.reset_index(drop=True, inplace=True)
        selection = pd.DataFrame(row[["silhouette","davies_bouldin","adjusted_rand_score"]])
        if "homogeneity_completeness_v_measure" in row.columns:
            selection['v_measure'] = [[float(f) for f in float_pattern.findall(v)][2] for v in
                               row['homogeneity_completeness_v_measure'].values]
        elif "v_measure" in algorithms[alg].columns:
            selection['v_measure'] = row["v_measure"]
        else:
            raise Exception("homogeneity_completeness_v_measure column not found")
        selection.loc[0,"name"] = alg
        results[dn] = pd.concat([results[dn], selection], axis=0, ignore_index=True)

# %%
fig, axs = plt.subplots(4, len(results), figsize=(len(results)*4.5, 7))
for i, (dataset, df) in enumerate(results.items()):
    axs[0,i].scatter(df['name'].values, df['silhouette'].values)
    axs[1,i].scatter(df['name'].values, df['davies_bouldin'].values)
    axs[2,i].scatter(df['name'].values, df['adjusted_rand_score'].values)
    axs[3,i].scatter(df['name'].values, df['v_measure'].values)
    print(dataset)
    print(df[['name', 'silhouette', 'davies_bouldin', 'adjusted_rand_score', 'v_measure']])

    axs[0,i].set_ylabel("Silhouette")
    axs[0,i].set_title(f"{dataset}: Silhouette score for algorithms")
    axs[1,i].set_ylabel("Davies Bouldin score")
    axs[1,i].set_title(f"{dataset}: Davies Bouldin score for algorithms")
    axs[2,i].set_ylabel("Adjusted Rand score")
    axs[2,i].set_title(f"{dataset}: Adjusted Rand score for algorithms")
    axs[3,i].set_ylabel("HC score")
    axs[3,i].set_title(f"{dataset}: HC score for algorithms")
    for j in range(4):
        axs[j,i].set_xlabel("Distance")
        axs[j,i].grid(True)

plt.tight_layout()
plt.show()