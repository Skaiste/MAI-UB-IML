import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, KMeans
from sklearn.preprocessing import StandardScaler
from data_parser import get_data


try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
sys.path.append(str(curr_dir))

from data_parser import get_data

data_dir = curr_dir / "datasets"
dataset_name = "cmc"

cache_dir = curr_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

dataset = data_dir / f"{dataset_name}.arff"
if not dataset.is_file():
    raise Exception(f"Dataset {dataset} could not be found.")

normalise_nominal = True if dataset_name != "cmc" else False
train_input, train_output, test_input, test_output = get_data(dataset, cache_dir=cache_dir, cache=False, normalise_nominal=normalise_nominal)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_input)
# %%


def spectral_clustering(X, min_samples=5, min_cluster_size=0.1, metric="minkowski", xi=0.05):
 
    optics_model = OPTICS(min_samples=min_samples, min_cluster_size=min_cluster_size, metric=metric, xi=xi)
    optics_model.fit(X)
    return optics_model.labels_, optics_model.reachability_, optics_model.ordering_


print("OPTICS Clustering")
labels, reachability, ordering = spectral_clustering(
    scaled_data, min_samples=10, min_cluster_size=0.05, metric="euclidean", xi=0.05)

    # Visualization: Reachability plot
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(reachability)), reachability[ordering], 'b.', alpha=0.7)
plt.title('Reachability Plot')
plt.xlabel('Sample Index')
plt.ylabel('Reachability Distance')
plt.show()

    # Visualization: OPTICS clustering results
plt.figure(figsize=(10, 5))
colors = ['r.', 'g.', 'b.', 'y.', 'c.']

for cluster_label in set(labels):
     if cluster_label == -1:  # Noise
            color = 'k.'
else:
            color = colors[cluster_label % len(colors)]

cluster_member_mask = (labels == cluster_label)
plt.plot(
            scaled_data[cluster_member_mask, 0],
            scaled_data[cluster_member_mask, 1],
            color,
            alpha=0.7
        )

plt.title('OPTICS Clustering')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.show()