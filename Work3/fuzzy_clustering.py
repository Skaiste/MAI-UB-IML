import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_parser import get_data
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
sys.path.append(str(curr_dir))

data_dir = curr_dir / "datasets"
dataset_name = "cmc"

cache_dir = curr_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

dataset = data_dir / f"{dataset_name}.arff"
if not dataset.is_file():
    raise Exception(f"Dataset {dataset} could not be found.")

normalise_nominal = True if dataset_name != "cmc" else False
train_input, train_output, test_input, test_output = get_data(
    dataset, cache_dir=cache_dir, cache=False, normalise_nominal=normalise_nominal
)

# reprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_input)

# %%
def matrix(n_samples, n_clusters):

    membership_matrix = np.random.rand(n_samples, n_clusters)
    membership_matrix = membership_matrix / np.sum(membership_matrix, axis=1, keepdims=True)     # each row sums to 1
    return membership_matrix

def cluster_centers(X, U, m):
 
    U_m = U ** m  # fuzzify the values
    cluster_centers = (U_m.T @ X) / np.sum(U_m.T, axis=1, keepdims=True)
    return cluster_centers

def update_matrix(X, centers, m):  # udate matrix based on clister centers
  
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)  # Calculate distances to cluster centers
    distances = np.fmax(distances, np.finfo(np.float64).eps)  # Avoid division by zero
    reciprocal_distances = 1.0 / distances
    U_new = reciprocal_distances / np.sum(reciprocal_distances, axis=1, keepdims=True)  # Normalize
    return U_new

def fuzzy_c_means(X, n_clusters=3, m=2.0, max_iter=150, error=1e-5):  # fuzzy c-means

    n_samples, n_features = X.shape
    U = matrix(n_samples, n_clusters)
    for iteration in range(max_iter):
        centers = cluster_centers(X, U, m)
        U_new = update_matrix(X, centers, m)
        if np.linalg.norm(U_new - U) < error:   # check for convergence
            break
        U = U_new
    return centers, U


print("Fuzzy C-Means Clustering")
n_clusters = 3
centers, membership_matrix = fuzzy_c_means(scaled_data, n_clusters=n_clusters, m=2.0)

# clusters based on maximum membership / cluster crips
labels = np.argmax(membership_matrix, axis=1)

# evaluate crisp clusters
silhouette_avg = silhouette_score(scaled_data, labels)
davies_bouldin = davies_bouldin_score(scaled_data, labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Index: {davies_bouldin}")

plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap="viridis", s=50)
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centers")
plt.title("Fuzzy C-Means Clustering")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.legend()
plt.show()
print("Membership matrix (first 4 samples):")
print(membership_matrix[:4])
