import matplotlib.pyplot as plt
import numpy as np

# Sample data for three datasets
datasets = ['mushroom dataset', 'sick dataset', 'cmc dataset']
k_values = np.arange(2, 10)

# Generate random data for SSE (Sum of Squared Error) for each dataset
np.random.seed(42)  # For reproducibility
sse_data = {
    dataset: np.sort(np.random.randint(500, 40000, len(k_values)))[::-1]
    for dataset in datasets
}

# Set up subplots
fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5), sharey=True)

for idx, dataset in enumerate(datasets):
    ax = axes[idx]

    # Plot SSE vs. K
    ax.plot(k_values, sse_data[dataset], label=dataset, marker='o', linestyle='-')



    ax.set_title(f'{dataset}: SSE for K')
    ax.set_xlabel('K')
    if idx == 0:
        ax.set_ylabel('Sum of Squared Error')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
fig.savefig("plots/kmeans++_elbow.png", dpi=300)
plt.close(fig)