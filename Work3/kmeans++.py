import os
import sys
import pathlib
import numpy as np
import pandas as pd
import argparse

try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
sys.path.append(str(curr_dir))


from K_Means import load_data,  kmeans_fit, minkowski_distance, cosine_distance


def create_centroids(data,k,distance):
    indices = np.array([np.random.choice(len(data))])
    centroids = pd.DataFrame(data.iloc[indices])
    while len(indices)<k:

        distances = np.array([distance(row, centroids).min()  for index, row  in data.iterrows()])
        probabilities = distances**2/(np.sum(distances**2))
        index = np.random.choice(data.index,p = probabilities)

        if index not in indices:
            indices = np.append(indices, index)

            centroids = pd.DataFrame(data.iloc[indices])

    return np.array(centroids)

def kmeans_plus_plus_fit(X, k,distance,max_iter = 100):

    centroids = create_centroids(X, k,distance)
    return kmeans_fit(X, k,distance,max_iter, centroids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs an algorithm once on provided data")
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="k number of clusters in kmeans++"
    )
    parser.add_argument(
        "-distance_type",
        type=str,
        default="minkowski_3",
        help="distance type"
    )
    parser.add_argument(
        "-dataset_name",
        type=str,
        default="mushroom",
        help="Dataset's file's name"
    )


    args = parser.parse_args()
    data_dir = curr_dir / "datasets"
    dataset_name = args.dataset_name
    cache_dir = curr_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    input, output = load_data(data_dir, dataset_name, cache=False, cache_dir=cache_dir)
    np.random.seed(42)
    if args.distance_type == "cosine":
        distance = cosine_distance
    elif args.distance_type[:10] == "minkowski_":
        distance = lambda x,y: minkowski_distance(x,y,r=int(args.distance_type[10:]))
    else:

        raise ValueError("Non valid distance metric")
    result = kmeans_plus_plus_fit(input,4,distance)



