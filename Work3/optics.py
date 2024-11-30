import os
import sys
import math
import random
import pathlib
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
import argparse
try:
    curr_dir = pathlib.Path(__file__).parent
except:
    curr_dir = pathlib.Path(os.getcwd()) / "Work3"
sys.path.append(str(curr_dir))

from K_Means import load_data

def fit_optics(data,distance, algorithm_):
    model = OPTICS(metric=distance, algorithm=algorithm_)
    model.fit(data)
    return model


def main():
    parser = argparse.ArgumentParser(description="Runs an algorithm once on provided data")

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

    distances = ['euclidean', 'cosine', 'l1']
    algorithms = ['auto', 'kd_tree']
    for distance in distances:
        for algorithm in algorithms:
            print(f" OPTICS with {distance} and {algorithm}")
            model = fit_optics(input, distance, algorithm)
            labels = model.labels_
            
if __name__ == "__main__":
    main()
