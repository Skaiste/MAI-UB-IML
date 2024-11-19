import pathlib
from data_parser import get_data

data_dir = pathlib.Path(__file__).parent / "datasets"
dataset_name = "cmc"

cache_dir = pathlib.Path(__file__).parent / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

dataset = data_dir / f"{dataset_name}.arff"
if not dataset.is_file():
    raise Exception(f"Dataset {dataset} could not be found.")

print("Loading data")
train_input, train_output, test_input, test_output = get_data(dataset, cache_dir=cache_dir)

