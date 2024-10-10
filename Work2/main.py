import argparse
import pathlib
import scipy

curr_dir = pathlib.Path(__file__).parent

def loadarff(f):
    # TODO: load arff data file
    pass

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="adult",
        help="Name of the dataset to process, adult on default."
    )
    parser.add_argument(
        "-p", "--path",
        type=pathlib.Path,
        default=curr_dir / "datasets",
        help="Path to the data directory."
    )
    parser.add_argument(
        "-t", "--test",
        action='store_true',
        default=False,
        help="Boolean flag to indicate whether to test the model (default: False)."
    )
    
    # Parse the command line arguments
    args = parser.parse_args()

    data_dir = args.path / args.dataset
    if not data_dir.is_dir():
        raise argparse.ArgumentTypeError(f"The dataset directory {data_dir} could not be found.")
    
    arrf_train_data = []
    arrf_test_data = []
    for fn in data_dir.iterdir():
        if not fn.is_file():
            continue
        if '.train' in fn.suffixes:
            arrf_train_data.append(loadarff(fn))
            print(f"Loaded training data file {fn}")
        elif '.test' in fn.suffixes and args.test:
            arrf_test_data.append(loadarff(fn))
            print(f"Loaded testing data file {fn}")

if __name__ == "__main__":
    main()