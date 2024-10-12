import argparse
import pathlib

from parser import get_data


curr_dir = pathlib.Path(__file__).parent

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
    
    training_fns = []
    testing_fns = []
    for fn in data_dir.iterdir():
        if not fn.is_file():
            continue
        if '.train' in fn.suffixes:
            training_fns.append(fn)
        elif '.test' in fn.suffixes and args.test:
            testing_fns.append(fn)

    (train_input, train_output), (test_input, test_output) = get_data(training_fns, testing_fns)


if __name__ == "__main__":
    main()