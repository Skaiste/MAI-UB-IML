import argparse
import pathlib
from sklearn.metrics import accuracy_score

from data_parser import get_data
from kNN import kNN, DistanceType, VotingSchemas, WeigthingStrategies


curr_dir = pathlib.Path(__file__).parent

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="sick",
        help="Name of the dataset to process, adult on default."
    )
    parser.add_argument(
        "-p", "--path",
        type=pathlib.Path,
        default=curr_dir / "datasets",
        help="Path to the data directory."
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=10,
        help="Limit to how many datasets to load"
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
        if int(fn.suffixes[1][1:]) > int(args.limit):
            continue
        if '.train' in fn.suffixes:
            training_fns.append(fn)
        elif '.test' in fn.suffixes:
            testing_fns.append(fn)

    train_input, train_output, test_input, test_output = get_data(training_fns, testing_fns)


    knn = kNN(k=100, dm=DistanceType.MINKOWSKI, vs=VotingSchemas.INVERSE_DISTANCE, ws=WeigthingStrategies.EQUAL)
    knn.fit(train_input, train_output)
    for fold in range(len(train_input)):
        predictions = knn.predict(test_input[fold], fold)
        matches = test_output[fold] == predictions.reindex(test_output[fold].index)
        result_counts = matches.value_counts()
        accuracy = result_counts[True] / matches.count() * 100
        # accuracy = accuracy_score(test_output[fold], predictions)
        print(f"Fold {fold}: Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()