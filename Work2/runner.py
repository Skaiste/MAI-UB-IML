import argparse
import pathlib

from main import run_knn, run_svm
from kNN import DistanceType, VotingSchemas, WeigthingStrategies, ReductionMethod
from SVM import KernelType
from data_parser import get_data

curr_dir = pathlib.Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Runs all variations of the algorithms for result collection")
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
        "-r", "--result-directory",
        type=pathlib.Path,
        default=curr_dir / "results",
        help="Path to the results directory."
    )
    parser.add_argument(
        "-c", "--cache-directory",
        type=pathlib.Path,
        default=curr_dir / "cache",
        help="Path to the cache directory for normalised values."
    )
    parser.add_argument(
        "--disable-cache",
        action='store_true',
        default=False,
        help="Disable saving to and loading normalised data from cache"
    )
    args = parser.parse_args()
    
    args.result_directory = args.result_directory / args.dataset
    args.result_directory.mkdir(parents=True, exist_ok=True)

    if not args.disable_cache:
        args.cache_directory = args.cache_directory / args.dataset
        args.cache_directory.mkdir(parents=True, exist_ok=True)

    data_dir = args.path / args.dataset
    if not data_dir.is_dir():
        raise argparse.ArgumentTypeError(f"The dataset directory {data_dir} could not be found.")
    
    data_fns = {}
    for fn in data_dir.iterdir():
        if not fn.is_file():
            continue
        fold = int(fn.suffixes[1][1:])
        if fold not in data_fns:
            data_fns[fold] = {}
        if '.train' in fn.suffixes:
            data_fns[fold]['training'] = fn
        elif '.test' in fn.suffixes:
            data_fns[fold]['testing'] = fn
   
    train_input, train_output, test_input, test_output = get_data(data_fns, not args.disable_cache, args.cache_directory)

    args.minkowski_r = 1 # default value, need to set it even when minkowski isn't used for distance
    args.instance_reduction = ReductionMethod.NO_REDUCTION

    # run KNN models
    # itertools
    possible_k = [1, 3, 5, 7]
    for dt in DistanceType:
        args.distance_type = dt
        for vs in VotingSchemas:
            args.voting_schema = vs
            for ws in WeigthingStrategies:
                args.weighting_strategy = ws
                for k in possible_k:
                    args.k = k
                    if dt == DistanceType.MINKOWSKI:
                        for r in [3, 4]:
                            args.minkowski_r = r
                            print(f"Running KNN with: k = {k}, distance = {dt.value} where r = {r}, voting = {vs.value}, weights = {ws.value}")
                            run_knn(train_input, train_output, test_input, test_output, args, skip_if_exists=True)
                    else:
                        print(f"Running KNN with: k = {k}, distance = {dt.value}, voting = {vs.value}, weights = {ws.value}")
                        run_knn(train_input, train_output, test_input, test_output, args, skip_if_exists=True)

    # run SVM models
    for kt in KernelType:
        args.kernel = kt
        print(f"Running SVM with kernel = {kt.value}")
        run_svm(train_input, train_output, test_input, test_output, args, skip_if_exists=True)


if __name__ == "__main__":
    main()