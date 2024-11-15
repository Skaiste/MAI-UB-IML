import argparse
import pathlib
import json
import time
import numpy as np
from enum import Enum


from data_parser import get_data
from kNN import KNN, DistanceType, VotingSchemas, WeigthingStrategies, ReductionMethod
from SVM import SVM, KernelType


curr_dir = pathlib.Path(__file__).parent


class ModelTypes(Enum):
    KNN = 'KNN'
    SVM = 'SVM'


def run_knn(train_input, train_output, test_input, test_output, args, skip_if_exists=False):
    shorten_name = lambda n: ''.join([p[:2].capitalize() for p in n.split('_')])
    results_fn = f"k{args.k}_{shorten_name(args.distance_type.value)}"
    if args.distance_type == DistanceType.MINKOWSKI:
        results_fn += str(args.minkowski_r)
    results_fn += f"_{shorten_name(args.voting_schema.value)}"
    results_fn += f"_{shorten_name(args.weighting_strategy.value)}"
    if args.instance_reduction != ReductionMethod.NO_REDUCTION:
        results_fn += f"_{args.instance_reduction.value}"
    results_fn += ".json"
    results_path = args.result_directory / results_fn
    if skip_if_exists and results_path.is_file():
        return

    result = {
        'model': 'KNN',
        'k': args.k,
        'folds': {i:{} for i in range(len(train_input))},
        'dataset': args.dataset,
        'distance': args.distance_type.value,
        'voting': args.voting_schema.value,
        'weighting': args.weighting_strategy.value,
        'reduction': args.instance_reduction.value
    }
    if args.distance_type == DistanceType.MINKOWSKI:
        result['minkowski_r'] = args.minkowski_r

    for fold in range(len(train_input)):
        knn = KNN(k=args.k, dm=args.distance_type, vs=args.voting_schema, ws=args.weighting_strategy, rm=args.instance_reduction, r=args.minkowski_r)

        # load cached weights
        wcdir = args.cache_directory / "weighted"
        weighted_fn = wcdir / f"fold.{fold}.{args.weighting_strategy.value}.weights.{args.instance_reduction.value}.npy"
        cached_weights = None
        if not args.disable_cache and weighted_fn.is_file():
            cached_weights = np.load(weighted_fn)

        knn.fit(train_input[fold], train_output[fold], cached_weights)

        start_time = time.time()
        predictions = knn.predict(test_input[fold])
        end_time = time.time()

        matches = test_output[fold] == predictions.reindex(test_output[fold].index)
        result_counts = matches.value_counts()
        accuracy = result_counts[True] / matches.count() * 100
        print(f"Fold {fold}: Accuracy: {accuracy:.2f}%")

        result['folds'][fold]['accuracy'] = accuracy
        result['folds'][fold]['storage'] = knn.train_input.shape[0] / train_input[fold].shape[0] * 100
        result['folds'][fold]['correct'] = int(result_counts[True] if True in result_counts else 0)
        result['folds'][fold]['incorrect'] = int(result_counts[False] if False in result_counts else 0)
        result['folds'][fold]['predictions'] = list([s for s in predictions])
        result['folds'][fold]['pred_time'] = end_time - start_time

        # cache weighted input set
        if not args.disable_cache:
            wcdir.mkdir(parents=True, exist_ok=True)
            np.save(weighted_fn, knn.feature_weights)


    with open(results_path, "w") as f:
        json.dump(result, f, indent=4)


def run_svm(train_input, train_output, test_input, test_output, args, skip_if_exists=False):
    results_fn = f"SVM_{args.kernel.value}"
    if args.instance_reduction != ReductionMethod.NO_REDUCTION:
        results_fn += f"_{args.instance_reduction.value}"
    results_fn += ".json"
    results_path = args.result_directory / results_fn
    if skip_if_exists and results_path.is_file():
        return
        
    result = {
        'model': 'SVM',
        'folds': {i:{} for i in range(len(train_input))},
        'dataset': args.dataset,
        'kernel': args.kernel.value,
        'reduction': args.instance_reduction.value
    }
    for fold in range(len(train_input)):
        svm = SVM(args.kernel, reduction=args.instance_reduction)
        svm.fit(train_input[fold], train_output[fold])

        start_time = time.time()
        predictions = svm.predict(test_input[fold])
        end_time = time.time()

        matches = test_output[fold] == predictions
        result_counts = matches.value_counts()
        accuracy = result_counts[True] / matches.count() * 100
        print(f"Fold {fold}: Accuracy: {accuracy:.2f}%")

        result['folds'][fold]['accuracy'] = accuracy
        result['folds'][fold]['storage'] = svm.train_input.shape[0] / train_input[fold].shape[0] * 100
        result['folds'][fold]['correct'] = int(result_counts[True] if True in result_counts else 0)
        result['folds'][fold]['incorrect'] = int(result_counts[False] if False in result_counts else 0)
        result['folds'][fold]['predictions'] = list([s for s in predictions])
        result['folds'][fold]['pred_time'] = end_time - start_time

    with open(results_path, "w") as f:
        json.dump(result, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Runs an algorithm once on provided data")
    parser.add_argument(
        "-k",
        type=int,
        default=1,
        help="k number of neighbours to use in k-NN algorithm"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="sick",
        help="Name of the dataset to process, sick on default."
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
    parser.add_argument(
        "-t", "--distance-type",
        choices=[dt.value for dt in DistanceType],
        default=DistanceType.EUCLIDEAN.value,
        help=f"KNN: Function for calculating distance, default: {DistanceType.EUCLIDEAN.value}."
    )
    parser.add_argument(
        "-v", "--voting-schema",
        choices=[vs.value for vs in VotingSchemas],
        default=VotingSchemas.MAJORITY_CLASS.value,
        help=f"KNN: Voting schema for selecting neighbors, default: {VotingSchemas.MAJORITY_CLASS.value}."
    )
    parser.add_argument(
        "-R", "--minkowski-r",
        type=int,
        default=3,
        help="KNN: r value for minkowski algorithm, default: 3"
    )
    parser.add_argument(
        "-w", "--weighting-strategy",
        choices=[ws.value for ws in WeigthingStrategies],
        default=WeigthingStrategies.EQUAL.value,
        help=f"KNN: Weighting scheme for scaling neighbors, default: {WeigthingStrategies.EQUAL.value}."
    )
    parser.add_argument(
        "-i", "--instance-reduction",
        choices=[rm.value for rm in ReductionMethod],
        default=ReductionMethod.NO_REDUCTION.value,
        help=f"Instance reduction method, default: {ReductionMethod.NO_REDUCTION.value}."
    )
    parser.add_argument(
        "-m", "--model",
        choices=[m.value for m in ModelTypes],
        default=ModelTypes.KNN.value,
        help="Choose model to run, KNN set on default"
    )
    parser.add_argument(
        "-K", "--kernel",
        choices=[k.value for k in KernelType],
        default=KernelType.POLYNOMIAL.value,
        help="SVM: Choose classifier kernel, polynomial set on default"
    )
    
    # Parse the command line arguments
    args = parser.parse_args()

    args.distance_type = DistanceType(args.distance_type.lower())
    args.voting_schema = VotingSchemas(args.voting_schema.lower())
    args.weighting_strategy = WeigthingStrategies(args.weighting_strategy.lower())
    args.kernel = KernelType(args.kernel.lower())
    args.instance_reduction = ReductionMethod(args.instance_reduction.lower())

    if args.instance_reduction != ReductionMethod.NO_REDUCTION:
        args.result_directory = pathlib.Path(str(args.result_directory) + "_reduced")
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
        if fold > int(args.limit):
            continue
        if fold not in data_fns:
            data_fns[fold] = {}
        if '.train' in fn.suffixes:
            data_fns[fold]['training'] = fn
        elif '.test' in fn.suffixes:
            data_fns[fold]['testing'] = fn

    train_input, train_output, test_input, test_output = get_data(
        data_fns, 
        not args.disable_cache, 
        args.cache_directory
    )

    if ModelTypes(args.model) == ModelTypes.KNN:
        run_knn(train_input, train_output, test_input, test_output, args)
    else: #SVM
        run_svm(train_input, train_output, test_input, test_output, args)



if __name__ == "__main__":
    main()