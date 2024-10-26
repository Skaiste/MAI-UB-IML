import pathlib
import json
import argparse
import numpy as np
import pandas as pd


curr_dir = pathlib.Path(__file__).parent

def load_results(res_dir):
    if not res_dir.is_dir():
        raise Exception(f"Result directory {res_dir} does not exist")
    
    results = {'KNN':{}, 'SVM':{}}
    for fn in res_dir.iterdir():
        with open(fn, 'r') as f:
            data = json.load(f)
        if data['model'] == 'KNN':
            # k number
            k = int(fn.name[1])
            name = ''.join(fn.stem.split('_')[1:])
            if name not in results['KNN']:
                results['KNN'][name] = {
                    'distance': data['distance'],
                    'voting': data['voting'],
                    'weighting': data['weighting'],
                    'k': {}
                }
            results['KNN'][name]['k'][k] = {
                'accuracy': {f:data['folds'][f]['accuracy'] for f in data['folds']},
                'correct': {f:data['folds'][f]['correct'] for f in data['folds']},
                'incorrect': {f:data['folds'][f]['incorrect'] for f in data['folds']},
                'pred_time': {f:data['folds'][f]['pred_time'] for f in data['folds']}
            }
        else:
            results['SVM'][data['kernel']] = {
                'accuracy': {f:data['folds'][f]['accuracy'] for f in data['folds']},
                'correct': {f:data['folds'][f]['correct'] for f in data['folds']},
                'incorrect': {f:data['folds'][f]['incorrect'] for f in data['folds']},
                'pred_time': {f:data['folds'][f]['pred_time'] for f in data['folds']}
            }
    return results


def calculate_averages(results):
    for alg in results['KNN']:
        for k in results['KNN'][alg]['k']:
            results['KNN'][alg]['k'][k]['accuracy_mean'] = np.mean(list(results['KNN'][alg]['k'][k]['accuracy'].values()))
            results['KNN'][alg]['k'][k]['pred_time_mean'] = np.mean(list(results['KNN'][alg]['k'][k]['pred_time'].values()))

    for alg in results['SVM']:
        results['SVM'][alg]['accuracy_mean'] = np.mean(list(results['SVM'][alg]['accuracy'].values()))
        results['SVM'][alg]['pred_time_mean'] = np.mean(list(results['SVM'][alg]['pred_time'].values()))


def sort_and_prepare_results(results):
    # Prepare KNN table
    knn_data = []
    for name, knn_results in results['KNN'].items():
        row = {
            ('Distance', ''): knn_results['distance'],
            ('Voting', ''): knn_results['voting'],
            ('Weighting', ''): knn_results['weighting']
        }
        score = []
        for k in [1, 3, 5, 7]:
            row[(f'Accuracy', f'K{k}')] = knn_results['k'].get(k, {}).get('accuracy_mean', np.nan)
            row[(f'Pred. Time', f'K{k}')] = knn_results['k'].get(k, {}).get('pred_time_mean', np.nan)
            score.append(row[(f'Accuracy', f'K{k}')] / row[(f'Pred. Time', f'K{k}')] / 100)

        row[('Score','')] = max(score)
        knn_data.append(row)

    # Create KNN DataFrame and sort by K1 accuracy mean
    knn_df = pd.DataFrame(knn_data)
    knn_df.columns = pd.MultiIndex.from_tuples(knn_df.columns)
    knn_df = knn_df.sort_values(by=('Score', ''), ascending=False).drop(columns=[('Score','')])

    
    # Prepare SVM table
    svm_data = []
    for kernel, svm_results in results['SVM'].items():
        svm_data.append({
            'Kernel': kernel,
            'Accuracy': svm_results.get('accuracy_mean', np.nan),
            'Prediction Time': svm_results.get('pred_time_mean', np.nan)
        })

    # Create SVM DataFrame and sort by accuracy mean
    svm_df = pd.DataFrame(svm_data).sort_values(by='Accuracy', ascending=False)

    # Display both tables
    # print("KNN Sorted Results")
    # print(knn_df.to_string(index=False))
    # print("\nSVM Sorted Results")
    # print(svm_df.to_string(index=False))
    return knn_df, svm_df


def dataframe_to_markdown(df):
    header = "| " + " | ".join([c[0] for c in df.columns]) + " |"
    separator = "|---" * len(df.columns) + "|"
    rows = ["| " + " | ".join([c[1] for c in df.columns]) + " |"]
    float_format = "{:.5f}"
    rows += [
        "| " + " | ".join(
            [float_format.format(value) if isinstance(value, float) else str(value) for value in row]
        ) + " |"
        for row in df.values
    ]
    markdown_table = "\n".join([header, separator] + rows)
    markdown_table = markdown_table.replace("  |", "|")
    return markdown_table


def main():
    parser = argparse.ArgumentParser(description="Runs an algorithm once on provided data")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="sick",
        help="Name of the dataset to process, adult on default."
    )
    parser.add_argument(
        "-r", "--result-directory",
        type=pathlib.Path,
        default=curr_dir / "results",
        help="Path to the results directory."
    )
    args = parser.parse_args()

    args.result_directory = args.result_directory / args.dataset

    results = load_results(args.result_directory)
    calculate_averages(results)

    knn, svm = sort_and_prepare_results(results)
    print(dataframe_to_markdown(knn))

    # breakpoint()
    # i=0

if __name__ == "__main__":
    main()
