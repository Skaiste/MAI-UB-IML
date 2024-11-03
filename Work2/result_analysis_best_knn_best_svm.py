import pathlib
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import studentized_range

curr_dir = pathlib.Path(__file__).parent


def load_results(res_dir):
    if not res_dir.is_dir():
        raise Exception(f"Result directory {res_dir} does not exist")

    results = {'KNN': {}, 'SVM': {}}
    for fn in res_dir.iterdir():

        with open(fn, 'r') as f:
            data = json.load(f)
        if data['model'] == 'KNN':
            # k number
            k = int(fn.name[1])
            name = ''.join(fn.stem.split('_')[1:])
            if name not in results['KNN']:


                results['KNN'] = {
                    'accuracy': {f: data['folds'][f]['accuracy'] for f in data['folds']},
                    'correct': {f: data['folds'][f]['correct'] for f in data['folds']},
                    'incorrect': {f: data['folds'][f]['incorrect'] for f in data['folds']},
                    'pred_time': {f: data['folds'][f]['pred_time'] for f in data['folds']},
                    'storage': {f: data['folds'][f].get('storage', {}) for f in data['folds']}
                }

        else:
            results['SVM'] = {
                'accuracy': {f: data['folds'][f]['accuracy'] for f in data['folds']},
                'correct': {f: data['folds'][f]['correct'] for f in data['folds']},
                'incorrect': {f: data['folds'][f]['incorrect'] for f in data['folds']},
                'pred_time': {f: data['folds'][f]['pred_time'] for f in data['folds']},
                'storage': {f: data['folds'][f].get('storage', {}) for f in data['folds']}

            }

    return results




def sort_and_prepare_results(results, metric, confidence):
    # Prepare KNN table
    final_result = []



    for fold in range(10):


        row = {'Model':'KNN','Accuracy': results['KNN'].get('accuracy', np.nan).get(str(fold), np.nan),
                   'Pred. Time':  results['KNN'].get('pred_time', np.nan).get(str(fold), np.nan), 'Fold': fold}

        if results['KNN'].get('storage', {}).get(str(fold), {}) == {}:

            row['Storage'] = 100
        else:
            row['Storage'] = results['KNN'].get('storage', {}).get(str(fold), {})


        final_result.append(row)


    for fold in range(10):

        row = {'Model': "SVM", 'Accuracy': results["SVM"].get('accuracy', np.nan).get(str(fold), np.nan),
                   'Pred. Time': results["SVM"].get('pred_time', np.nan).get(str(fold), np.nan), 'Fold': fold}

        if results["SVM"].get('storage', {}).get(str(fold), {}) == {}:

                row['Storage'] = 100
        else:
            row['Storage'] = results["SVM"].get('storage', {}).get(str(fold), {})

        final_result.append(row)

    final_result = pd.DataFrame(final_result)
    if metric == "Pred. Time":
        ascend = True
    else:
        ascend = False

    final_result_grouped = final_result.groupby(['Model']).agg({
        'Accuracy': 'mean',
        'Pred. Time': 'mean',
        'Storage': 'mean'
    }).reset_index().sort_values(
        by= metric, ascending=ascend)



    model_names = []
    model = []
    models = []
    for i in range(1, len(final_result) + 1):

        if i % 10 == 0:
            models.append(model)
            model.append(final_result.iloc[i - 1][metric])
            model_names.append(str(final_result["Model"].iloc[i - 1]))
            model = []
        else:
            model.append(final_result.iloc[i - 1][metric])
    if all(i == models[0] for i in models):
        print("All of the elements of the " + metric + " are the same, so we can't perform the test")

    else:
        t_stat, p_value = stats.ttest_rel(models[0], models[1])

        if p_value > (1 - float(confidence)):
            print(" The values are not significantly different")
        else:
            print(" The values are significantly different")

    print(final_result_grouped)






def main():
    parser = argparse.ArgumentParser(description="Runs an algorithm once on provided data")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="mushroom",
        help="Name of the dataset to process, adult on default."
    )
    parser.add_argument(
        "-r", "--result-directory",
        type=pathlib.Path,
        default=curr_dir / "best_models",
        help="Path to the results directory."
    )
    parser.add_argument(
        "-m", "--metric",
        choices=['Pred. Time', 'Accuracy', 'Storage'],
        default='Accuracy'

    )
    parser.add_argument(
        "-c", "--confidence",
        default=0.95

    )
    args = parser.parse_args()

    args.result_directory = args.result_directory / args.dataset

    results = load_results(args.result_directory)


    sort_and_prepare_results(results, args.metric, args.confidence)



if __name__ == "__main__":
    main()