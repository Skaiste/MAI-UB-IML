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
                results['KNN'][name] = {
                    'distance': data['distance'],
                    'voting': data['voting'],
                    'weighting': data['weighting'],
                    'k': {},
                    'reduction': data['reduction']
                }
                if data['distance'] == 'minkowski':
                    results['KNN'][name]['distance'] = results['KNN'][name]['distance'] + "_" + str(data['minkowski_r'])

            results['KNN'][name]['k'][k] = {
                'accuracy': {f: data['folds'][f]['accuracy'] for f in data['folds']},
                'correct': {f: data['folds'][f]['correct'] for f in data['folds']},
                'incorrect': {f: data['folds'][f]['incorrect'] for f in data['folds']},
                'pred_time': {f: data['folds'][f]['pred_time'] for f in data['folds']},
                'storage': {f: data['folds'][f].get('storage', {}) for f in data['folds']}

            }

        else:
            name = ''.join(fn.stem.split('_')[1:])
            results['SVM'][name] = {
                'kernel': data['kernel'],
                'reduction': data.get('reduction', 'no_reduction'),
                'accuracy': {f: data['folds'][f]['accuracy'] for f in data['folds']},
                'correct': {f: data['folds'][f]['correct'] for f in data['folds']},
                'incorrect': {f: data['folds'][f]['incorrect'] for f in data['folds']},
                'pred_time': {f: data['folds'][f]['pred_time'] for f in data['folds']},
                'storage': {f: data['folds'][f].get('storage', {}) for f in data['folds']}

            }

    return results




def sort_and_prepare_results(results, metric, confidence, svm_ir):
    # Prepare KNN table
    knn_data = []
    print(" KNN ")
    for name, knn_results in results['KNN'].items():

        for fold in range(10):

            row_ = {
                'Distance': knn_results['distance'],
                'Voting': knn_results['voting'],
                'Weighting': knn_results['weighting'],
                'Reduction' : knn_results['reduction']
            }

            for k in [1, 3, 5, 7]:
                if k not in knn_results['k']:
                    continue

                row = row_.copy()
                row['Accuracy'] = knn_results['k'].get(k, np.nan).get('accuracy', np.nan).get(str(fold), np.nan)
                row['Pred. Time'] = knn_results['k'].get(k, np.nan).get('pred_time', np.nan).get(str(fold), np.nan)
                row['Fold'] = fold
                if knn_results['k'].get(k, np.nan).get('storage', {}).get(str(fold), {}) == {}:

                    row['Storage'] = 100
                else:
                    row['Storage'] = knn_results['k'].get(k, np.nan).get('storage', {}).get(str(fold), {})

                row['K'] = k

                knn_data.append(row)

    knn_df = pd.DataFrame(knn_data)
    if metric == "Pred. Time":
        ascend = True
    else:
        ascend = False

    knn_df_grouped = knn_df.groupby(['Distance', 'Voting', 'Weighting', 'K','Reduction']).agg({
        'Accuracy': 'mean',
        'Pred. Time': 'mean',
        'Storage': 'mean'
    }).reset_index().sort_values(
        by= metric, ascending=ascend)

    top_knn = knn_df_grouped.iloc[:5]

    print("the best models \n", top_knn)

    filtered_knn_df = pd.DataFrame()
    for _, row in top_knn.iterrows():
        match = knn_df[
            (knn_df['Distance'] == row['Distance']) &
            (knn_df['Voting'] == row['Voting']) &
            (knn_df['Weighting'] == row['Weighting']) &
            (knn_df['K'] == row['K']) &
            (knn_df['Reduction'] == row['Reduction'])
            ]
        filtered_knn_df = pd.concat([filtered_knn_df, match])
    filtered_knn_df.reset_index(drop=True, inplace=True)

    models = []
    model_names = []
    model = []

    for i in range(1, len(filtered_knn_df) + 1):

        if i % 10 == 0:
            models.append(model)
            model.append(filtered_knn_df.iloc[i - 1][metric])
            model_names.append(str(filtered_knn_df["Distance"].iloc[i - 1]) + " " + str(
                filtered_knn_df["Voting"].iloc[i - 1]) + " " + str(
                filtered_knn_df["Weighting"].iloc[i - 1]) + " " + str(filtered_knn_df["K"].iloc[i - 1])+" " + str(filtered_knn_df["Reduction"].iloc[i - 1]))
            model = []

        else:
            model.append(filtered_knn_df.iloc[i - 1][metric])

    if all(i == models[0] for i in models):
        print("All of the elements of the " + metric + " are the same, so we can't perform the test")
    else:
        res = friedmanchisquare(*models)
        pvalue = res.pvalue
        ranked_models = [rankdata(model,method = "average") for model in models]

        average_ranks = [sum(ranks) / len(ranks) for ranks in ranked_models]

        if pvalue > (1 - float(confidence)):
            print(" The values are not significantly different")
        else:
            print(" The values are significantly different, H₀ is rejected, let's apply Nemenyi Post-Doc test")
            models = np.array(models).T
            nemenyi =sp.posthoc_nemenyi_friedman(np.array(models))
            print(nemenyi)

            models = models.T
            if metric == "Pred. Time":

                models = [np.mean(model) for model in models]
            else:
                models = [-1*np.mean(model) for model in models]
            ranked_models = rankdata(models)




            # Create a figure and an axis
            plt.figure(figsize=(10, 2), dpi=100)
            plt.title('Critical difference diagram of average score ranks')
            print(ranked_models)
            sp.critical_difference_diagram(
                ranked_models,
                nemenyi,

            )
            plt.show()



    # Prepare SVM table
    print("SVM")
    svm_data = []
    for name, svm_results in results['SVM'].items():
        for fold in range(10):
            print(svm_results)
            if svm_results.get('storage', {}) == {}:
                storage = 100

            else:
                storage = svm_results.get('storage', {}).get(str(fold), {})
            svm_data.append({
                'Kernel': svm_results.get('kernel', 'no_kernel'),
                'Reduction': svm_results.get('reduction', 'no_reduction'),
                'Accuracy': svm_results.get('accuracy', np.nan).get(str(fold), np.nan),
                'Pred. Time': svm_results.get('pred_time', {}).get(str(fold), np.nan),
                'Fold': fold,
                'Storage': storage

            })
    # Create SVM DataFrame and sort by accuracy mean
    svm_df = pd.DataFrame(svm_data)
    model_names = []
    model = []
    models = []
    for i in range(1, len(svm_df) + 1):

        if i % 10 == 0:
            models.append(model)
            model.append(svm_df.iloc[i - 1][metric])
            model_names.append(str(svm_df["Kernel"].iloc[i - 1])+str(svm_df["Reduction"].iloc[i - 1]))
            model = []
        else:
            model.append(svm_df.iloc[i - 1][metric])
    if all(i == models[0] for i in models):
        print("All of the elements of the " + metric + " are the same, so we can't perform the test")
    elif svm_ir:
        res = friedmanchisquare(*models)
        pvalue = res.pvalue
        ranked_models = [rankdata(model, method="average") for model in models]

        average_ranks = [sum(ranks) / len(ranks) for ranks in ranked_models]

        if pvalue > (1 - float(confidence)):
            print(" The values are not significantly different")
        else:
            print(" The values are significantly different, H₀ is rejected, let's apply Nemenyi Post-Doc test")
            models = np.array(models).T
            nemenyi = sp.posthoc_nemenyi_friedman(np.array(models))
            print(nemenyi)

            models = models.T
            if metric == "Pred. Time":

                models = [np.mean(model) for model in models]
            else:
                models = [-1 * np.mean(model) for model in models]
            ranked_models = rankdata(models)

            # Create a figure and an axis
            plt.figure(figsize=(10, 2), dpi=100)
            plt.title('Critical difference diagram of average score ranks')
            print(ranked_models)
            sp.critical_difference_diagram(
                ranked_models,
                nemenyi,

            )
            plt.show()
    else:
        t_stat, p_value = stats.ttest_rel(models[0], models[1])
        if p_value > (1 - float(confidence)):
            print(" The values are not significantly different")

        else:
            print(" The values are significantly different")
    if metric == "Pred. Time":
        ascend = True
    else:
        ascend = False

    svm_df_grouped = svm_df.groupby(['Kernel','Reduction']).agg({
        'Accuracy': 'mean',
        'Pred. Time': 'mean',
        'Storage': 'mean'

    }).reset_index().sort_values(
        by=metric, ascending=ascend)

    print(svm_df_grouped)

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
        default="mushroom",
        help="Name of the dataset to process, adult on default."
    )
    parser.add_argument(
        "-r", "--result-directory",
        type=pathlib.Path,
        default=curr_dir / "results",
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
    parser.add_argument(
        "-svm_ir", "--svm_ir",
        default=False

    )

    args = parser.parse_args()

    args.result_directory = args.result_directory / args.dataset

    results = load_results(args.result_directory)


    knn, svm = sort_and_prepare_results(results, args.metric, args.confidence,args.svm_ir)

    # print(dataframe_to_markdown(knn))
    # print(dataframe_to_markdown(svm))


if __name__ == "__main__":
    main()