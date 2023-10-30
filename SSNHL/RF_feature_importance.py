import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from SSNHL.util import load_data


def train_models(X, y):
    clfs = []
    for random_state in range(1, 51):
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )
        clf.fit(X_train, y_train)
        clfs.append(clf)

    return clfs


@click.command()
@click.option("--data_path", help=".xlsx file path", type=str)
@click.option("--output_dir", help="Folder path for results output", type=str)
@click.option("--preprocess_func", default="default", type=str)
def run(data_path, output_dir, preprocess_func):
    X, y, filter_data = load_data(data_path, preprocess_func)
    targets = {
        "effective": {0: 0, 1: 1, 2: 1, 3: 1},
        "markedly effective": {0: 0, 1: 0, 2: 1, 3: 1},
        "cured": {0: 0, 1: 0, 2: 0, 3: 1},
        "all": {0: 0, 1: 1, 2: 2, 3: 3},
    }

    results = pd.DataFrame()
    for target in targets:
        _y = y.copy()
        target_map = targets[target]
        for origin_class in target_map:
            _y[_y == origin_class] = target_map[origin_class]

        clfs = train_models(X, _y)

        _importance = pd.DataFrame(columns=filter_data.columns)
        for clf in clfs:
            _importance.loc[len(_importance)] = clf.feature_importances_
        _importance["target"] = target

        results = pd.concat([results, _importance])
    pd.DataFrame(results).to_csv("{}/RF_feature_importance.csv".format(output_dir))


if __name__ == "__main__":
    run()
