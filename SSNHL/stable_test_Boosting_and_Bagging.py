import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split
from util import load_data, train_model


def calculate(X, y):
    n_class = np.unique(y).size

    functions = [
        # Boosting
        AdaBoostClassifier,
        GradientBoostingClassifier,
        # Bagging
        RandomForestClassifier,
        ExtraTreesClassifier,
    ]
    results, rocs = dict(), dict()
    for function in functions:
        accuracy, roc = [], [[], [], []]
        for random_state in range(1, 51):
            clf = function(n_estimators=100, random_state=random_state)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=random_state
            )
            _accuracy, _roc = train_model(
                clf, X_train, y_train, X_test, y_test, n_class
            )

            accuracy.append(_accuracy)
            roc[0].extend(_roc[0].tolist())
            roc[1].extend(_roc[1].tolist())
            roc[2].append(_roc[2])
        roc[2] = sum(roc[2]) / len(roc[2])

        results[clf.__class__.__name__] = accuracy
        rocs[clf.__class__.__name__] = roc

    return results, rocs


if __name__ == "__main__":
    X, y = load_data("../raw_data/all.xlsx")
    targets = {
        "effective": {0: 0, 1: 1, 2: 1, 3: 1},
        "markedly effective": {0: 0, 1: 0, 2: 1, 3: 1},
        "cured": {0: 0, 1: 0, 2: 0, 3: 1},
        "all": {0: 0, 1: 1, 2: 2, 3: 3},
    }

    for target in targets:
        _y = y.copy()
        target_map = targets[target]
        for origin_class in target_map:
            _y[_y == origin_class] = target_map[origin_class]

        results, rocs = calculate(X, _y)

        for func_name in results.keys():
            result, roc = results[func_name], rocs[func_name]

            plt.scatter(roc[0], roc[1], label=f"AUC_{func_name}=%0.3f" % roc[2])
            plt.legend(loc="best", frameon=False)
            plt.plot([0, 1], [0, 1], "r--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.ylabel("Recall")
            plt.xlabel("Fall-out")

        plt.savefig("../output/ROC_{}.pdf".format(target))
        plt.cla()
        pd.DataFrame(results).to_csv("../output/accuracy_{}.csv".format(target))
