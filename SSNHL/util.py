import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def load_data(file_path):
    data = pd.read_excel(file_path, index_col=0, skiprows=[1, 4], header=[0, 1, 2])
    data = data.drop(["note"], axis=1)
    filter_data = Preprocess().do(data, "default")

    y = np.array(
        filter_data["efficacy evaluation"][
            filter_data["efficacy evaluation"].columns[0]
        ]
    )
    filter_data = filter_data.drop(["efficacy evaluation"], axis=1)
    X = np.array(filter_data)

    return X, y


class Preprocess:
    def __init__(self):
        self.function_map = {
            "default": self.default,
        }
    
    def do(self, data: pd.DataFrame, func_name: str):
        filter_data = self.function_map[func_name](data)

        for column in filter_data.columns:
            filter_data[column] = pd.to_numeric(filter_data[column])
        
        return filter_data

    def _clean_row(self, data: pd.DataFrame, string: str) -> pd.DataFrame:
        return data[
            ~data.apply(lambda row: row.astype(str).str.contains(string).any(), axis=1)
        ]

    def default(self, data: pd.DataFrame) -> pd.DataFrame:
        filter_data = data.copy()
        filter_data = filter_data.drop(
            ("coagulation function", "fibrinogen", "after treatment"), axis=1
        )
        filter_data = filter_data.drop(
            ("blood lipids", "Unnamed: 17_level_1", "triglycerides"), axis=1
        )
        filter_data = filter_data.drop(
            ("treatment plan", "Unnamed: 27_level_1", "tympanic injection time"), axis=1
        )

        filter_data["ear"] = filter_data["ear"].replace(["L", "R", "L/R"], [1, 2, 3])

        filter_data = filter_data.dropna()
        filter_data = self._clean_row(filter_data, "-")
        filter_data = self._clean_row(filter_data, "normal")
        return filter_data


def train_model(clf, X_train, y_train, X_test, y_test, n_class):
    clf.fit(X_train, y_train)

    # evaluate
    accuracy = clf.score(X_test, y_test)
    y_pred_proba = clf.predict_proba(X_test)

    if n_class == 2:
        false_positive_rate, recall, _ = roc_curve(y_test, y_pred_proba[:, 1])
    else:
        # multi class
        y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
        false_positive_rate, recall, _ = roc_curve(
            y_test_one_hot.ravel(), y_pred_proba.ravel()
        )
    roc_auc = auc(false_positive_rate, recall)
    roc = [false_positive_rate, recall, roc_auc]

    return accuracy, roc
