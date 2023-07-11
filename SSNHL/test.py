from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from util import clean_row, preprocess

random_state = 1
data = pd.read_csv("../data.csv", index_col="sample_index")

filter_data = data.dropna().copy()
filter_data = clean_row(filter_data, "-")
filter_data = preprocess(filter_data)
filter_data = clean_row(filter_data, "正常")

for column in filter_data.columns:
    filter_data[column] = pd.to_numeric(filter_data[column])

result = np.array(filter_data["result"])
# result[result > 1] = 1
filter_data = filter_data.drop(["result"], axis=1)
X = np.array(filter_data)

clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
# clf = ExtraTreesClassifier(n_estimators=100, random_state=random_state)
clf.fit(X, result)

print(cross_val_score(clf, X, result, cv=5))

# importances = dict()
# for i, importance in zip(filter_data.columns, clf.feature_importances_):
#     importances[i] = importance
# importances = sorted(importances.items(), key=itemgetter(1))
# _x, _y = [int(i) for i, _ in importances], [i for _, i in importances]
# plt.scatter(_x, _y)
# for i in range(len(_x)):
#     plt.text(_x[i] + 0.3, _y[i], _x[i], fontsize=12)
# plt.savefig("../image/RandomForest.pdf", format="pdf", bbox_inches="tight")
# plt.show()

# 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    clf,
    X,
    result,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure()
# plt.title("Learning Curve(RF)")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy Score")
plt.grid()
plt.fill_between(
    train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r"
)
plt.fill_between(
    train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g"
)
plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
plt.plot(train_sizes, test_mean, "o-", color="g", label="Validation Score")
plt.legend(loc="best")
plt.show()

# for column in filter_data.columns:
#     ax = filter_data[column].hist()
#     plt.savefig("../image/{}_distribution.pdf".format(column), format="pdf", bbox_inches="tight")
#     plt.show()
