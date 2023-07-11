from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import learning_curve
from util import clean_row

data = pd.read_csv("../data.csv", index_col="sample_index")

filter_data = data.copy()
filter_data = filter_data.replace(["-", "无"], np.nan)
# 1~27
filter_data["3"] = filter_data["3"].replace(["L", "R", "L/R"], [1, 2, 3])
filter_data["26"] = filter_data["26"].str.rstrip("D+")
filter_data["26"] = filter_data["26"].str.rstrip("D ")
filter_data["27"] = filter_data["27"].replace(["1（左）", "1（右）"], "1")
# 28~33
filter_data = filter_data[
    ~filter_data.apply(lambda row: row.astype(str).str.contains("无").any(), axis=1)
]
filter_data["29"] = filter_data["29"].replace(
    ["正常", "A", "B", "C", " C", "D"], ["0", "1", "2", "3", "3", "4"]
)
filter_data["30"] = filter_data["30"].replace(
    ["正常", "轻度", "中度", "中重度", "中重度混合", "重度", "极重度", "全聋"],
    ["0", "1", "2", "3", "4", "5", "6", "7"],
)
filter_data["31"] = filter_data["31"].replace(["正常"], "0")
filter_data["32"] = filter_data["32"].replace(
    ["正常", "A", "B", "C", "D"], ["0", "1", "2", "3", "4"]
)
filter_data["33"] = filter_data["33"].replace(
    ["正常", "轻度", "中度", "中重度", "重度", "极重度"],
    ["0", "1", "2", "3", "4", "5"],
)

filter_data = filter_data.replace(["正常"], np.nan)
filter_data = clean_row(filter_data, "混合性")

for column in filter_data.columns:
    filter_data[column] = pd.to_numeric(filter_data[column])

result = np.array(filter_data["result"])
result[result > 1] = 1
filter_data = filter_data.drop(["result"], axis=1)
X = np.array(filter_data)

rf = HistGradientBoostingClassifier(random_state=1)
rf.fit(X, result)

# importances = dict()
# for i, importance in zip(filter_data.columns, rf.feature_importances_):
#     importances[i] = importance
# importances = sorted(importances.items(), key=itemgetter(1))

# _x, _y = [int(i) for i, _ in importances], [i for _, i in importances]
# plt.scatter(_x, _y)
# for i in range(len(_x)):
#     plt.text(_x[i] + 0.3, _y[i], _x[i], fontsize=12)
# # plt.savefig("../image/RandomForest.pdf", format="pdf", bbox_inches="tight")
# plt.show()

# 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    rf,
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
