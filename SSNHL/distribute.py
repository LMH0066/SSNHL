import matplotlib.pyplot as plt
import pandas as pd
from util import clean_row, preprocess

if __name__ == "__main__":
    data = pd.read_csv("../data.csv", index_col="sample_index")

    filter_data = data.dropna().copy()
    filter_data = clean_row(filter_data, "-")
    filter_data = preprocess(filter_data)
    filter_data = clean_row(filter_data, "正常")

    for column in filter_data.columns:
        filter_data[column] = pd.to_numeric(filter_data[column])

    for column in filter_data.columns:
        ax = filter_data[column].hist()
        plt.savefig(
            "../image/{}_distribution.pdf".format(column),
            format="pdf",
            bbox_inches="tight",
        )
        plt.show()
