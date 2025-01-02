import click
import numpy as np
import pandas as pd

from SSNHL.stable_test import calculate
from SSNHL.util import load_data


@click.command()
@click.option("--data_path", help=".xlsx file path", type=str)
@click.option("--after_treatment_data_path", help=".xlsx file path", type=str)
@click.option("--output_dir", help="Folder path for results output", type=str)
@click.option("--preprocess_func", default="default", type=str)
def run(data_path, after_treatment_data_path, output_dir, preprocess_func):
    result_col = "WHO classify (affected side) after treatment"
    data = pd.read_excel(data_path, index_col=0, header=[0])
    data = data.drop("prognostic(no_recovery=0, minor_recovery=1, important_recovery=2, full_recovery=3)", axis=1)
    after_treatment = pd.read_excel(after_treatment_data_path, index_col=0, header=[0])
    after_treatment = after_treatment[[result_col]]

    data = pd.merge(data, after_treatment, on="patient NO.", how="inner")
    X, y, _ = load_data(
        data,
        preprocess_func,
        result_col,
    )

    results, rocs = calculate(X, y)

    pd.DataFrame(results).to_csv("{}/accuracy.csv".format(output_dir))
    np.save("{}/ROC.npy".format(output_dir), rocs)


if __name__ == "__main__":
    run()
