from args import parse_args
from constants import TARGET_COLUMNS
from load_data import load_data
from processing import process
from modelling import make_simple_model
from result_view import show_results

import pandas as pd
import matplotlib.pyplot as plt


def main():
    args = parse_args()
    loaded_data = load_data()
    (X_train, y_train, X_test) = process(loaded_data)
    X = pd.concat([X_train, X_test])

    trained_models = []
    for target in TARGET_COLUMNS:
        model = make_simple_model(target)
        model = model.fit(X_train, y_train[target])
        trained_models.append(model)

    y_fits = []
    y_preds = []
    for m in trained_models:
        result = m.predict(X)
        result_df = pd.DataFrame(result, index=X.index)

        y_fitted = result_df.loc["2020-03-01 00:00:00":"2020-04-30 23:30:00", :]
        y_pred = result_df.loc["2020-05-01 00:00:00":"2020-07-22 23:30:00", :]

        y_fits.append(y_fitted)
        y_preds.append(y_pred)

    y_fits_df = pd.concat(y_fits, axis=1)
    y_fits_df.columns = [f"{c}_fitted" for c in TARGET_COLUMNS]

    fits = pd.concat(
        [y_fits_df, y_train.loc["2020-03-01 00:00:00":, :]], axis=1)

    sub = pd.concat(y_preds, axis=1)
    sub.columns = TARGET_COLUMNS
    sub.to_csv(f'submission.csv')

    if args.verbose:
        show_results(trained_models, fits)
        fits.plot()
        sub.plot()
        plt.show()


if __name__ == "__main__":
    main()
