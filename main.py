from args import parse_args
from constants import TARGET_COLUMNS
from load_data import load_data
from processing import process
from modelling import make_simple_model

import pandas as pd
import matplotlib.pyplot as plt


def main():
    args = parse_args()
    loaded_data = load_data()
    (X_train, y_train, X_test) = process(loaded_data)
    X = pd.concat([X_train, X_test])

    shifts = [175, 185, 195]

    trained_models = []
    for target in TARGET_COLUMNS:
        model = make_simple_model(shifts)
        model = model.fit(X_train, y_train[target])
        trained_models.append(model)

    for t in trained_models:
        print(t.best_estimator_)

    y_fits = []
    y_preds = []
    for m in trained_models:
        result = m.predict(X)
        result_df = pd.DataFrame(result, index=X.index)

        y_fitted = result_df.loc["2020-01-01 00:00:00":"2020-04-30 23:30:00", :]
        y_pred = result_df.loc["2020-05-01 00:00:00":"2020-07-22 23:30:00", :]

        y_fits.append(y_fitted)
        y_preds.append(y_pred)

    y_fits_df = pd.concat(y_fits, axis=1)
    y_fits_df.columns = [f"{c} fitted" for c in TARGET_COLUMNS]

    fits = pd.concat([y_fits_df, y_train], axis=1)
    fits.plot()

    sub = pd.concat(y_preds, axis=1)
    sub.columns = TARGET_COLUMNS
    sub.to_csv(f'submission.csv')

    sub.plot()

    plt.show()


if __name__ == "__main__":
    main()
