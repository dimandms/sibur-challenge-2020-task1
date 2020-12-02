from args import parse_args
from constants import TARGET_COLUMNS
from load_data import load_data
from processing import process
from modelling import make_stacked_model

import pandas as pd


def main():
    args = parse_args()
    loaded_data = load_data()
    (X_train, y_train, X_test) = process(loaded_data)
    X = pd.concat([X_train, X_test])

    shifts = [175, 185, 195]

    trained_models = []
    for target in TARGET_COLUMNS:
        model = make_stacked_model(shifts)
        model = model.fit(X_train, y_train[target])
        trained_models.append(model)

    for t in trained_models:
        print([e.best_estimator_ for e in t.estimators_])
    # y_fits = []
    y_preds = []
    for m in trained_models:
        result = m.predict(X)
        result_df = pd.DataFrame(result, index=X.index)

        # y_fitted = result.loc["2020-01-01 00:00:00":"2020-04-30 23:30:00", :]
        y_pred = result_df.loc["2020-05-01 00:00:00":"2020-07-22 23:30:00", :]

        y_preds.append(y_pred)

    sub = pd.concat(y_preds, axis=1)
    sub.to_csv(f'submission.csv')


if __name__ == "__main__":
    main()
