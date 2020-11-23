from constants import FEATURE_COLUMNS

from load_data import load_data
from nan_processsing import process_na
from modelling import evaluate_training
from plotting import plot_fitted_values
from submission import create_submission


def main():
    data = load_data()
    X_train, y_train, X_test = process_na(data)
    models = evaluate_training(X_train, y_train)

    y_preds = []
    for target, model in models.items():
        y_pred = model.predict(X_test[FEATURE_COLUMNS])
        y_preds.append(y_pred)
        plot_fitted_values(model.best_estimator_,
                           X_train[FEATURE_COLUMNS], y_train[target], target)

    sub = create_submission(X_test["timestamp"], y_preds)
    sub.to_csv(f'submission.csv', index=False)


if __name__ == "__main__":
    main()
