from typing import Iterable


from typing import Iterable


def predict(models, X_test) -> Iterable:
    y_preds = []
    for model in models:
        y_pred = model.predict(X_test)
        y_preds.append(y_pred)

    return y_preds
