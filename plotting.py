import matplotlib.pyplot as plt


def plot_fitted_values(estimator, X, y_true, title):
    y_pred = estimator.predict(X)
    _, ax = plt.subplots(1, 1, figsize=(15, 3))

    ax.plot(y_true, label="true")
    ax.plot(y_pred, label="prediction")
    ax.legend()
    ax.set_title(title)