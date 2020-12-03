from constants import TARGET_COLUMNS
import numpy as np
from metrics import mean_absolute_percentage_error


def show_results(models, fits):
    best_params = [model.best_params_ for model in models]

    scores = []
    for target in TARGET_COLUMNS:
        score = mean_absolute_percentage_error(
            fits[target], fits[f"{target}_fitted"])
        scores.append(score)

    print("\n==================== Train results ====================")
    print(f"scores: {scores}")
    print(f"total score: {np.mean(scores)}")
    for params in best_params:
        print(params)
    print("========================= End =========================")
