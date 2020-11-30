import numpy as np


def show_results(models):
    scores = [model.best_score_ * -1 for model in models]
    best_params = [model.best_params_ for model in models]

    print("\n==================== Train results ====================")
    print(f"scores: {scores}")
    print(f"total score: {np.mean(scores)}")
    for params in best_params:
        print(params)
    print("========================= End =========================")
