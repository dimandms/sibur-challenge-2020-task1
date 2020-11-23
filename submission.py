from matplotlib.pyplot import axis
import numpy as np
import pandas as pd

from constants import FEATURE_COLUMNS, TARGET_COLUMNS


def create_submission(timestamps, y_preds):
    if len(y_preds) != 4:
        raise Exception(f"missmatch of y_preds, want 4, has: {len(y_preds)}")

    y_preds_array = np.concatenate([y_pred.reshape(-1, 1) for y_pred in y_preds], axis=1)
    y_preds_df = pd.DataFrame(y_preds_array, columns=TARGET_COLUMNS)

    return pd.concat([timestamps, y_preds_df], axis=1)
