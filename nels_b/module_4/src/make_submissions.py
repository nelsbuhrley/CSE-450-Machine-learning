"""Generate the named NorthWind submission files.

Two models × two holdouts → four files in predictions/:
  v2 = best single V2 MLP (pure neural network, models_v2/model1_b256.keras)
  v3 = holiday-aware MLP + XGBoost blend (best_model_v3)

  NorthWind_Nels-model{v2|v3}-module4-{mini_holdout|holdout}-predictions.csv
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, root_mean_squared_error

import best_model_v3 as v3
import module4 as m
import paths

OUT = paths.PREDICTIONS
ANSWERS_URL = paths.ANSWERS_URL
HOLDOUTS = {
    "mini_holdout": paths.MINI_HOLDOUT,
    "holdout": paths.BIG_HOLDOUT,
}


def v2_predict(csv: str) -> np.ndarray:
    """Pure-NN prediction: best single V2 MLP with the V2 scalers (saved in best_model/)."""
    with open(os.path.join(paths.BEST_MODEL, "feature_cols.json")) as f:
        feat = json.load(f)
    x_scaler = joblib.load(os.path.join(paths.BEST_MODEL, "x_scaler.joblib"))
    y_scaler = joblib.load(os.path.join(paths.BEST_MODEL, "y_scaler.joblib"))
    mlp = tf.keras.models.load_model(os.path.join(paths.BEST_MODEL, "mlp.keras"), compile=False)
    X = m.load_and_engineer_features(csv)[feat]
    total = m.invert_targets(y_scaler, mlp.predict(x_scaler.transform(X), verbose=0)).sum(axis=1)
    return np.clip(np.round(total, 1), 0, None)


def main():
    m.configure_cpu_parallelism()
    os.makedirs(OUT, exist_ok=True)
    answers = pd.read_csv(ANSWERS_URL)
    mini_actual = (answers["casual"] + answers["registered"]).to_numpy()

    for tag, csv in HOLDOUTS.items():
        for ver, preds in (("v2", v2_predict(csv)), ("v3", v3.predict(csv))):
            path = f"{OUT}/NorthWind_Nels-model{ver}-module4-{tag}-predictions.csv"
            pd.DataFrame({"predictions": preds}).to_csv(path, index=False)
            note = ""
            if tag == "mini_holdout":
                note = (f"  RMSE {root_mean_squared_error(mini_actual, preds):.2f}"
                        f"  R² {r2_score(mini_actual, preds):.3f}")
            print(f"{path}  ({len(preds)} rows, mean {preds.mean():.1f}){note}")


if __name__ == "__main__":
    main()
