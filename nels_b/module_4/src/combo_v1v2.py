"""Stack the best V1 model and best V2 model — genuinely different pipelines, so
their errors decorrelate and a meta-model has something real to exploit.

Meta-models are fit on base predictions over X_test (held out of base training) and
graded on the mini holdout. Both versions share the same train/test split rows
(RANDOM_STATE=42), so test predictions and y_test align row-for-row.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error

import module4 as m
import paths
import v1_pipeline as v1
from grade_holdout import ANSWERS_URL, PLOT_DIR, grade, report_and_plot

V1_BEST = os.path.join(paths.MODELS_V1, "model1_b256.keras")
V2_BEST = os.path.join(paths.MODELS_V2, "model1_b256.keras")


def main():
    import os

    import seaborn as sns

    m.configure_cpu_parallelism()
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set(rc={"figure.figsize": (11.7, 8.27)})

    answers = pd.read_csv(ANSWERS_URL)
    actual = (answers["casual"] + answers["registered"]).to_numpy()

    # V2 (current module4 pipeline)
    _, X_test_v2, y_test, y_scaler_v2, _ = m.prepare_data()
    y_test_total = y_test.sum(axis=1).to_numpy()
    from grade_holdout import holdout_features as holdout_v2

    x_scaler_v2 = _rebuild_v2_xscaler()
    Xho_v2 = holdout_v2(x_scaler_v2)
    mdl_v2 = tf.keras.models.load_model(V2_BEST, compile=False)
    v2_test = m.invert_targets(y_scaler_v2, mdl_v2.predict(X_test_v2, verbose=0)).sum(axis=1)
    v2_hold = m.invert_targets(y_scaler_v2, mdl_v2.predict(Xho_v2, verbose=0)).sum(axis=1)

    # V1 (reconstructed pipeline)
    x_scaler_v1, y_scaler_v1, X_test_v1, y_test_v1, num_scaler_v1 = v1.prepare_v1()
    assert np.array_equal(
        y_test_v1.sum(axis=1).to_numpy(), y_test_total
    ), "V1/V2 test splits diverged"
    Xho_v1 = v1.holdout_features_v1(x_scaler_v1, num_scaler_v1)
    mdl_v1 = tf.keras.models.load_model(V1_BEST, compile=False)
    v1_test = v1.invert_v1(y_scaler_v1, mdl_v1.predict(X_test_v1, verbose=0)).sum(axis=1)
    v1_hold = v1.invert_v1(y_scaler_v1, mdl_v1.predict(Xho_v1, verbose=0)).sum(axis=1)

    rows = [
        grade("V1_best", v1_hold, actual),
        grade("V2_best", v2_hold, actual),
    ]

    # Simple mean of the two.
    avg = np.clip(np.round((v1_hold + v2_hold) / 2, 1), 0, None)
    rows.append(grade("combo_avg", avg, actual))
    pd.DataFrame({"predictions": avg}).to_csv(paths.expl("combo_avg-predictions.csv"), index=False)

    # Stacking meta-models on [v1, v2] base totals.
    meta_train = np.column_stack([v1_test, v2_test])
    meta_hold = np.column_stack([v1_hold, v2_hold])
    metas = {
        "ols": LinearRegression(),
        "ridge": Ridge(alpha=100.0),
        "nonneg": LinearRegression(positive=True),
    }
    fitted = {}
    for name, est in metas.items():
        est.fit(meta_train, y_test_total)
        fitted[name] = est
        sp = np.clip(np.round(est.predict(meta_hold), 1), 0, None)
        rows.append(grade(f"combo_{name}", sp, actual))
        pd.DataFrame({"predictions": sp}).to_csv(paths.expl(f"combo_{name}-predictions.csv"), index=False)

    for r in rows:
        report_and_plot(r)

    print("\nCorrelation of V1 vs V2 errors on X_test:")
    e1, e2 = v1_test - y_test_total, v2_test - y_test_total
    print(f"  corr = {np.corrcoef(e1, e2)[0, 1]:.3f}  (lower = more diverse = better to stack)")
    print("\nCombo meta-models (fit on [V1,V2] base totals over X_test):")
    for name, est in fitted.items():
        print(
            f"  {name:7s} intercept={est.intercept_:+7.3f}  "
            f"V1={est.coef_[0]:+.3f}  V2={est.coef_[1]:+.3f}"
        )

    df = (
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        .set_index("model")
        .round(3)
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== V1+V2 combo grading (mini holdout) ===")
    print(df.to_string())


def _rebuild_v2_xscaler():
    """V2 x_scaler fit on training X_train (matches grade_holdout)."""
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    data = m.load_and_engineer_features(m.DATA_URL)
    x_raw = data.drop(columns=["dteday", "hr", "dow", "casual", "registered"])
    y_raw = data[["casual", "registered"]]
    X_tr, _, _, _ = train_test_split(x_raw, y_raw, test_size=0.3, random_state=m.RANDOM_STATE)
    return MinMaxScaler().fit(X_tr)


if __name__ == "__main__":
    main()
