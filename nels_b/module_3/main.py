import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Paths resolved relative to this file so the script works from any cwd.
_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA  = _ROOT / 'module_3' / 'data' / 'training_data' / 'housing.csv'
MINI_HOLDOUT   = _ROOT / 'module_3' / 'data' / 'test_data' / 'housing_holdout_test_mini.csv'
PREDICTIONS_DIR = Path(__file__).resolve().parent / 'predictions'

# Make evaluate_housing importable
sys.path.insert(0, str(_ROOT / 'module_3' / 'scripts'))
from evaluate_housing import load_answers, evaluate_predictions, print_results, plot_results


def load_data(path):
    return pd.read_csv(path)


def preprocess(data, date_min=None):
    """Preprocess housing data.

    date_min: pass the value from training so holdout uses same date baseline.
              If None, computed from data (training mode).
    Returns (data, date_min).
    """
    # Drop 'id' if present; it is not a useful feature.
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    # Encode 'zipcode' as categorical, then one-hot encode.
    data['zipcode'] = data['zipcode'].astype('category')
    data = pd.get_dummies(data, columns=['zipcode'], drop_first=True)

    # Convert 'date' to days since a fixed baseline.
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%dT%H%M%S')
    if date_min is None:
        date_min = data['date'].min()
    data['date'] = (data['date'] - date_min).dt.days

    # Log-transform 'price' to stabilize variance (training only).
    if 'price' in data.columns:
        data['price'] = data['price'].apply(lambda x: np.log(x) if x > 0 else 0)

    return data, date_min


def add_geo_features(data, km=None, knn=None, n_clusters=20, knn_k=10, n_folds=5, random_state=42):
    """Add K-Means centroid distances and KNN price meta-feature.

    Training mode (km=None): fits transformers and generates OOF predictions to avoid leakage.
    Inference mode (km provided): applies pre-fitted transformers to new data.

    Returns (data, km, knn) — pass km and knn back in for inference.
    """
    # Coordinates and distance feature names.
    coords = data[['lat', 'long']].values
    dist_cols = [f'km_dist_{i}' for i in range(n_clusters)]

    # Fit KMeans and KNN on training data, or apply pre-fitted models to new data.
    if km is None:
        # Fit KMeans.
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        km.fit(coords)

        # OOF KNN predictions to avoid leakage in the geo meta-feature.
        price_arr = data['price'].values
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        oof_knn = np.zeros(len(data))
        for tr_idx, val_idx in kf.split(coords):
            _knn = KNeighborsRegressor(n_neighbors=knn_k)
            _knn.fit(coords[tr_idx], price_arr[tr_idx])
            oof_knn[val_idx] = _knn.predict(coords[val_idx])
        data['knn_geo_price'] = oof_knn

        # Fit final KNN on all data for inference use.
        knn = KNeighborsRegressor(n_neighbors=knn_k)
        knn.fit(coords, price_arr)
    else:
        data['knn_geo_price'] = knn.predict(coords)

    data[dist_cols] = km.transform(coords)
    return data, km, knn


def normalize_features(data, feature_cols=None, stats=None):
    """Normalize features using training data statistics.

    Returns (data, stats) where stats is a dict of {col: (mean, std)}.
    """
    # If feature_cols not provided, infer from stats keys or numeric columns (excluding 'price').
    if feature_cols is None:
        if stats is not None:
            feature_cols = list(stats.keys())
        else:
            feature_cols = data.select_dtypes(include=[np.number]).columns.drop('price', errors='ignore')

    # If stats not provided, compute from data (training mode). Otherwise, use provided stats (inference mode).
    if stats is None:
        stats = {}
        for col in feature_cols:
            mean = data[col].mean()
            std = data[col].std()
            stats[col] = (mean, std)

    # Normalize using provided stats, adding missing columns as zeros if needed.
    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0
        mean, std = stats[col]
        if std > 0:
            data[col] = (data[col] - mean) / std
        else:
            data[col] = 0  # If no variance, set to zero
    return data, stats


def split_features(data, test_size=0.2, random_state=42):
    X = data.drop(columns=['price'])
    y = data['price']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return model


def compute_smearing_factor(y_true_log, y_pred_log):
    # Smearing estimator to correct log-to-linear bias.
    residuals = y_true_log - y_pred_log
    return float(np.mean(np.exp(residuals)))


def evaluate_model(model, X_test, y_test, use_log_correction=True):
    # Predict in log space and optionally apply smearing correction.
    y_pred_log = model.predict(X_test)
    smearing_factor = compute_smearing_factor(y_test, y_pred_log) if use_log_correction else 1.0
    y_pred = np.exp(y_pred_log) * smearing_factor
    y_test_orig = np.exp(y_test)
    rmse = sk.metrics.root_mean_squared_error(y_test_orig, y_pred)
    mae = sk.metrics.mean_absolute_error(y_test_orig, y_pred)
    print(f'Validation RMSE: {rmse:,.2f}')
    print(f'Validation MAE:  {mae:,.2f}')
    return rmse, mae, smearing_factor


def predict_holdout(
    model,
    km,
    knn,
    date_min,
    train_columns,
    norm_stats,
    holdout_path=MINI_HOLDOUT,
    save=False,
    use_log_correction=True,
    smearing_factor=1.0,
):
    """Predict prices for holdout CSV. Returns numpy array of predictions.

    save=True writes predictions to PREDICTIONS_DIR/mini_holdout_predictions.csv.
    """

    # Load and preprocess holdout data using the same steps and stats as training.
    holdout = load_data(holdout_path)
    holdout, _ = preprocess(holdout, date_min=date_min)
    holdout, _, _ = add_geo_features(holdout, km=km, knn=knn)
    holdout = holdout.reindex(columns=train_columns, fill_value=0)
    holdout, _ = normalize_features(holdout, feature_cols=train_columns, stats=norm_stats)

    X_holdout = holdout[train_columns]
    preds_log = model.predict(X_holdout)
    factor = smearing_factor if use_log_correction else 1.0
    preds = np.exp(preds_log) * factor

    if save:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PREDICTIONS_DIR / 'NorthWind_Nels-Module2-mini_holdout-predictions.csv'
        pd.DataFrame({'price': preds}).to_csv(out_path, index=False)
        print(f'Predictions saved → {out_path}')

    return preds


def evaluate_holdout(preds_array, label='mini_holdout', plot=True):
    """Score a predictions array against the mini holdout answer key."""
    # Load answer key and evaluate predictions, then print and plot results.
    answers = load_answers()
    results = [evaluate_predictions(answers, preds_array, label=label)]
    print_results(results)
    if plot:
        plot_results(results)
    return results


if __name__ == '__main__':
    # Train
    data = load_data(TRAINING_DATA)
    base_data, date_min = preprocess(data)

    random_states = [1, 12, 123, 1234, 12345]
    # random_states = [42]  # For quick testing, replace with all_random_states for more runs
    use_log_correction = True

    for rs in random_states:
        print(f'\n{"="*20} Random State: {rs} {"="*20}\n')
        data = base_data.copy()
        data, km, knn = add_geo_features(data, random_state=rs)
        X_train, X_test, y_train, y_test = split_features(data, random_state=rs, test_size=0.05)
        X_train, norm_stats = normalize_features(X_train, feature_cols=X_train.columns)
        X_test, _ = normalize_features(X_test, feature_cols=X_train.columns, stats=norm_stats)
        model = train_model(X_train, y_train, n_estimators=800, learning_rate=0.05, max_depth=6)

        # Validate on held-out training split
        _, _, smearing_factor = evaluate_model(model, X_test, y_test, use_log_correction=use_log_correction)

        # Predict and score against mini holdout
        preds = predict_holdout(
            model,
            km,
            knn,
            date_min,
            X_train.columns,
            norm_stats,
            save=True,
            use_log_correction=use_log_correction,
            smearing_factor=smearing_factor,
         
        )
        evaluate_holdout(preds, plot=False)