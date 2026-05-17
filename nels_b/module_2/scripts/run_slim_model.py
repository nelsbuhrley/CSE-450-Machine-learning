"""
Slimmed-down model: drop job, marital, loan, emp.var.rate from Caleb's pipeline.
Trains on bank.csv, predicts on bank_holdout_test_mini.csv.
Saves caleb_slim-predictions.csv in the format the grading notebook expects.
"""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path("/sessions/clever-hopeful-keller/mnt/CSE-450-Machine-learning")
NELS = ROOT / "nels_b"

DROP_FEATURES = ['job', 'marital', 'loan', 'emp.var.rate']

def build(drop_extra=None, tag="slim", seed=42, n_est=100):
    df = pd.read_csv(ROOT/"bank-data/bank.csv")
    df['never_contacted'] = np.where(df['pdays']==999,1,0)
    df = df.drop('pdays', axis=1)
    if drop_extra:
        for c in drop_extra:
            if c in df.columns:
                df = df.drop(c, axis=1)

    X = df.drop('y', axis=1)
    X_enc = pd.get_dummies(X, drop_first=True)
    train_cols = X_enc.columns
    le = LabelEncoder()
    y = le.fit_transform(df['y'])
    X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=0.2, random_state=seed)
    X_r, y_r = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
    m = RandomForestClassifier(
        n_estimators=n_est, max_depth=5,
        min_samples_leaf=10, min_samples_split=20,
        class_weight={0:1, 1:1.5}, random_state=seed, n_jobs=-1
    )
    m.fit(X_r, y_r)

    print(f"\n=== {tag.upper()}  (seed={seed}, n_est={n_est}) ===")
    print(f"Features used: {len(train_cols)} columns after one-hot")
    print(f"Internal test accuracy: {accuracy_score(y_te, m.predict(X_te)):.4f}")
    imp = pd.Series(m.feature_importances_, index=X_enc.columns).sort_values(ascending=False).head(8)
    print("Top 8 features:")
    print(imp.round(4).to_string())

    hold = pd.read_csv(ROOT/"bank-data/bank_holdout_test_mini.csv")
    hold.columns = [c.lstrip('﻿') for c in hold.columns]
    hold['never_contacted'] = np.where(hold['pdays']==999,1,0)
    hold = hold.drop('pdays', axis=1)
    if drop_extra:
        for c in drop_extra:
            if c in hold.columns:
                hold = hold.drop(c, axis=1)
    Xh = pd.get_dummies(hold, drop_first=True).reindex(columns=train_cols, fill_value=0)
    pred = le.inverse_transform(m.predict(Xh))
    print(f"Holdout: predicted yes = {(pred=='yes').sum()}/{len(pred)}")
    out = NELS / f"caleb_{tag}-predictions.csv"
    pd.DataFrame({'predictions': pred}).to_csv(out, index=False)
    print(f"Saved: {out}")
    return pred

# Slim model (the recommended drop list)
slim_pred = build(drop_extra=DROP_FEATURES, tag="slim")

# For comparison: also build "ultra-slim" = also drop default, day_of_week (neutral features)
ultra_pred = build(drop_extra=DROP_FEATURES + ['default','day_of_week'], tag="ultraslim")
