"""
Run Caleb's RandomForest model in two configurations:
  1. baseline   - identical to CalebsFolder/Decisiontree.py
  2. no_month   - same model, but the 'month' feature is dropped before training & inference

Reads from local files (no network) and produces:
  - nels_b/caleb_baseline-predictions.csv
  - nels_b/caleb_nomonth-predictions.csv
Both are single-column CSVs with header 'predictions' so they match the
module02_bank_grading_mini.ipynb format.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

ROOT = Path("/sessions/clever-hopeful-keller/mnt/CSE-450-Machine-learning")
NELS = ROOT / "nels_b"
TRAIN = ROOT / "bank-data/bank.csv"
HOLDOUT = ROOT / "bank-data/bank_holdout_test_mini.csv"

def build_and_predict(drop_month: bool, tag: str):
    print(f"\n{'='*60}\n  RUN: {tag}  (drop_month={drop_month})\n{'='*60}")
    df = pd.read_csv(TRAIN)
    df['never_contacted'] = np.where(df['pdays'] == 999, 1, 0)
    df = df.drop('pdays', axis=1)
    if drop_month:
        df = df.drop('month', axis=1)

    X = df.drop('y', axis=1)
    X_enc = pd.get_dummies(X, drop_first=True)
    train_cols = X_enc.columns

    le = LabelEncoder()
    y = le.fit_transform(df['y'])

    X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=0.2, random_state=42)
    X_tr_r, y_tr_r = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    model = RandomForestClassifier(
        max_depth=5, min_samples_leaf=10, min_samples_split=20,
        class_weight={0:1, 1:1.5}, random_state=42
    )
    model.fit(X_tr_r, y_tr_r)

    # Internal validation
    y_pred = model.predict(X_te)
    print(f"Internal test accuracy: {accuracy_score(y_te, y_pred):.4f}")
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    # Top features
    imp = pd.Series(model.feature_importances_, index=X_enc.columns).sort_values(ascending=False).head(10)
    print("Top 10 features:")
    print(imp.round(4).to_string())

    # Holdout
    hold = pd.read_csv(HOLDOUT)
    hold.columns = [c.lstrip('﻿') for c in hold.columns]
    hold['never_contacted'] = np.where(hold['pdays'] == 999, 1, 0)
    hold = hold.drop('pdays', axis=1)
    if drop_month:
        hold = hold.drop('month', axis=1)
    Xh = pd.get_dummies(hold, drop_first=True)
    Xh = Xh.reindex(columns=train_cols, fill_value=0)

    pred_num = model.predict(Xh)
    pred_lab = le.inverse_transform(pred_num)
    print(f"\nHoldout: predicted yes = {(pred_lab=='yes').sum()} of {len(pred_lab)}")

    out = NELS / f"caleb_{tag}-predictions.csv"
    pd.DataFrame({'predictions': pred_lab}).to_csv(out, index=False)
    print(f"Saved: {out}")
    return pred_lab

base_pred = build_and_predict(drop_month=False, tag="baseline")
nomo_pred = build_and_predict(drop_month=True,  tag="nomonth")

# Compare the two models head-to-head
print(f"\n{'='*60}\n  MODEL DISAGREEMENT\n{'='*60}")
print(f"Rows where the two models DISAGREE: {(base_pred != nomo_pred).sum()} / {len(base_pred)}")
disagree_to_no = ((base_pred=='yes') & (nomo_pred=='no')).sum()
disagree_to_yes = ((base_pred=='no')  & (nomo_pred=='yes')).sum()
print(f"  baseline=yes, no-month=no:  {disagree_to_no}")
print(f"  baseline=no,  no-month=yes: {disagree_to_yes}")
