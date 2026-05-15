# cv/harness.py
# Cross-validated cost-benefit scoring (no holdout overfitting, no reconstructed truth).
# - Drops `month` from EVERY variant (per user direction).
# - Uses the EXACT $/call formulas from module02_bank_grading_mini.ipynb:
#     wasted (predicted yes, actually no): -$5.50
#     correct (predicted yes, actually yes): +$39.14   (=44.64 benefit - 5.50 call cost)
#     missed yes (predicted no, actually yes): $0
# - Reports per-row $ AND scaled-to-410-rows $ for comparability with the mini-grading.

import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = Path("/sessions/clever-hopeful-keller/mnt/CSE-450-Machine-learning")
BANK = ROOT / "bank-data" / "bank.csv"

# === EXACT VALUES FROM THE GRADING NOTEBOOK ===
TIME_ON_CALL = 0.5
WAGE = -11
CALL_COST = WAGE * TIME_ON_CALL              # -5.50
AVG_SAV = 4960
PCT_TERM = 0.75
NIM = 0.012
POS_BENEFIT = AVG_SAV * PCT_TERM * NIM       # 44.64
# total = wasted*CALL_COST + correct*CALL_COST + correct*POS_BENEFIT
# net per correct = -5.50 + 44.64 = 39.14
# net per wasted  = -5.50

def value_of_calls(wasted: int, correct: int) -> float:
    return wasted*CALL_COST + correct*CALL_COST + correct*POS_BENEFIT

def grade_predictions(y_true_yes_idx: np.ndarray, pred_yes_mask: np.ndarray) -> dict:
    """y_true_yes_idx: 0/1 array of true labels; pred_yes_mask: 0/1 array of predictions."""
    correct = int(((pred_yes_mask == 1) & (y_true_yes_idx == 1)).sum())
    wasted  = int(((pred_yes_mask == 1) & (y_true_yes_idx == 0)).sum())
    return {"correct": correct, "wasted": wasted, "value": value_of_calls(wasted, correct)}

# === DATA LOADING (always drops 'month' per user direction) ===
NUM_FEATURES_FULL = ['age','campaign','pdays','previous','emp.var.rate',
                     'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
CAT_FEATURES_FULL = ['job','marital','education','default','housing','loan',
                     'contact','month','day_of_week']
ALWAYS_DROP = ['month']  # baked in per user request

def load_data(extra_drop=None):
    extra_drop = list(extra_drop or [])
    drop = list(set(ALWAYS_DROP + extra_drop))
    df = pd.read_csv(BANK)
    keep = [c for c in df.columns if c not in drop]
    df = df[keep].copy()
    le = LabelEncoder()
    y = le.fit_transform(df['y'])  # 'no'->0, 'yes'->1 alphabetical
    X = df.drop('y', axis=1)
    num = [c for c in NUM_FEATURES_FULL if c in X.columns]
    cat = [c for c in CAT_FEATURES_FULL if c in X.columns]
    return X, y, num, cat, drop

# === MODEL FACTORIES ===

def make_caleb(seed: int, num: list, cat: list):
    """Caleb's RandomForest with SMOTE.  Uses pdays->never_contacted-only feature engineering
    via a tiny custom pre-step: we just drop pdays and add never_contacted in the dataframe before
    handing to the pipeline (done in run_caleb_fold, since pipeline-side it's simpler)."""
    pre = ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat)
    ])
    rf = RandomForestClassifier(
        max_depth=5, min_samples_leaf=10, min_samples_split=20,
        class_weight={0:1, 1:1.5}, random_state=seed, n_jobs=-1
    )
    smote = SMOTE(random_state=seed)
    return ImbPipeline([('pre', pre), ('smote', smote), ('rf', rf)])

def make_nels(seed: int, num: list, cat: list):
    pre = ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat)
    ])
    rf = SkPipeline([('pre', pre),
                     ('rf', RandomForestClassifier(n_estimators=10, max_depth=5,
                              random_state=seed, n_jobs=-1, class_weight='balanced'))])
    knn = SkPipeline([('pre', pre),
                      ('knn', KNeighborsClassifier(n_neighbors=3, n_jobs=-1, algorithm='ball_tree'))])
    meta = LogisticRegression(random_state=seed, max_iter=100, class_weight={0:1, 1:4})
    return StackingClassifier(estimators=[('rf', rf), ('knn', knn)],
                              final_estimator=meta, cv=3, n_jobs=1)

# === CALEB-SPECIFIC FE ===
def caleb_fe(X: pd.DataFrame) -> pd.DataFrame:
    Z = X.copy()
    if 'pdays' in Z.columns:
        Z['never_contacted'] = (Z['pdays'] == 999).astype(int)
        Z = Z.drop('pdays', axis=1)
    return Z

# === CV CORE ===
def cv_value(model_kind: str, extra_drop=None, n_splits=5, seeds=(42, 7, 123),
             threshold=None, verbose=False, subsample=None):
    """
    model_kind: 'caleb' or 'nels'
    subsample: if set, stratified-subsample to N rows BEFORE CV (for slow models).
    Returns dict with per-row $ + total over CV (sum across val folds).
    """
    X_raw, y, num, cat, drops = load_data(extra_drop=extra_drop)
    if subsample is not None and subsample < len(y):
        from sklearn.model_selection import train_test_split
        X_raw, _, y, _ = train_test_split(X_raw, y, train_size=subsample,
                                          stratify=y, random_state=12345)
        # reset index to keep iloc happy
        X_raw = X_raw.reset_index(drop=True)

    fold_values_per_seed = []  # list of lists
    fold_correct_per_seed = []
    fold_wasted_per_seed = []
    n_total = len(y)

    for seed in seeds:
        # for Caleb, do FE before splitting (only column-wise transform, no fit)
        if model_kind == 'caleb':
            X = caleb_fe(X_raw)
            num_eff = [c for c in num if c in X.columns]
            cat_eff = [c for c in cat if c in X.columns]
        else:
            X = X_raw
            num_eff, cat_eff = num, cat

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        v_seed, c_seed, w_seed = [], [], []
        for fi, (tr, te) in enumerate(skf.split(X, y)):
            t0 = time.time()
            if model_kind == 'caleb':
                clf = make_caleb(seed, num_eff, cat_eff)
                clf.fit(X.iloc[tr], y[tr])
                if threshold is None:
                    pred = clf.predict(X.iloc[te])
                else:
                    proba = clf.predict_proba(X.iloc[te])[:,1]
                    pred = (proba >= threshold).astype(int)
            else:
                clf = make_nels(seed, num_eff, cat_eff)
                clf.fit(X.iloc[tr], y[tr])
                proba = clf.predict_proba(X.iloc[te])[:,1]
                thr = 0.61 if threshold is None else threshold
                pred = (proba >= thr).astype(int)

            res = grade_predictions(y[te], pred)
            v_seed.append(res['value'])
            c_seed.append(res['correct'])
            w_seed.append(res['wasted'])
            if verbose:
                print(f"  seed={seed} fold={fi} val_n={len(te)} correct={res['correct']} "
                      f"wasted={res['wasted']} value=${res['value']:.2f} ({time.time()-t0:.1f}s)")
        fold_values_per_seed.append(v_seed)
        fold_correct_per_seed.append(c_seed)
        fold_wasted_per_seed.append(w_seed)

    # Per-seed totals (sum of fold values = score over the full training set passed once)
    per_seed_total = [sum(vs) for vs in fold_values_per_seed]
    per_seed_correct = [sum(cs) for cs in fold_correct_per_seed]
    per_seed_wasted = [sum(ws) for ws in fold_wasted_per_seed]

    mean_total = float(np.mean(per_seed_total))
    std_total  = float(np.std(per_seed_total, ddof=1)) if len(per_seed_total) > 1 else 0.0
    per_row = mean_total / n_total
    scaled_410 = per_row * 410

    return {
        "model": model_kind,
        "drops": drops,
        "n_splits": n_splits,
        "seeds": list(seeds),
        "n_total": n_total,
        "per_seed_total_value": per_seed_total,
        "per_seed_correct": per_seed_correct,
        "per_seed_wasted": per_seed_wasted,
        "mean_total_value": mean_total,
        "std_total_value": std_total,
        "per_row_value": per_row,
        "scaled_410_value": scaled_410,
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=['caleb','nels'])
    ap.add_argument("--drop", default="", help="comma-sep extra features to drop (besides month)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="exp")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--subsample", type=int, default=None)
    args = ap.parse_args()

    extra = [x for x in args.drop.split(",") if x.strip()]
    seeds = tuple(int(s) for s in args.seeds.split(","))
    res = cv_value(args.model, extra_drop=extra, n_splits=args.folds,
                   seeds=seeds, threshold=args.threshold, verbose=args.verbose,
                   subsample=args.subsample)
    res["label"] = args.label
    print(json.dumps(res, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if path.exists():
            existing = json.loads(path.read_text())
        existing.append(res)
        path.write_text(json.dumps(existing, indent=2))
