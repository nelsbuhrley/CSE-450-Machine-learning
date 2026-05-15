"""Run ONE experiment of Nels's stacking model on full bank.csv. Saves to JSON."""
import sys, time, json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

ROOT = Path("/sessions/clever-hopeful-keller/mnt/CSE-450-Machine-learning")
truth = pd.read_csv("/sessions/clever-hopeful-keller/mnt/outputs/analysis/mini_with_truth.csv")
y_true = truth['y_truth'].values
THRESH = 0.61
SEED = 42
RESULTS_FILE = "/sessions/clever-hopeful-keller/mnt/outputs/analysis/nels_full_loo.json"

def value(fp, tp): return fp*-5.5 + tp*-5.5 + tp*44.64

def run(drop_list, seed=SEED):
    data = pd.read_csv(ROOT/"bank-data/bank.csv")
    holdout = pd.read_csv(ROOT/"bank-data/bank_holdout_test_mini.csv")
    holdout.columns = [c.lstrip('﻿') for c in holdout.columns]
    for d in drop_list:
        if d in data.columns: data = data.drop(d, axis=1)
        if d in holdout.columns: holdout = holdout.drop(d, axis=1)

    X = data.drop('y', axis=1); y = data['y']
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    num_all = ['age','campaign','pdays','previous','emp.var.rate','cons.price.idx',
               'cons.conf.idx','euribor3m','nr.employed']
    cat_all = ['job','marital','education','default','housing','loan','contact','month','day_of_week']
    num = [c for c in num_all if c in X.columns]
    cat = [c for c in cat_all if c in X.columns]
    pre = ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat)
    ])
    rf = Pipeline([('pre',pre),('rf',RandomForestClassifier(n_estimators=10, max_depth=5,
                random_state=seed, n_jobs=-1, class_weight='balanced'))])
    knn = Pipeline([('pre',pre),('knn',KNeighborsClassifier(n_neighbors=3, n_jobs=-1, algorithm='ball_tree'))])
    meta = LogisticRegression(random_state=seed, max_iter=20, class_weight={0:1, 1:4})
    clf = StackingClassifier(estimators=[('rf',rf),('knn',knn)], final_estimator=meta, cv=3, n_jobs=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=seed)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(holdout)[:,1]
    pred = np.where(proba >= THRESH, 'yes', 'no')
    cm = confusion_matrix(pred, y_true, labels=['no','yes'])
    return value(cm[1][0], cm[1][1]), int(cm[1][1]), int(cm[1][0])

# Parse args: experiment name + comma-separated features to drop
exp_name = sys.argv[1]
drop_list = sys.argv[2].split(',') if (len(sys.argv) > 2 and sys.argv[2]) else []
seed = int(sys.argv[3]) if len(sys.argv) > 3 else SEED

# Load existing
results = {}
if Path(RESULTS_FILE).exists():
    with open(RESULTS_FILE) as fh: results = json.load(fh)

key = f"{exp_name}__seed{seed}"
if key in results:
    print(f"SKIP (already done): {key}")
    print(f"  v=${results[key]['v']:.2f}  tp={results[key]['tp']}  fp={results[key]['fp']}")
    sys.exit(0)

t0 = time.time()
v, tp, fp = run(drop_list, seed)
t = time.time() - t0
results[key] = {'v':v, 'tp':tp, 'fp':fp, 'drop':drop_list, 'time':t}
with open(RESULTS_FILE, 'w') as fh: json.dump(results, fh, indent=2)
print(f"DONE {key}: v=${v:.2f}  tp={tp}  fp={fp}  ({t:.1f}s)")
