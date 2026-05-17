from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
import polars as pl
import numpy as np

print("Loading data...")
data = pl.read_csv('../bank-data/bank.csv',schema_overrides={'nr.employed': pl.Float64})
mini_holdout = pl.read_csv('../bank-data/bank_holdout_test_mini.csv',schema_overrides={'nr.employed': pl.Float64})
print("Data loaded successfully.")
# print(data.schema)

X = data.drop('y')
y = data['y']

x_holdout = mini_holdout

print("Data preprocessing...")
# Encode target: 'yes' -> 1, 'no' -> 0
le = LabelEncoder()
y = le.fit_transform(y)

numerical_features = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

randomstate = None

rf = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=randomstate, n_jobs=-1, class_weight='balanced'))
])

knn = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier(n_neighbors=3, n_jobs=-1))
])

base = [
    ('rf', rf),
    ('knn', knn)
]

meta = LogisticRegression(random_state=randomstate, max_iter=20, class_weight={ 0: 1, 1: 4 })

stacking_clf = StackingClassifier(
    estimators=base,
    final_estimator=meta,
    cv=3,
    n_jobs=-1
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training stacking classifier...")
stacking_clf.fit(X_train, y_train)
print("Model trained successfully.")

print("Evaluating model...")
# Save predictions to CSV for each threshold
y_pred_proba_holdout = stacking_clf.predict_proba(x_holdout)[:, 1] # type: ignore
y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1] # type: ignore

threshold = 0.61


y_pred_holdout = (y_pred_proba_holdout >= threshold).astype(int)
y_pred = (y_pred_proba >= threshold).astype(int)
threshold_text = f"{threshold:.3f}"
filename = f"{threshold_text}-module2-predictions.csv"
predictions_data = pl.DataFrame({'predictions': y_pred_holdout})
predictions_data.write_csv(filename)
print(f"Saved predictions to {filename}")

print(classification_report(y_test, y_pred, target_names=le.classes_))