import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')

df['never_contacted'] = np.where(df['pdays'] == 999, 1, 0)
df = df.drop('pdays', axis=1)

X = df.drop('y', axis=1)
X_encoded = pd.get_dummies(X, drop_first=True)

le = LabelEncoder()
y = le.fit_transform(df['y'])


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

refined_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=20,
    class_weight='balanced',
    random_state=42
)

refined_model.fit(X_train_resampled, y_train_resampled)

y_pred = refined_model.predict(X_test)

print("--- DECISION TREE MODEL ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

importances = pd.Series(
    refined_model.feature_importances_
)

print("\n--- TOP FEATURES ---")
print(importances.sort_values(ascending=False).head(10))