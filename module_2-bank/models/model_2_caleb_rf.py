import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

campaignData = pd.read_csv('../training_data/bank.csv')

campaignData['never_contacted'] = np.where(campaignData['pdays'] == 999, 1, 0)
campaignData = campaignData.drop('pdays', axis=1)

X = campaignData.drop(['y', 'month'], axis=1)
X_encoded = pd.get_dummies(X, drop_first=True)
training_column_names = X_encoded.columns

le = LabelEncoder()
y = le.fit_transform(campaignData['y'])

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  # type: ignore

model = RandomForestClassifier(
    max_depth=5,
    min_samples_leaf=10,
    min_samples_split=20,
    class_weight='balanced'
)

model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

print("--- RANDOM FOREST (BALANCED) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
print("\n--- TOP FEATURES ---")
print(importances.sort_values(ascending=False).head(10))


def predict_and_save(path, out_path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['never_contacted'] = np.where(df['pdays'] == 999, 1, 0)
    df = df.drop('pdays', axis=1)
    X_h = pd.get_dummies(df, drop_first=True)
    X_h = X_h.reindex(columns=training_column_names, fill_value=0)
    preds = model.predict(X_h)
    pd.DataFrame({'predicted_y': preds}).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


predict_and_save(
    '../test_data/bank_holdout_test.csv',
    '../predictions/holdout/NorthWindModule2_2_caleb_rf_holdout-predictions.csv'
)

predict_and_save(
    '../test_data/bank_holdout_test_mini.csv',
    '../predictions/mini-holdout/NorthWindModule2_2_caleb_rf_mini-holdout-predictions.csv'
)
