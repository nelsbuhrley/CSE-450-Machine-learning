import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

campaignData = pd.read_csv('../training_data/bank.csv')

campaignData['never_contacted'] = np.where(
    campaignData['pdays'] == 999,
    1,
    0
)

campaignData = campaignData.drop('pdays', axis=1)

X = campaignData.drop(['y', 'month'], axis=1)

X_encoded = pd.get_dummies(X, drop_first=True)

training_column_names = X_encoded.columns

le = LabelEncoder()

y = le.fit_transform(campaignData['y'])

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train) # type: ignore

refined_model = RandomForestClassifier(
    max_depth=5,
    min_samples_leaf=10,
    min_samples_split=20,
    class_weight={0: 1, 1: 1.5},
    random_state=42
)

refined_model.fit(X_train_resampled, y_train_resampled)

y_pred = refined_model.predict(X_test)

print("--- RANDOM FOREST MODEL ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=['0', '1']
))

importances = pd.Series(
    refined_model.feature_importances_,
    index=X_encoded.columns
)

print("\n--- TOP FEATURES ---")
print(importances.sort_values(ascending=False).head(10))

holdout_df = pd.read_csv('../test_data/bank_holdout_test.csv', encoding='utf-8-sig')

holdout_df['never_contacted'] = np.where(
    holdout_df['pdays'] == 999,
    1,
    0
)

holdout_df = holdout_df.drop('pdays', axis=1)

X_holdout_encoded = pd.get_dummies(
    holdout_df,
    drop_first=True
)

X_holdout_encoded = X_holdout_encoded.reindex(
    columns=training_column_names,
    fill_value=0
)

holdout_predictions_numeric = refined_model.predict(
    X_holdout_encoded
)

submission = pd.DataFrame({
    'predicted_y': holdout_predictions_numeric
})

submission.to_csv(
    '../predictions/holdout/NorthWindModule2_1_caleb_rf_holdout-predictions.csv',
    index=False
)

mini_holdout_df = pd.read_csv('../test_data/bank_holdout_test_mini.csv', encoding='utf-8-sig')
mini_holdout_df['never_contacted'] = np.where(mini_holdout_df['pdays'] == 999, 1, 0)
mini_holdout_df = mini_holdout_df.drop('pdays', axis=1)
X_mini_encoded = pd.get_dummies(mini_holdout_df, drop_first=True)
X_mini_encoded = X_mini_encoded.reindex(columns=training_column_names, fill_value=0)
mini_preds = refined_model.predict(X_mini_encoded)
pd.DataFrame({'predicted_y': mini_preds}).to_csv(
    '../predictions/mini-holdout/NorthWindModule2_1_caleb_rf_mini-holdout-predictions.csv',
    index=False
)

print("\nPrediction files created successfully.")