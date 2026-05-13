import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
train_campaignData = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')
train_campaignData['never_contacted'] = np.where(train_campaignData['pdays'] == 999, 1, 0)
train_campaignData = train_campaignData.drop('pdays', axis=1)

train_campaignData['y'] = train_campaignData['y'].map({'no': 0, 'yes': 1})

X_train = train_campaignData.drop('y', axis=1)
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
training_column_names = X_train_encoded.columns

le = LabelEncoder()
y_train = le.fit_transform(train_campaignData['y'])

model = RandomForestClassifier(max_depth=5, class_weight='{0: 1, 1: 1.5}')
model.fit(X_train_encoded, y_train)

new_df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test_mini.csv') 

new_df['never_contacted'] = np.where(new_df['pdays'] == 999, 1, 0)
new_df = new_df.drop('pdays', axis=1)

X_new_encoded = pd.get_dummies(new_df, drop_first=True)

X_new_encoded = X_new_encoded.reindex(columns=training_column_names, fill_value=0)

predictions_numeric = model.predict(X_new_encoded)
predictions_labels = le.inverse_transform(predictions_numeric)

new_df['predicted_y'] = predictions_labels

new_df[['predicted_y']].to_csv('NorthWindModule2-predictions.csv', index=False)
print(new_df.head())
"""
campaignData = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')

campaignData['never_contacted'] = np.where(campaignData['pdays'] == 999, 1, 0)
campaignData = campaignData.drop('pdays', axis=1)

X = campaignData.drop('y', axis=1)
X_encoded = pd.get_dummies(X, drop_first=True)

le = LabelEncoder()
y = le.fit_transform(campaignData['y'])


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
)

smote = SMOTE()

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

refined_model = RandomForestClassifier(
    max_depth=5,
    min_samples_leaf=10,
    min_samples_split=20,
    class_weight= {0: 1, 1: 1.5}
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

campaignData.to_csv('decision_tree_predictions.csv', index=False)
"""