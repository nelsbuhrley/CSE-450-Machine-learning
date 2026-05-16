# Module 2 - Bank Marketing Term Deposit Prediction

## Problem Statement

A Portuguese bank ran direct phone marketing campaigns to convince clients to subscribe to term deposits. The majority of calls result in rejection, making unfiltered campaigns costly. Our task: build a classifier that predicts which clients will subscribe, so the bank can focus its calls on high-probability leads and turn a losing campaign into a profitable one.

The dataset originates from the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and contains 37,069 client records with 20 features spanning demographics, economic indicators, and prior campaign history. The target class is heavily imbalanced -- only 11.4% of clients subscribed.

## Theoretical Background

### Class Imbalance

With an 88.6/11.4 split, a naive classifier that always predicts "no" achieves ~89% accuracy but is useless for our goal. We addressed this through:

- **SMOTE (Synthetic Minority Oversampling Technique):** Generates synthetic positive-class samples by interpolating between existing minority instances in feature space. Used in Models 1 and 2 during training.
- **Class weighting:** Penalizes misclassification of the minority class more heavily in the loss function, effectively shifting the decision boundary. Used across all three models.
- **Threshold tuning:** Rather than using the default 0.5 probability cutoff, Model 3 tunes the classification threshold (0.61) to balance precision and recall for our cost-sensitive objective.

### Cost-Sensitive Evaluation

Accuracy is the wrong metric here. We defined a business value function based on realistic assumptions about bank employee wages (~$11/hr), call duration (~30 min), average savings ($4,960), term deposit allocation (75%), and net interest margin (1.2%). Each correct positive call generates ~$44.64 in expected value; each wasted call costs ~$5.50. This framing means a model that makes fewer but more precise calls can outperform one with higher recall.

## Our Models

### Model 1 - Random Forest with SMOTE (Caleb)

**File:** `models/model_1_caleb_rf.py`

- Engineered a binary `never_contacted` feature from `pdays` (999 = never contacted)
- Dropped `month` to prevent temporal leakage
- Applied SMOTE to the training split, then trained a `RandomForestClassifier` with `max_depth=5`, `min_samples_leaf=10`, `min_samples_split=20`, and manual class weights `{0: 1, 1: 1.5}`
- One-hot encoded categoricals with `pd.get_dummies(drop_first=True)`

**Result:** Broadest call list (129 calls on mini-holdout). Catches the most true positives (31) but at lower precision (24.0%), diluting per-call value.

### Model 2 - Random Forest, Balanced Weights (Caleb)

**File:** `models/model_2_caleb_rf.py`

- Same pipeline as Model 1 but switched to `class_weight='balanced'`, which automatically sets weights inversely proportional to class frequency
- Removed the fixed `random_state` on the classifier to allow variance across runs

**Result:** Tighter call list (82 calls). Higher precision (34.1%) and the best projected total value at scale ($7,775 on 4,119 contacts).

### Model 3 - Stacking Classifier: RF + KNN (Nels)

**File:** `models/model_3_nels_stack_rf_knn.py`

- Used `sklearn.ensemble.StackingClassifier` with two base learners:
  - **Random Forest** (`n_estimators=10`, `max_depth=5`, `class_weight='balanced'`)
  - **K-Nearest Neighbors** (`n_neighbors=3`)
- Meta-learner: `LogisticRegression` with `class_weight={0: 1, 1: 4}` and `max_iter=20`
- Both base learners wrapped in their own `Pipeline` with `StandardScaler` (numerics) + `OneHotEncoder(drop='first')` (categoricals)
- 3-fold cross-validation for generating meta-features
- Probability threshold tuned to 0.61
- Built on Polars for data loading

**Result:** Most selective call list (53 calls). Highest precision (47.2%) and best per-call value ($15.56). The stacking architecture lets KNN capture local neighborhood patterns that complement the RF's global splits.

## Results

### Campaign Value

![Campaign Value](visualisation/output/marketing/dark_bars/campaign_value.png)

| Model | Calls | TP | Precision | Campaign Value |
|-------|------:|---:|----------:|---------------:|
| No model (call everyone) | 410 | 47 | 11.5% | **$-156.92** |
| Model 1 (RF + SMOTE) | 129 | 31 | 24.0% | $674.34 |
| Model 2 (RF balanced) | 82 | 28 | 34.1% | $798.92 |
| Model 3 (Stacking) | 53 | 25 | 47.2% | **$824.50** |

### Value per Call

| Model | Value / Call | Lift vs. Baseline |
|-------|------------:|-----------:|
| Model 3 (Stacking) | **$15.56** | +$15.94 |
| Model 2 (RF balanced) | $9.74 | +$10.13 |
| Model 1 (RF + SMOTE) | $5.23 | +$5.61 |
| No model | $-0.38 | -- |

### Projected Full Campaign (4,119 contacts)

| Model | Projected Calls | Projected Value |
|-------|----------------:|----------------:|
| Model 2 (RF balanced) | 798 | **$7,774.86** |
| Model 3 (Stacking) | 484 | $7,529.40 |
| Model 1 (RF + SMOTE) | 1,268 | $6,628.40 |
| No model | 4,119 | $-1,576.47 |

### Call List Filtering

![Golden List / Black List](visualisation/output/marketing/dark_bars/golden_blacklist.png)

The models reshape the call list by concentrating effort on high-conversion groups and filtering out low-yield contacts:

| Group | Unfiltered | Model 3 | Shift |
|-------|----------:|--------:|------:|
| Previously converted | 2.9% | 20.8% | +17.8pp |
| Retirees | 4.9% | 22.6% | +17.8pp |
| Students | 2.9% | 9.4% | +6.5pp |
| Reached by landline | 37.1% | 9.4% | -27.6pp |
| Blue-collar workers | 20.7% | 1.9% | -18.8pp |
| Basic education (9yr) | 15.9% | 1.9% | -14.0pp |

## Key Takeaways

- All three models turn a money-losing campaign profitable
- **Model 3** maximizes per-call efficiency -- best choice when call capacity is limited
- **Model 2** maximizes total projected value at scale -- best choice when you can staff more callers
- The stacking architecture in Model 3 achieves nearly 5x the precision of the unfiltered baseline while only sacrificing 22 true positives out of 47

## Directory Structure

```
models/              Model training scripts (3 models)
predictions/
  holdout/           Full holdout predictions per model
  mini-holdout/      Mini holdout predictions per model
training_data/       Bank marketing training CSV
test_data/           Holdout and mini-holdout test CSVs
visualisation/       Analysis scripts and output plots
  output/
    detailed/        Technical analysis report and plots
    marketing/       Stakeholder-facing report and plots
dashboards/          Interactive HTML dashboard
verification/        Grading notebook and scripts
```
