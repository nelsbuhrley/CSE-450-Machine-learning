---
layout: default
title: "Module 2 · Bank Marketing — Overview"
project: module_2-bank
project_title: "Module 2 · Bank Marketing"
page_type: overview
permalink: /module_2-bank/
---

{% include project-nav.html %}

# Module 2 — Bank Marketing Term-Deposit Prediction

**Course:** CSE 450 — Machine Learning · **Term:** Spring 2026 · **Team:** NorthWind (Caleb Dilley, Dallin Wagner, Jonathan Oliphant, Nels Buhrley)

A Portuguese retail bank ran direct-phone marketing campaigns to convince clients to subscribe to term deposits. Most calls ended in rejection, so unfiltered outreach burned more in labor than it earned in deposits. Our project asked: *can a classifier select the ~11% of clients who will actually subscribe, so the bank stops dialing the other ~89%?*

## Why this is an interesting ML problem

The dataset (UCI Bank Marketing — 37,069 records, 20 features) sits on top of three pedagogically rich tensions that show up everywhere in real classification work.

**Severe class imbalance.** Only 11.4% of records are positives. A model that always predicts "no" hits ~89% accuracy and is useless. Accuracy is the wrong metric.

**Cost-asymmetric errors.** A false negative is a missed deposit. A false positive is a $5.50 wasted call. A true positive is roughly $44.64 of expected value. The classifier needs to be aware that its mistakes don't cost the same.

**Operational realism.** The bank doesn't care about ROC-AUC. It cares about *campaign value* — dollars in versus dollars out across the call list. We had to translate the ML output into that metric and tune for it directly.

These are exactly the conditions where the textbook reflex — "tune for accuracy on a balanced dataset" — produces useless models. Module 2 is the part of the course where that reflex breaks.

## What this project demonstrates

The work covers the full classification pipeline end to end: exploratory data analysis on a tabular dataset with both numeric and categorical features; feature engineering (binary recode of `pdays` into `never_contacted`, dropping `month` to prevent temporal leakage); three distinct strategies for handling class imbalance (SMOTE, automatic class weighting, and probability-threshold tuning); a custom cost-sensitive evaluation function tied to a business metric; and a stacking ensemble that combines a global classifier (Random Forest) with a local one (KNN) under a tuned logistic meta-learner. Three models are compared head-to-head on a held-out test set and on a projected full-scale campaign.

## Headline result

All three of our models flip a money-losing campaign into a profitable one. On a 410-contact test, calling everyone loses **$157**; our best model (a stacking classifier with RF + KNN base learners and a tuned threshold of 0.61) returns **$824** — a precision of 47.2% versus the 11.5% baseline. Projected to a 4,119-contact campaign, the balanced-RF variant returns the highest projected value at **$7,775**.

![Campaign value with and without our model]({{ '/module_2-bank/visualisation/output/marketing/white_bars/campaign_value.png' | relative_url }}){: data-dark-src="{{ '/module_2-bank/visualisation/output/marketing/dark_bars/campaign_value.png' | relative_url }}" }

*Recent 410-contact test (left) and projected 4,119-contact campaign (right), with and without the model.*
{: .figure-caption }

## Concepts in use

| Concept | How it shows up |
|---|---|
| Class imbalance | SMOTE oversampling (Models 1 & 2), `class_weight='balanced'` (Models 2 & 3), threshold tuning (Model 3) |
| Feature engineering | `pdays == 999 → never_contacted`; `month` dropped to avoid temporal leakage |
| Encoding | `pd.get_dummies(drop_first=True)` and `OneHotEncoder(drop='first')` for categoricals; `StandardScaler` for numerics |
| Cost-sensitive evaluation | Custom business-value function: ~$44.64 per TP, ~$5.50 per FP; campaign value as primary metric |
| Ensemble methods | `StackingClassifier` with RF + KNN base learners and a `LogisticRegression` meta-learner over 3-fold CV |
| Threshold tuning | Probability cutoff swept from 0.5 → 0.61 to trade recall for precision |

## Next

See the **[Executive Summary]({{ '/module_2-bank/page/summary/' | relative_url }})** for the stakeholder-facing version of these findings, or the **[Technical Deep-Dive]({{ '/module_2-bank/page/technical/' | relative_url }})** for the pipeline, code, and evaluation methodology in detail.
