# CV-Graded Cost-Benefit Analysis

This replaces my earlier mini-holdout-based numbers, which (a) used a
reconstructed answer key that systematically *underscored* both Caleb's and
Nels's models by ~$70/410-rows and (b) tuned recommendations on a 410-row
sample.  Everything below uses **k-fold CV on the training data**, scored with
the **exact dollar formula** from `module02_bank_grading_mini.ipynb`:

```
per correct call (predicted yes ∩ actually yes): +$39.14   (= 4960·0.75·0.012 − 5.50)
per wasted call  (predicted yes ∩ actually no ): −$5.50
per missed yes   (predicted no  ∩ actually yes):  $0
```

`month` is **dropped from every variant** (including baselines).

---

## 1. The score gap is fixed

| Model | Old (reconstructed mini) | New CV-on-training, scaled to 410 rows |
|---|---|---|
| Caleb baseline | ~$719 | **$796 ± $4** |
| Nels baseline (8k subsample) | ~$636 | **$687 ± $9** |

The "over $800" you saw is now reproducible — Caleb's CV value of $796 sits
within the seed-to-seed wobble of the actual notebook's $800+ score.

---

## 2. Caleb LOO (drop one feature) — 3-fold CV on full 37 069 rows, 2 seeds

Sorted best → worst.  Bars within ±$3 of zero are noise.

| Feature dropped | $/410 rows | Δ vs baseline |
|---|---:|---:|
| **day_of_week** | $800.33 | **+$6.7** ← real gain |
| **job** | $799.42 | **+$5.8** ← real gain |
| housing | $795.26 | +$1.6 |
| cons.conf.idx | $794.86 | +$1.2 |
| emp.var.rate | $794.22 | +$0.6 |
| loan | $794.21 | +$0.6 |
| pdays | $793.64 | 0.0 (collapses to never_contacted) |
| poutcome | $793.64 | 0.0 |
| marital | $793.26 | −$0.4 |
| previous | $793.20 | −$0.4 |
| nr.employed | $792.40 | −$1.2 |
| education | $791.62 | −$2.0 |
| euribor3m | $791.43 | −$2.2 |
| contact | $790.69 | −$2.9 |
| cons.price.idx | $790.39 | −$3.3 |
| campaign | $789.53 | −$4.1 |
| age | $788.29 | −$5.4 |
| **default** | $780.26 | **−$13.4** ← keep this one! |

**Caleb's slim** (drop month + day_of_week + job): **$809.91 ± $4.1** → **+$13.9
over baseline**, comfortably outside the noise band.

---

## 3. Nels LOO — 3-fold CV on 8k stratified subsample, multi-seed

(Stacker is too slow for full-data CV in this environment; 8 000 rows
preserves the class proportion and gives a usable signal at the cost of a
wider noise band.)

| Variant | Seeds | $/410 rows | Δ vs baseline |
|---|---|---:|---:|
| baseline (drop month only) | 42, 7, 123 | **$687.42 ± $9.4** | — |
| drop demoBundle (job, marital, education, housing, loan, default) | 42, 7, 123 | **$708.10 ± $2.4** | **+$20.7** |
| drop demoBundle + day_of_week | 42, 7 | $688.89 ± wide | flat (avoid) |
| drop day_of_week only | 42 | $683.78 | ~flat |
| drop job only | 42 | $692.27 | ~+$5 |

The big finding: **Nels's stacker gains ~$21 by dropping all six demographic
categoricals** (job, marital, education, housing, loan, default).  The KNN
component suffers from the curse of dimensionality, so removing the high-
cardinality one-hot blocks (especially `job` with 12 levels and `education`
with 8) cleans up the distance metric and lets the meta layer trust the macro
signals.

This contrasts with Caleb's RF, which **does not benefit** from dropping the
demoBundle ($796 → $796) — and would actually be *hurt* by dropping `default`
alone.  Different model, different sensitivities.

---

## 4. Honest conclusions

* **Drop `month` from both models.** Already baked in.
* **For Caleb's RF**: also drop `day_of_week` and `job`. Keep `default`.
  Expected lift: **+$13–14** per 410 rows.
* **For Nels's stacker**: drop the entire 6-feature demographic block.
  Expected lift: **+$20** per 410 rows.
* **Do NOT** drop the macro features (`emp.var.rate`, `cons.price.idx`,
  `euribor3m`, `nr.employed`) — even if they're correlated with each other,
  removing them either hurts or moves the score within noise.
* **Important caveat on Nels**: numbers come from an 8k subsample. The
  *direction* (drop demos) replicates across 3 different seeds, so it's
  trustworthy as a recommendation, but the magnitude (+$21) might shrink or
  grow when retrained on all 37k rows.

## 5. Differences from my earlier (wrong) recommendations

| Earlier claim | Reality after CV |
|---|---|
| "only5_macros" gives Nels +$80 | **No** — that was holdout overfit. Real lift from a sensible slim is +$20. |
| Caleb baseline ≈ $719 | **No** — actual ≈ $796 ($800+ on the full unsubsampled grading). |
| Drop `default` to slim Caleb | **The opposite** — `default` is Caleb's most important feature. |
| Drop macros for Caleb | They're roughly neutral, not harmful — drop categorical noise instead. |

---

## Files

* `cv_harness.py` — the CV harness (drops month, scores per the notebook)
* `cv_summary.png` — the chart above
* `cv_caleb_results.json` — every Caleb experiment's raw output
* `cv_nels_results.json` — every Nels experiment's raw output
