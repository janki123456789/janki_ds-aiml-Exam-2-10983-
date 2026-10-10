<div align="center">

# 📊 Campaign Response Prediction & Audience Segmentation

### Data Science & AI/ML — Final Practical Exam (Set A)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Notebook-Executed-brightgreen)
![Data](https://img.shields.io/badge/Data-Synthetic-blue)

</div>

---

## 👩‍💻 Student Information

| Details          | Information                    |
| ---------------- | ------------------------------ |
| **Student Name** | Janki Dholariya                |
| **Exam Set**     | Set A                          |
| **Project Type** | Data Science & AI/ML Practical |
| **Language**     | Python                         |
| **Notebook**     | `exam.ipynb`                   |

---

## 🎯 Objective

Predict whether a customer **responds to a marketing campaign** (`response` = 1/0) and discover **audience segments** using:

1. 📐 Maths & Statistics → 2. 🧹 Preprocessing & Feature Engineering → 3. 🤖 Supervised Learning → 4. 🧩 Unsupervised Learning → 5. 🧠 ANN (Deep Learning)

> ⚠️ All data is **synthetic practice data**. No causal claims and no deployment-readiness claims are made.

---

## 🗂️ Dataset

- [set_d](data/raw/set_d.csv)

### Data Dictionary

| Column | Type | Description |
|---|---|---|
| `record_id` | Identifier | Unique customer record (**excluded from every model**) |
| `visits` | Numeric | Synthetic visit index (has missing values) |
| `recency` | Numeric | Synthetic recency index (has missing values) |
| `engagement` | Numeric | Synthetic engagement index |
| `spend` | Numeric | Synthetic spend index |
| `group` | Categorical | Operational cohort: `G1` / `G2` |
| `response` | **Target** | `1` = responded, `0` = no response |

---

## 🧹 Data Rules at a Glance

| Item | Value |
|---|---|
| Generator command | `python src/generate_data.py` (run once from repository root) |
| Raw rows → clean rows | **305 → 300** (5 exact duplicates removed *before* splitting) |
| Raw file | Never modified |
| Split | 80/20 stratified (`random_state=42`), then 20% of train for validation (`random_state=42`) |
| Partition sizes | **192 fit • 48 validation • 60 test** |
| Imputation | Median (fitted on fit set only) |
| Encoding | One-hot for `group`, `handle_unknown="ignore"` (kept unscaled) |
| Scaling | `StandardScaler` on 5 numeric features, fitted on fit set only |
| Engineered feature | `engineered_feature = engagement / (recency + 1)` (after imputation, before scaling) |
| Model seed | `42` |

---

## 🛠️ Tools & Versions

| Package | Version |
|---|---|
| Python | 3.12.3 |
| NumPy | 2.4.4 |
| pandas | 3.0.2 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| TensorFlow / Keras | 2.21.0 |

---

## 📁 Folder Map

```text
ds-aiml-set-d-YOUR-STUDENT-ID/
├── README.md                    ← you are here
├── requirements.txt             ← package versions
├── .gitignore
├── src/
│   └── generate_data.py         ← supplied generator (unchanged)
├── data/
│   └── raw/set_d.csv            ← original raw data
├── notebooks/
│   └── exam.ipynb               ← executed notebook (all 5 modules)
├── outputs/
│   ├── splits.csv               ← fit / validation / test IDs
│   ├── audit.json               ← shape, duplicates, missing, targets
│   ├── preprocessing_audit.csv
│   ├── stats_summary.json       ← descriptive stats + Welch test + CI
│   ├── linear_algebra.json      ← covariance + eigenvalues
│   ├── model_comparison.csv     ← baseline vs logistic vs ANN
│   ├── logistic_predictions.csv
│   ├── kmeans_scores.csv
│   ├── cluster_profiles.csv
│   ├── versions.json
│   └── figures/                 ← histogram, confusion matrices, k-means, loss curves
└── models/
    ├── ann_model.keras          ← trained ANN
    ├── ann.weights.h5
    ├── ann_predictions.csv
    └── preprocessing.joblib     ← fitted imputer + encoder + scaler
```

---

## 📐 Statistics (fit set, observed values only)

### Descriptive — `engagement` (n = 192)

| Mean | Median | Std (ddof=1) |
|---|---|---|
| 51.00 | 51.41 | 10.29 |

### Hypothesis Test — Welch t-test (two-sided, α = 0.05)

- **H0:** mean engagement of G1 = mean engagement of G2
- **H1:** mean engagement of G1 ≠ mean engagement of G2
- **Counts:** G1 = 84, G2 = 108
- **Result:** t = 0.723, **p = 0.471** → fail to reject H0 (no evidence of a difference)

### 95% Confidence Interval — overall mean engagement

**(49.54, 52.47)** — if sampling were repeated many times, about 95% of such intervals would contain the true mean.
**Assumptions:** independent records and approximately normal engagement (large n helps via the CLT). No causal interpretation.

### Linear Algebra — engagement & visits (n = 181 complete rows)

- Sample covariance computed manually as `X_centered.T @ X_centered / (n-1)`
- Eigenvalues via `np.linalg.eigh`
- **Largest eigenvalue / sum = 0.52** → PC1 explains ~52% of the variance, so the two variables are nearly uncorrelated and PC1 is weak.

---

## 🤖 Models & Held-out Results (same 60 test records)

**Target:** `response` • **Predictors:** `visits, recency, engagement, spend, engineered_feature, group_G1, group_G2` • **Threshold:** 0.5
**Undefined precision/recall** are handled with `zero_division=0`.

| Model | Settings |
|---|---|
| Baseline | `DummyClassifier(strategy="most_frequent")` |
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| ANN | 7 → 16 (ReLU) → 8 (ReLU) → 1 (sigmoid), 273 trainable params, binary cross-entropy, Adam (lr=0.001), batch 16, ≤ 50 epochs, early stopping (val_loss, patience 5, restore best weights), seed 42 |

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (Dummy) | 0.583 | 0.583 | 1.000 | 0.737 |
| Logistic Regression | 0.817 | 0.833 | 0.857 | 0.845 |
| **ANN** | **0.833** | **0.838** | **0.886** | **0.861** |

**Logistic Regression confusion matrix:** TN = 19, FP = 6, FN = 5, TP = 30.

**Interpretation:**
- A false positive wastes campaign cost; a false negative misses a likely responder (lost revenue).
- Both models clearly beat the baseline, but the ANN's F1 gain over Logistic Regression is only 0.016 (about 1 record), which is within noise on 60 records.
- **Model preference: Logistic Regression** — simpler, easier to explain, similar performance.
- The ANN ran all 50 epochs (best epoch = 50); train and validation losses are close, so there is no sign of overfitting and it may be slightly under-trained.

---

## 🧩 Clustering (K-Means, fit set only)

Features: 4 scaled numeric predictors + `engineered_feature` (no group one-hot, no target, no ID).

| k | Inertia | Silhouette |
|---|---|---|
| **2** | 713.03 | **0.226** |
| 3 | 613.55 | 0.187 |
| 4 | 541.65 | 0.191 |

**Chosen k = 2** (highest silhouette). A silhouette below 0.25 means the clusters are weak and overlapping.

| Cluster | Name | Size | Engagement | Recency | Suggested action |
|---|---|---|---|---|---|
| 0 | ❄️ Cool / Low-Engagement | 109 | 46.09 | 55.25 | Send a re-engagement offer |
| 1 | 🔥 Active / High-Engagement | 83 | 57.45 | 44.17 | Run a loyalty or upsell offer |

Cluster IDs are arbitrary labels and do **not** correspond to the target classes (response 0/1).

---

## 🔎 Key Findings

1. **Logistic Regression reached F1 = 0.845 and accuracy = 0.817** on the 60-record test set versus a baseline accuracy of 0.583 (about +23 points).
2. **Mean engagement does not differ significantly between G1 and G2** (Welch p = 0.471; 95% CI for overall mean 49.54–52.47).

**💡 Recommendation:** Use the Logistic Regression model to prioritise customers for campaigns, and target the low-engagement segment with re-engagement offers.

**⚠️ Limitation:** The test set has only 60 records and the data is synthetic, so metrics carry high uncertainty and the results are not proof of real-world or deployment readiness.

---


## 🎥 Project Explanation Video

🎬 **[Watch Project Explanation Video]()**

---

# 👩‍💻 Author

**Janki Dholariya**

Data Science & AI/ML Practical Project – Set A

---

