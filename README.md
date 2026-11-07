# ML Assignment 2 — Multi-Model Classification with Streamlit Deployment

M.Tech (AIML) — Machine Learning — BITS Pilani WILP

- **GitHub Repository:** `<paste your repo URL here>`
- **Live Streamlit App:** `<paste your *.streamlit.app URL here>`

---

## a. Problem Statement

Breast cancer is diagnosed partly from Fine Needle Aspirate (FNA) images of breast masses.
The goal is to build a binary classifier that predicts whether a tumour is **malignant** or
**benign** from 30 numeric cell-nucleus features, compare five classical ML classifiers on
the same train/test split using six evaluation metrics, and serve the trained models through
an interactive Streamlit web app that evaluates them on uploaded test data without retraining.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository (ID 17)
- **Source:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Instances:** 569 (meets minimum of 500)
- **Features:** 30 real-valued features (meets minimum of 12) + 1 target
- **Feature groups:** for each of 10 nucleus properties (radius, texture, perimeter, area,
  smoothness, compactness, concavity, concave points, symmetry, fractal dimension) the
  **mean**, **standard error** and **worst** (mean of the 3 largest values) are recorded
- **Target column:** `diagnosis` — `benign` (357, 62.7%) / `malignant` (212, 37.3%)
- **Missing values:** none
- **Type:** binary classification, mildly imbalanced
- **Positive class for Precision/Recall/F1:** `malignant` (label-encoded as 1)

### Preprocessing
- Duplicate rows dropped; target label-encoded (`benign`=0, `malignant`=1)
- Stratified 80/20 train/test split, `random_state=42` → 455 train / 114 test rows
- Numeric features: median imputation + `StandardScaler` (inside a sklearn `Pipeline`,
  so the scaler is fitted on training data only and saved with each model)
- The 114-row held-out split is saved as `test_data.csv` and used in the Streamlit app

## c. GitHub Repository

Link: `<paste your repo URL here>`

```
project-folder/
│-- app.py                     # Streamlit app (loads saved models, no retraining)
│-- train_models.py            # Training + evaluation script
│-- requirements.txt
│-- README.md
│-- test_data.csv              # Held-out test split (features + target)
│-- model/
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- support_vector_machine.joblib
│   │-- label_encoder.joblib
│   │-- metadata.json          # target name, feature list, classes
│   │-- metrics_comparison.csv
```

## d. Models Used

| Model | Key settings |
|---|---|
| Logistic Regression | L2, `max_iter=2000` |
| Decision Tree | `max_depth=6`, `min_samples_leaf=4` |
| KNN | `n_neighbors=7`, Euclidean |
| Naive Bayes | Gaussian NB (features are continuous) |
| Support Vector Machine | RBF kernel, `C=1.0`, `probability=True` |

### Comparison Table (test set, n = 114)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.8596 | 0.9271 | 0.9062 | 0.6905 | 0.7838 | 0.6965 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong (Acc 0.965, MCC 0.925) and the **highest AUC (0.996)**, meaning its probability ranking of malignant vs benign is nearly perfect. The classes are close to linearly separable once features are scaled, which suits a linear model. Missed 3 of 42 malignant cases. |
| Decision Tree | Weakest model (Acc 0.860, MCC 0.697). Recall of only 0.690 means ~13 malignant tumours were predicted benign — unacceptable in a medical setting. Axis-parallel splits struggle with the many correlated features (radius/perimeter/area), and a single tree has high variance on a 455-row training set. |
| KNN | Good (Acc 0.956, MCC 0.906) after standard scaling, which is essential for a distance-based method. Slightly lower recall (0.905) than LR/SVM; with 30 dimensions, distances become less discriminative. |
| Naive Bayes | Reasonable (Acc 0.921) with a high AUC (0.989), but lower recall (0.857) and MCC (0.829). Its feature-independence assumption is violated by strongly correlated features (radius, perimeter, area), which inflates over-confident probabilities. |
| Support Vector Machine | **Best overall**: highest Accuracy (0.974), F1 (0.963) and MCC (0.944), and Precision = 1.0 (no benign tumour was flagged malignant). The RBF kernel captures a smooth non-linear margin in the scaled feature space. |
| **Overall winner for this dataset** | **SVM (RBF)** — best on Accuracy, F1 and MCC (MCC is the most reliable single metric under class imbalance). Logistic Regression is a very close second with the best AUC and is the simpler, more interpretable choice. Since missing a malignant case is costly, the decision threshold of either model could be lowered to trade some precision for higher recall. |

## Streamlit App Features

- CSV upload of test data (sidebar) + one-click download of the sample `test_data.csv`
- Model selection dropdown (5 models)
- Six evaluation metrics shown as metric cards
- Confusion matrix heatmap and classification report
- Optional side-by-side comparison of all 5 models with best-model highlight
- Validates that uploaded CSV contains all required feature columns and the target

## How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py          # trains, saves models, writes test_data.csv
streamlit run app.py
```
# ML_Assignment2
