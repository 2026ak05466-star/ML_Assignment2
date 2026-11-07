"""
ML Assignment 2 - Training script
Trains 5 classifiers on one dataset, saves models + preprocessing artifacts,
writes a held-out test_data.csv and a metrics comparison table.

Usage (on BITS Virtual Lab):
    python train_models.py                                   # default: Breast Cancer Wisconsin (UCI)
    python train_models.py --csv data/your.csv --target your_target_column
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
MODEL_DIR = "model"


def load_dataset(csv_path, target):
    """Load the CSV given by the user, or the default UCI Breast Cancer dataset."""
    if csv_path:
        df = pd.read_csv(csv_path)
        return df, target
    from sklearn.datasets import load_breast_cancer
    bc = load_breast_cancer(as_frame=True)
    df = bc.frame.copy()
    # 0 = malignant, 1 = benign in sklearn -> make it readable
    df["target"] = df["target"].map({0: "malignant", 1: "benign"})
    df = df.rename(columns={"target": "diagnosis"})
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df, "diagnosis"


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                  sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=4,
                                                random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True,
                                      random_state=RANDOM_STATE),
    }


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    avg = "binary" if n_classes == 2 else "weighted"
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df, target = load_dataset(args.csv, args.target)
    df = df.dropna(subset=[target]).drop_duplicates()
    X = df.drop(columns=[target])
    y_raw = df[target].astype(str)

    print(f"Dataset shape: {df.shape}  |  features: {X.shape[1]}  |  target: {target}")
    print("Class distribution:\n", y_raw.value_counts())

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE)

    # Save held-out test data (features + original target labels) for the Streamlit app
    test_df = X_test.copy()
    test_df[target] = le.inverse_transform(y_test)
    test_df.to_csv("test_data.csv", index=False)

    os.makedirs(MODEL_DIR, exist_ok=True)
    preprocessor = build_preprocessor(X_train)
    results = []
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_proba, n_classes)
        results.append({"ML Model Name": name, **{k: round(v, 4) for k, v in m.items()}})
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
        print(f"Saved {fname}: {m}")

    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    meta = {"target": target, "features": X.columns.tolist(),
            "classes": le.classes_.tolist(), "n_train": len(X_train), "n_test": len(X_test)}
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"), index=False)
    print("\n=== Comparison Table ===")
    print(res_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
