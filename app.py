"""
ML Assignment 2 - Streamlit app
Loads the 5 pre-trained models and evaluates them on an uploaded TEST csv.
No retraining happens in this app.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Classifier Bench | ML Assignment 2",
                   page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifacts():
    models = {n: joblib.load(os.path.join(MODEL_DIR, f)) for n, f in MODEL_FILES.items()}
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return models, le, meta


def evaluate(model, X, y, n_classes):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    avg = "binary" if n_classes == 2 else "weighted"
    try:
        auc = (roc_auc_score(y, y_proba[:, 1]) if n_classes == 2
               else roc_auc_score(y, y_proba, multi_class="ovr", average="weighted"))
    except ValueError:
        auc = float("nan")  # e.g. only one class present in uploaded file
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


models, le, meta = load_artifacts()
target, features, classes = meta["target"], meta["features"], meta["classes"]

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload TEST data (CSV)", type=["csv"])
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
compare_all = st.sidebar.checkbox("Also compare all 5 models", value=True)
st.sidebar.caption(f"Target column expected: `{target}`  \nClasses: {', '.join(classes)}")

if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button("⬇️ Download sample test_data.csv", f,
                                   file_name="test_data.csv")

# ---------------- Main ----------------
st.title("🩺 Classifier Bench — ML Assignment 2")
st.write("Upload the held-out test CSV, pick a model, and see how the **pre-trained** "
         "model performs. Models are loaded from `model/` — nothing is retrained here.")

if uploaded is None:
    st.info("👈 Upload a test CSV from the sidebar to begin "
            "(you can download the sample test_data.csv there).")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

unknown = set(df[target].astype(str)) - set(classes)
if unknown:
    st.error(f"Target contains labels not seen during training: {unknown}")
    st.stop()

X = df[features]
y = le.transform(df[target].astype(str))

with st.expander(f"Preview uploaded data ({df.shape[0]} rows × {df.shape[1]} cols)"):
    st.dataframe(df.head(20), width="stretch")

# ----- Selected model -----
metrics, y_pred = evaluate(models[model_name], X, y, len(classes))
st.subheader(f"📊 Evaluation metrics — {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y, y_pred, labels=range(len(classes)),
                                target_names=classes, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")

# ----- All models -----
if compare_all:
    st.subheader("🏁 All models on this test data")
    rows = []
    for name, mdl in models.items():
        m, _ = evaluate(mdl, X, y, len(classes))
        rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in m.items()}})
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp.style.highlight_max(axis=0, color="#c6efce"),
                 width="stretch")
    best = comp["F1"].idxmax()
    st.success(f"Best model on this upload by F1: **{best}** "
               f"(F1 = {comp.loc[best, 'F1']:.4f}, MCC = {comp.loc[best, 'MCC']:.4f})")
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
