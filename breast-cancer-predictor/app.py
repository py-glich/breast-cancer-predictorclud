import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path so ml & backend can be imported
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Train model automatically on startup if not already trained
MODEL_FILE = ROOT_DIR / "ml" / "saved_models" / "breast_cancer_pipeline.joblib"
if not MODEL_FILE.exists():
    import subprocess
    subprocess.run([sys.executable, str(ROOT_DIR / "ml" / "train_model.py")], check=True)
    subprocess.run([sys.executable, str(ROOT_DIR / "ml" / "evaluate_model.py")], check=True)

from ml.predict import load_artifact, predict_one
from sklearn.datasets import load_breast_cancer

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Diagnostic Predictor")
st.caption("Educational Machine Learning demo powered by the Wisconsin Breast Cancer Dataset.")

st.warning(
    "⚠️ **Educational and Demonstration Use Only**: This tool is not medical software and "
    "must not be used for diagnosis, clinical evaluation, or treatment decisions."
)

tab_pred, tab_metrics, tab_about = st.tabs(["Prediction", "Model Metrics", "About Dataset"])

with tab_pred:
    col_ctrl, col_main = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Configuration")
        mode = st.radio("Feature Mode", ["quick", "full"], format_func=lambda x: "Quick (10 Mean Features)" if x == "quick" else "Full (All 30 Features)")
        
        st.write("---")
        st.subheader("Load Demo Samples")
        c1, c2 = st.columns(2)
        sample_to_load = None
        if c1.button("Benign Sample"):
            sample_to_load = "benign"
        if c2.button("Malignant Sample"):
            sample_to_load = "malignant"

    # Load artifact for feature names
    artifact = load_artifact(mode)
    feature_names = artifact["feature_names"]

    # Pre-fill sample if clicked
    defaults = {}
    if sample_to_load:
        dataset = load_breast_cancer(as_frame=True)
        target_val = 0 if sample_to_load == "malignant" else 1
        row = dataset.data[dataset.target == target_val].iloc[0]
        if mode == "quick":
            row = row[[c for c in row.index if c.startswith("mean ")]]
        defaults = row.to_dict()

    with col_main:
        st.subheader(f"Input Features ({len(feature_names)} features)")
        inputs = {}
        
        grid_cols = st.columns(2)
        for i, feat in enumerate(feature_names):
            with grid_cols[i % 2]:
                default_val = float(defaults.get(feat, 0.0))
                inputs[feat] = st.number_input(
                    feat.replace("_", " ").title(),
                    value=default_val,
                    format="%.4f",
                    key=f"{mode}_{feat}"
                )

        if st.button("Run Prediction", type="primary", use_container_width=True):
            try:
                res = predict_one(inputs, mode)
                is_malignant = res["prediction"] == "malignant"
                
                st.write("---")
                if is_malignant:
                    st.error(f"### Prediction: Malignant (Positive)")
                else:
                    st.success(f"### Prediction: Benign (Negative)")
                    
                p_mal = res["probability_malignant"]
                p_ben = res["probability_benign"]
                
                st.progress(p_mal, text=f"Malignant Probability: {p_mal * 100:.2f}%")
                
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Malignant Probability", f"{p_mal * 100:.2f}%")
                res_col2.metric("Benign Probability", f"{p_ben * 100:.2f}%")
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

with tab_metrics:
    st.subheader("Model Evaluation Metrics")
    import json
    metrics_path = ROOT_DIR / "ml" / "saved_models" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Overall Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        m_col2.metric("Malignant Recall", f"{metrics.get('malignant_recall', 0)*100:.2f}%")
        m_col3.metric("Malignant Precision", f"{metrics.get('malignant_precision', 0)*100:.2f}%")
        m_col4.metric("ROC AUC", f"{metrics.get('roc_auc_malignant', 0):.4f}")
        
        cm_path = ROOT_DIR / "ml" / "saved_models" / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix", use_container_width=False, width=500)
    else:
        st.info("Metrics not found. Please train and evaluate model.")

with tab_about:
    st.subheader("About the Wisconsin Breast Cancer Diagnostic Dataset")
    st.markdown("""
    Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 
    They describe characteristics of the cell nuclei present in the image.
    
    - **Total Features**: 30 nuclear characteristics
    - **Target**: Benign (Non-cancerous) or Malignant (Cancerous)
    - **Model**: Scikit-Learn Pipeline with Standard Scaling and Logistic Regression
    """)
