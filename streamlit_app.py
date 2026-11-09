"""Streamlit version of the breast cancer educational ML demo.

Run locally:   streamlit run streamlit_app.py
Streamlit Cloud: set "Main file path" to streamlit_app.py
Educational demonstration only - not a medical device.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))  # train/evaluate scripts import "preprocess"

import streamlit as st
from sklearn.datasets import load_breast_cancer

from ml.import_utils import parse_feature_file
from ml.predict import load_artifact, predict_one

MODEL_DIR = ROOT / "ml" / "saved_models"
FULL_MODEL = MODEL_DIR / "breast_cancer_pipeline.joblib"
QUICK_MODEL = MODEL_DIR / "breast_cancer_quick_pipeline.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

st.set_page_config(page_title="Breast Cancer ML Demo", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner="Training the models for the first time (about 30 seconds)...")
def ensure_models():
    """Train and evaluate the models if they do not exist yet (they are not in git)."""
    if not (FULL_MODEL.exists() and QUICK_MODEL.exists()):
        import train_model
        train_model.main()
    if not METRICS_FILE.exists():
        import evaluate_model
        evaluate_model.main()
    return True


@st.cache_data
def load_samples():
    data = load_breast_cancer(as_frame=True)
    return {
        "malignant": data.data[data.target == 0].iloc[0].to_dict(),
        "benign": data.data[data.target == 1].iloc[0].to_dict(),
    }


def widget_key(mode, name):
    return f"{mode}::{name}"


def fill_sample(kind, mode, names):
    sample = load_samples()[kind]
    for name in names:
        st.session_state[widget_key(mode, name)] = float(sample[name])
    st.session_state["result"] = None


def clear_form(mode, names):
    for name in names:
        st.session_state[widget_key(mode, name)] = None
    st.session_state["result"] = None


ensure_models()

# ---------- Header ----------
st.title("🔬 Breast tumour classification")
st.caption("Machine learning · classification · Wisconsin Breast Cancer Diagnostic dataset")
st.warning(
    "**Educational use only.** This is not a medical device and does not provide a "
    "diagnosis. It must not be used for screening or treatment decisions. "
    "Inputs must be FNA-derived measurements, not symptoms or photographs."
)

# ---------- Mode ----------
mode_label = st.radio(
    "Input mode",
    ["Quick · 10 mean values", "Full · 30 values"],
    horizontal=True,
)
mode = "quick" if mode_label.startswith("Quick") else "full"
artifact = load_artifact(mode)
names = artifact["feature_names"]

st.session_state.setdefault("result", None)

left, right = st.columns([1.6, 1], gap="large")

with left:
    st.subheader("Tumour measurements")
    st.write(
        "Enter the 10 average (mean) measurements from an FNA report."
        if mode == "quick"
        else "Enter all 30 measurements (mean, error and worst values) from an FNA report."
    )

    b1, b2, b3 = st.columns(3)
    b1.button("Demo: malignant sample", on_click=fill_sample, args=("malignant", mode, names), use_container_width=True)
    b2.button("Demo: benign sample", on_click=fill_sample, args=("benign", mode, names), use_container_width=True)
    b3.button("Clear", on_click=clear_form, args=(mode, names), use_container_width=True)

    upload = st.file_uploader("Or import a CSV / JSON file", type=None, key=f"upload-{mode}")
    if upload is not None:
        file_id = f"{upload.name}-{upload.size}-{mode}"
        if st.session_state.get("applied_upload") != file_id:
            try:
                parsed = parse_feature_file(upload.name, upload.getvalue(), names)
                for name, value in parsed.items():
                    st.session_state[widget_key(mode, name)] = value
                st.session_state["applied_upload"] = file_id
                st.session_state["result"] = None
                st.success("File imported. Review the values and press Run classification.")
            except Exception as error:  # noqa: BLE001 - show any parsing problem to the user
                st.session_state["applied_upload"] = file_id
                st.error(str(error))

    with st.form("predict-form"):
        columns = st.columns(2)
        for index, name in enumerate(names):
            columns[index % 2].number_input(
                name.capitalize(),
                value=None,
                format="%.5f",
                step=None,
                placeholder="Enter a number",
                key=widget_key(mode, name),
            )
        submitted = st.form_submit_button("Run classification", type="primary", use_container_width=True)

    if submitted:
        values = {name: st.session_state.get(widget_key(mode, name)) for name in names}
        missing = [n for n, v in values.items() if v is None]
        if missing:
            st.session_state["result"] = None
            st.error(f"Please enter a value for: {', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else ""))
        else:
            try:
                st.session_state["result"] = predict_one(values, mode)
            except ValueError as error:
                st.session_state["result"] = None
                st.error(str(error))

with right:
    st.subheader("Prediction result")
    result = st.session_state.get("result")
    if not result:
        st.info("Your model result will appear here after you submit all values.")
    else:
        if result["prediction"] == "malignant":
            st.error(f"### {result['prediction'].upper()}")
        else:
            st.success(f"### {result['prediction'].upper()}")
        for label in ("malignant", "benign"):
            p = result["probabilities"][label]
            st.write(f"{label.capitalize()} model probability: **{p * 100:.2f}%**")
            st.progress(min(max(p, 0.0), 1.0))
        st.caption(result["notice"])

    st.subheader("Model performance")
    if METRICS_FILE.exists():
        metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        shown = metrics.get("quick", metrics) if mode == "quick" else metrics
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{shown['accuracy'] * 100:.2f}%")
        m2.metric("Malignant recall", f"{shown['malignant_recall'] * 100:.2f}%")
        m3, m4 = st.columns(2)
        m3.metric("Malignant precision", f"{shown['malignant_precision'] * 100:.2f}%")
        m4.metric("ROC-AUC", f"{shown['roc_auc_malignant'] * 100:.2f}%")
        st.caption("Held-out test set of 114 samples; describes this educational dataset only.")

    with st.expander("Why can't I upload a photo?"):
        st.write(
            "This model was trained on numeric measurements taken from fine-needle aspirate "
            "(FNA) cell images, not on pictures. A photo of a person, a scan or a report image "
            "would give a meaningless result, so the app does not pretend to classify it. "
            "If you notice a lump or any change, please see a doctor."
        )

st.caption("Built with Streamlit and Scikit-learn · Public educational dataset")
