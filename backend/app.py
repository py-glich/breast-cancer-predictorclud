"""Streamlit version of the breast cancer interactive educational & assessment app.

Features:
1. Dynamic Conversational Symptoms & Risk Assessment (Questionnaire with adaptive follow-ups)
2. Wisconsin FNA Biopsy Diagnostic Model (Full 30 or Quick 10 mean features)
3. Model Metrics & Confusion Matrix
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

import streamlit as st
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

from ml.import_utils import parse_feature_file
from ml.predict import load_artifact, predict_one

MODEL_DIR = ROOT / "ml" / "saved_models"
FULL_MODEL = MODEL_DIR / "breast_cancer_pipeline.joblib"
QUICK_MODEL = MODEL_DIR / "breast_cancer_quick_pipeline.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

st.set_page_config(
    page_title="Breast Health & Cancer Risk Assessment",
    page_icon="🔬",
    layout="wide"
)

# ---------- Cache and helpers ----------
@st.cache_resource(show_spinner="Preparing models...")
def ensure_models():
    if not (FULL_MODEL.exists() and QUICK_MODEL.exists()):
        try:
            import train_model
            train_model.main()
        except Exception:
            pass
    if not METRICS_FILE.exists():
        try:
            import evaluate_model
            evaluate_model.main()
        except Exception:
            pass
    return True

@st.cache_data
def load_samples():
    data = load_breast_cancer(as_frame=True)
    return {
        "malignant": data.data[data.target == 0].iloc[0].to_dict(),
        "benign": data.data[data.target == 1].iloc[0].to_dict(),
    }

ensure_models()

# ---------- Header ----------
st.title("🔬 Breast Health Assessment & Cancer Prediction System")
st.caption("Interactive Symptom Checker & Machine Learning Diagnostic Predictor")

st.info(
    "ℹ️ **Clinical Advisory**: This system is designed for educational and preliminary screening awareness. "
    "Only qualified healthcare professionals with biopsy pathology and mammography can diagnose breast cancer."
)

tab_symp, tab_fna, tab_metrics, tab_info = st.tabs([
    "🩺 Symptom & Risk Questionnaire",
    "🔬 Lab Biopsy (FNA) Classifier",
    "📊 Model Performance & Metrics",
    "ℹ️ Clinical Guidance"
])

# =========================================================================
# TAB 1: SYMPTOM ASSESSMENT WITH COUNTER-QUESTIONS
# =========================================================================
with tab_symp:
    st.subheader("Step-by-Step Symptom & Risk Evaluation")
    st.write(
        "Answer the questions below. If any symptoms are reported, adaptive follow-up questions will appear "
        "to evaluate clinical risk indicators before producing an assessment."
    )
    
    with st.form("symptom_form"):
        # Section 1: Demographics & Context
        st.markdown("#### 1. General Profile")
        col_age, col_gen = st.columns(2)
        with col_age:
            age = st.number_input("Age (years)", min_value=18, max_value=110, value=40, step=1)
        with col_gen:
            gender = st.selectbox("Biological Sex", ["Female", "Male", "Other"])

        st.markdown("---")
        # Section 2: Primary Symptoms
        st.markdown("#### 2. Primary Symptom Check")
        has_lump = st.radio(
            "Have you noticed a new lump, mass, or thickening in either breast or underarm area?",
            ["No", "Yes"],
            index=0,
            horizontal=True
        ) == "Yes"

        # Adaptive follow-up questions for lump
        lump_firmness = "Soft / tender"
        lump_mobility = "Moves freely"
        lump_duration = "Less than 2 weeks"
        if has_lump:
            st.warning("🔎 **Follow-up questions regarding the lump:**")
            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                lump_firmness = st.selectbox(
                    "Texture / Firmness",
                    ["Soft / spongy / cystic", "Rubbery / smooth", "Hard / rigid / irregular edge"]
                )
            with col_l2:
                lump_mobility = st.selectbox(
                    "Mobility",
                    ["Moves easily when pressed", "Firmly attached / fixed in place"]
                )
            with col_l3:
                lump_duration = st.selectbox(
                    "Duration of lump",
                    ["Less than 2 weeks (e.g. menstrual-related)", "2 to 6 weeks", "More than 6 weeks / Growing"]
                )

        st.markdown("---")
        has_skin_changes = st.radio(
            "Are there noticeable skin changes (dimpling, puckering, redness, or orange-peel texture)?",
            ["No", "Yes"],
            index=0,
            horizontal=True
        ) == "Yes"

        skin_detail = "None"
        if has_skin_changes:
            st.warning("🔎 **Follow-up question regarding skin changes:**")
            skin_detail = st.selectbox(
                "Specific appearance",
                ["Redness / mild swelling", "Dimpling / puckering (peau d'orange)", "Skin retraction / ulceration"]
            )

        st.markdown("---")
        has_nipple_changes = st.radio(
            "Have you observed nipple retraction, inversion, or spontaneous fluid discharge?",
            ["No", "Yes"],
            index=0,
            horizontal=True
        ) == "Yes"

        nipple_detail = "None"
        if has_nipple_changes:
            st.warning("🔎 **Follow-up question regarding nipple changes:**")
            nipple_detail = st.selectbox(
                "Nature of discharge or change",
                ["Milky or clear discharge (both breasts)", "Spontaneous bloody or serous discharge from one breast", "Recent inverted / retracted nipple"]
            )

        st.markdown("---")
        # Section 3: History & Genetic Risks
        st.markdown("#### 3. Family & Personal History")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            family_history = st.selectbox(
                "Family history of breast/ovarian cancer (1st-degree relatives: mother, sister, daughter)?",
                ["No family history", "One relative diagnosed after age 50", "One relative diagnosed before age 50", "Multiple relatives or known BRCA1/BRCA2 mutation"]
            )
        with col_h2:
            prev_biopsy = st.selectbox(
                "Previous breast biopsies or atypical hyperplasia?",
                ["No prior biopsies", "Benign biopsy (fibroadenoma, cysts)", "Atypical ductal/lobular hyperplasia"]
            )

        evaluate_btn = st.form_submit_button("Analyze Symptoms & Predict Risk", type="primary", use_container_width=True)

    if evaluate_btn:
        # Clinical Risk Scoring Algorithm based on evidence-based screening criteria
        risk_score = 0
        reasons = []

        # Age scoring
        if age >= 55:
            risk_score += 15
            reasons.append("Age >= 55 (increased baseline demographic risk)")
        elif age >= 40:
            risk_score += 8

        # Lump analysis
        if has_lump:
            if lump_firmness == "Hard / rigid / irregular edge":
                risk_score += 35
                reasons.append("Hard mass with irregular/rigid boundaries")
            elif lump_firmness == "Rubbery / smooth":
                risk_score += 12
            else:
                risk_score += 5

            if lump_mobility == "Firmly attached / fixed in place":
                risk_score += 25
                reasons.append("Mass is fixed to underlying tissue")

            if lump_duration == "More than 6 weeks / Growing":
                risk_score += 15
                reasons.append("Persistent or growing mass (> 6 weeks)")

        # Skin changes
        if has_skin_changes:
            if "peau d'orange" in skin_detail or "ulceration" in skin_detail:
                risk_score += 30
                reasons.append("Skin dimpling, retraction, or peau d'orange texture")
            else:
                risk_score += 10

        # Nipple changes
        if has_nipple_changes:
            if "bloody" in nipple_detail or "inverted" in nipple_detail:
                risk_score += 25
                reasons.append("Unilateral bloody discharge or recent nipple inversion")
            else:
                risk_score += 8

        # Family History
        if "Multiple" in family_history or "BRCA" in family_history:
            risk_score += 25
            reasons.append("Strong family history (multiple relatives or BRCA mutation)")
        elif "before age 50" in family_history:
            risk_score += 15
            reasons.append("First-degree relative diagnosed pre-menopause")

        # Prior biopsy
        if "Atypical" in prev_biopsy:
            risk_score += 20
            reasons.append("History of atypical hyperplasia")

        # Clamp between 2% and 98%
        est_probability = min(max(risk_score, 3), 96)

        st.markdown("---")
        st.subheader("📋 Clinical Symptom Analysis Report")

        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            if est_probability >= 50:
                st.error("### ⚠️ Assessment: High Suspicion of Malignancy / High Risk")
                st.write(
                    "The combination of physical signs and follow-up responses presents features commonly associated "
                    "with malignant breast disease. **Immediate clinical evaluation (diagnostic mammogram + ultrasound ± core needle biopsy) is strongly recommended.**"
                )
            elif est_probability >= 25:
                st.warning("### ⚠️ Assessment: Moderate Risk / Undetermined Presentation")
                st.write(
                    "Some presenting symptoms warrant standard clinical breast examination (CBE) and routine imaging. "
                    "Many lumps with these features are benign (e.g. fibroadenomas or fluid cysts), but imaging is necessary for confirmation."
                )
            else:
                st.success("### ✅ Assessment: Low Suspicion / Benign Presentation")
                st.write(
                    "Reported signs do not show typical high-risk malignancy flags. Continue routine self-checks and periodic clinical screenings according to age guidelines."
                )

        with col_r2:
            st.metric("Estimated Symptom Risk Index", f"{est_probability}%")
            st.progress(est_probability / 100.0)

        if reasons:
            st.markdown("#### Primary Factors Contributing to Score:")
            for r in reasons:
                st.markdown(f"- **{r}**")


# =========================================================================
# TAB 2: FNA BIOPSY CLASSIFIER (Wisconsin ML Model)
# =========================================================================
with tab_fna:
    st.subheader("Wisconsin Diagnostic Biopsy Predictor")
    st.caption("Predicts Benign vs. Malignant from FNA digital cellular measurements.")

    mode = st.radio("Features to input:", ["quick", "full"], format_func=lambda x: "Quick (10 Mean Features)" if x == "quick" else "Full (30 Nuclear Measurements)", horizontal=True)

    artifact = load_artifact(mode)
    feature_names = artifact["feature_names"]

    c_s1, c_s2, c_clear = st.columns([1, 1, 2])
    sample_choice = None
    if c_s1.button("🧪 Fill Malignant Case"):
        sample_choice = "malignant"
    if c_s2.button("🧪 Fill Benign Case"):
        sample_choice = "benign"

    sample_dict = {}
    if sample_choice:
        dataset = load_breast_cancer(as_frame=True)
        target = 0 if sample_choice == "malignant" else 1
        row = dataset.data[dataset.target == target].iloc[0]
        if mode == "quick":
            row = row[[c for c in row.index if c.startswith("mean ")]]
        sample_dict = row.to_dict()

    with st.form("fna_predict_form"):
        grid = st.columns(3)
        input_data = {}
        for idx, feat in enumerate(feature_names):
            with grid[idx % 3]:
                default_val = float(sample_dict.get(feat, 0.0))
                input_data[feat] = st.number_input(
                    feat.replace("_", " ").title(),
                    value=default_val,
                    format="%.4f",
                    key=f"fna_{mode}_{feat}"
                )

        predict_btn = st.form_submit_button("Run Biopsy Prediction", type="primary", use_container_width=True)

    if predict_btn:
        try:
            res = predict_one(input_data, mode)
            is_mal = res["prediction"] == "malignant"
            p_mal = res["probability_malignant"]
            p_ben = res["probability_benign"]

            st.write("---")
            if is_mal:
                st.error("### Result: MALIGNANT (High Confidence of Cancerous Cells)")
            else:
                st.success("### Result: BENIGN (Non-cancerous Cellular Profile)")

            rc1, rc2 = st.columns(2)
            rc1.metric("Malignant Probability", f"{p_mal * 100:.2f}%")
            rc2.metric("Benign Probability", f"{p_ben * 100:.2f}%")
            st.progress(p_mal)
        except Exception as e:
            st.error(f"Prediction Error: {e}")


# =========================================================================
# TAB 3: MODEL PERFORMANCE & METRICS
# =========================================================================
with tab_metrics:
    st.subheader("Trained Model Performance")
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            m = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{m.get('accuracy', 0)*100:.2f}%")
        c2.metric("Malignant Sensitivity (Recall)", f"{m.get('malignant_recall', 0)*100:.2f}%")
        c3.metric("Malignant Precision", f"{m.get('malignant_precision', 0)*100:.2f}%")
        c4.metric("ROC AUC Score", f"{m.get('roc_auc_malignant', 0):.4f}")

        cm_file = MODEL_DIR / "confusion_matrix.png"
        if cm_file.exists():
            st.image(str(cm_file), caption="Model Confusion Matrix", width=500)
    else:
        st.info("Metrics not found.")


# =========================================================================
# TAB 4: CLINICAL GUIDANCE
# =========================================================================
with tab_info:
    st.markdown("""
    ### Clinical Screening Guidelines
    
    1. **Early Detection**:
       - Mammograms remain the gold standard for detecting tumours before physical lumps can be felt.
       - Annual or biennial mammograms are recommended for individuals aged 40–74 depending on clinical risk factors.
    
    2. **When to Seek Immediate Care**:
       - A new, painless, firm lump that does not fluctuate with your menstrual cycle.
       - Any unilateral blood-stained nipple discharge.
       - Localized skin dimpling, orange-peel changes, or inverted nipple.
    
    3. **About this Software**:
       - This tool is built using machine learning trained on the Wisconsin Breast Cancer Diagnostic dataset (Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian).
    """)
