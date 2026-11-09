"""FastAPI application for the breast cancer educational ML demo."""

from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.datasets import load_breast_cancer

from backend.schemas import PredictionRequest
from ml.predict import load_artifact, predict_one


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "ml" / "saved_models" / "metrics.json"

app = FastAPI(
    title="Breast Cancer Classification API",
    description=(
        "Educational API using the Wisconsin Breast Cancer Diagnostic dataset. "
        "Not intended for medical diagnosis."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Breast Cancer Classification API",
        "docs": "/docs",
        "notice": "Educational demonstration only; not a medical diagnostic tool.",
    }


@app.get("/health")
def health():
    try:
        load_artifact()
        model_ready = True
    except FileNotFoundError:
        model_ready = False

    return {"status": "ok", "model_ready": model_ready}


@app.get("/features")
def get_features(mode: str = "full"):
    try:
        artifact = load_artifact(mode)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))

    return {
        "feature_names": artifact["feature_names"],
        "feature_count": len(artifact["feature_names"]),
    }


@app.get("/sample")
def get_demo_sample(mode: str = "full", kind: str = "malignant"):
    """Return one real dataset row (malignant or benign) to make the demo testable."""
    dataset = load_breast_cancer(as_frame=True)
    wanted = 0 if kind == "malignant" else 1
    row = dataset.data[dataset.target == wanted].iloc[0]
    if mode == "quick":
        row = row[[c for c in row.index if c.startswith("mean ")]]
    sample = row.to_dict()
    return {
        "features": {name: float(value) for name, value in sample.items()},
        "notice": "This is a sample from the public educational dataset.",
    }


@app.get("/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run python ml/evaluate_model.py first.",
        )

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.post("/predict")
def make_prediction(request: PredictionRequest):
    try:
        return predict_one(request.features, request.mode)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
