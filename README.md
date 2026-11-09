# Breast Cancer Classification — Educational ML Web App

A portfolio project using the Wisconsin Breast Cancer Diagnostic dataset, Scikit-learn, FastAPI and React.

## Important limitation

This is an educational machine-learning demonstration, not a clinically validated medical device. It must not be used to diagnose, screen, or make treatment decisions. The model expects the 30 numeric features derived from fine-needle aspirate (FNA) images; it does not accept symptoms or ordinary photographs.

## Project structure

- `ml/`: dataset exploration, preprocessing utilities, model training, evaluation and prediction.
- `backend/`: FastAPI endpoints.
- `frontend/`: React web interface.

## 1. Set up Python

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Explore the dataset

```powershell
python ml/explore_data.py
```

## 3. Train the model

```powershell
python ml/train_model.py
```

This creates `ml/saved_models/breast_cancer_pipeline.joblib`.

## 4. Evaluate the model

```powershell
python ml/evaluate_model.py
```

This creates `metrics.json` and `confusion_matrix.png` in `ml/saved_models/`.

## 5. Test prediction logic from the terminal

```powershell
python ml/predict.py
```

## 6. Start FastAPI

From the project root, with the virtual environment active:

```powershell
uvicorn backend.main:app --reload
```

- API: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs

## 7. Start React

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally http://127.0.0.1:5173.

## API endpoints

- `GET /health`: API and model readiness.
- `GET /features`: ordered feature names required by the model.
- `GET /sample`: one sample row from the public dataset for demonstration.
- `GET /metrics`: held-out test metrics.
- `POST /predict`: predict from a JSON object containing all 30 named features.

## Target encoding

The built-in Scikit-learn dataset uses:
- `0` = malignant
- `1` = benign

The UI displays the human-readable class names.

## Evaluation note

The project uses a stratified 80/20 train/test split with a fixed random seed. Scaling is fitted inside a Scikit-learn Pipeline on training data only. Malignant recall is reported explicitly because false negatives are an important error type to examine in this educational classification exercise.

Do not interpret model probabilities as clinical certainty or calibrated medical risk.

## Input modes (new)

- **Quick (10 values):** the 10 mean measurements only, using a second model trained on those columns.
- **Full (30 values):** all mean, error and worst measurements.
- **Import CSV / JSON:** fills the form from a file. Both `mean radius` and `radius_mean` (Kaggle) column names work.
- **Photos are not supported.** The model learns from numeric FNA measurements, not pictures. A real image classifier would need a labelled image dataset (for example histopathology images) and a separate model.
