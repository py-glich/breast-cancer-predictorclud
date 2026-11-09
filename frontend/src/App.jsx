import React, { useEffect, useState } from "react";
import PredictionForm from "./components/PredictionForm";
import PredictionResult from "./components/PredictionResult";
import ModelMetrics from "./components/ModelMetrics";
import ImageNotice from "./components/ImageNotice";
import { readFeatureFile } from "./importFile";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export default function App() {
  const [featureNames, setFeatureNames] = useState([]);
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("Connecting");
  const [mode, setMode] = useState("quick");

  useEffect(() => {
    async function loadInitialData() {
      try {
        const [featureResponse, metricsResponse, healthResponse] = await Promise.all([
          fetch(`${API_BASE}/features?mode=${mode}`),
          fetch(`${API_BASE}/metrics`),
          fetch(`${API_BASE}/health`)
        ]);

        if (!featureResponse.ok) {
          throw new Error("Could not load model features. Train the model and start the API.");
        }

        const featureData = await featureResponse.json();
        setFeatureNames(featureData.feature_names);
        setValues(Object.fromEntries(featureData.feature_names.map((name) => [name, ""])));
        setResult(null);
        setError("");

        if (metricsResponse.ok) {
          setMetrics(await metricsResponse.json());
        }

        if (healthResponse.ok) {
          const health = await healthResponse.json();
          setApiStatus(health.model_ready ? "API and model online" : "API online; model missing");
        }
      } catch (err) {
        setApiStatus("API unavailable");
        setError(err.message);
      }
    }

    loadInitialData();
  }, [mode]);

  function updateValue(name, value) {
    setValues((previous) => ({ ...previous, [name]: value }));
  }

  async function loadSample(kind = "malignant") {
    setError("");
    try {
      const response = await fetch(`${API_BASE}/sample?mode=${mode}&kind=${kind}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not load sample.");
      setValues(
        Object.fromEntries(
          featureNames.map((name) => [name, String(data.features[name])])
        )
      );
      setResult(null);
    } catch (err) {
      setError(err.message);
    }
  }

  async function importFile(file) {
    setError("");
    try {
      setValues(await readFeatureFile(file, featureNames));
      setResult(null);
    } catch (err) {
      setError(err.message);
    }
  }

  function clearForm() {
    setValues(Object.fromEntries(featureNames.map((name) => [name, ""])));
    setResult(null);
    setError("");
  }

  async function submitPrediction(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const numericFeatures = {};
      for (const name of featureNames) {
        const raw = values[name];
        if (raw === "" || raw === undefined) {
          throw new Error(`Please enter a value for ${name}.`);
        }
        const number = Number(raw);
        if (!Number.isFinite(number)) {
          throw new Error(`${name} must be a valid finite number.`);
        }
        numericFeatures[name] = number;
      }

      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: numericFeatures, mode })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction request failed.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">BC</span>
          <span>Breast Cancer <strong>ML Lab</strong></span>
        </a>
        <span className="status-pill">
          <span className="status-dot" />
          {apiStatus}
        </span>
      </header>

      <section className="hero">
        <p className="eyebrow">MACHINE LEARNING · CLASSIFICATION</p>
        <h1>Breast tumour classification</h1>
        <p>
          An educational machine learning demonstration using the Wisconsin
          Breast Cancer Diagnostic dataset. Enter measurements, or import them from a
          CSV / JSON file, to see the model's prediction.
        </p>
      </section>

      <div className="notice-banner">
        <strong>Educational use only.</strong> This application is not a medical
        device, does not provide a diagnosis, and must not be used to make
        screening or treatment decisions. Inputs must come from appropriate
        FNA-derived measurements, not symptoms or photographs.
      </div>

      <div className="content-grid">
        <PredictionForm
          mode={mode}
          onModeChange={setMode}
          onImport={importFile}
          onClear={clearForm}
          featureNames={featureNames}
          values={values}
          onChange={updateValue}
          onSubmit={submitPrediction}
          onLoadSample={loadSample}
          loading={loading}
          error={error}
        />
        <div className="right-column">
          <PredictionResult result={result} />
          <ModelMetrics metrics={mode === "quick" && metrics?.quick ? metrics.quick : metrics} />
          <ImageNotice />
        </div>
      </div>

      <footer>
        Built with React, FastAPI and Scikit-learn · Public educational dataset
      </footer>
    </main>
  );
}
