import React from "react";

export default function PredictionResult({ result }) {
  if (!result) {
    return (
      <section className="panel result-panel">
        <p className="eyebrow">MODEL OUTPUT</p>
        <h2>Prediction result</h2>
        <p className="muted">
          Your model result will appear here after you submit all 30 features.
        </p>
      </section>
    );
  }

  const malignant = result.prediction === "malignant";

  return (
    <section className="panel result-panel">
      <p className="eyebrow">MODEL OUTPUT</p>
      <h2>Predicted class</h2>
      <div className={`result-label ${malignant ? "malignant" : "benign"}`}>
        {result.prediction.toUpperCase()}
      </div>

      <div className="probability-list">
        <div>
          <div className="probability-heading">
            <span>Malignant model probability</span>
            <strong>{(result.probabilities.malignant * 100).toFixed(2)}%</strong>
          </div>
          <div className="bar-track">
            <div
              className="bar-fill malignant-fill"
              style={{ width: `${result.probabilities.malignant * 100}%` }}
            />
          </div>
        </div>
        <div>
          <div className="probability-heading">
            <span>Benign model probability</span>
            <strong>{(result.probabilities.benign * 100).toFixed(2)}%</strong>
          </div>
          <div className="bar-track">
            <div
              className="bar-fill benign-fill"
              style={{ width: `${result.probabilities.benign * 100}%` }}
            />
          </div>
        </div>
      </div>

      <p className="disclaimer">{result.notice}</p>
    </section>
  );
}
