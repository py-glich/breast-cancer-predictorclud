import React from "react";

function Metric({ label, value }) {
  const display = typeof value === "number" ? `${(value * 100).toFixed(2)}%` : value;
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{display ?? "—"}</strong>
    </div>
  );
}

export default function ModelMetrics({ metrics }) {
  if (!metrics) {
    return (
      <section className="panel">
        <p className="eyebrow">MODEL PERFORMANCE</p>
        <h2>Evaluation metrics</h2>
        <p className="muted">Run model evaluation to display metrics.</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <p className="eyebrow">MODEL PERFORMANCE</p>
      <h2>Held-out test set</h2>
      <div className="metric-grid">
        <Metric label="Accuracy" value={metrics.accuracy} />
        <Metric label="Malignant recall" value={metrics.malignant_recall} />
        <Metric label="Malignant precision" value={metrics.malignant_precision} />
        <Metric label="Malignant F1" value={metrics.malignant_f1} />
        <Metric label="ROC-AUC" value={metrics.roc_auc_malignant} />
        <Metric label="Test samples" value={metrics.test_samples} />
      </div>
      <p className="muted small">
        Metrics are from the held-out test set and describe this educational dataset only.
      </p>
    </section>
  );
}
