import React, { useRef } from "react";

export default function PredictionForm({
  mode,
  onModeChange,
  featureNames,
  values,
  onChange,
  onSubmit,
  onLoadSample,
  onImport,
  onClear,
  loading,
  error
}) {
  const fileRef = useRef(null);

  return (
    <section className="panel">
      <p className="eyebrow">MODEL INPUT</p>
      <h2>Tumour measurements</h2>

      <div className="mode-toggle" role="tablist" aria-label="Input mode">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "quick"}
          className={mode === "quick" ? "active" : ""}
          onClick={() => onModeChange("quick")}
        >
          Quick · 10 values
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "full"}
          className={mode === "full" ? "active" : ""}
          onClick={() => onModeChange("full")}
        >
          Full · 30 values
        </button>
      </div>

      <p className="muted small-text">
        {mode === "quick"
          ? "Enter the 10 average (mean) measurements from an FNA report."
          : "Enter all 30 measurements (mean, error and worst values) from an FNA report."}
      </p>

      <div className="toolbar">
        <button className="secondary-button" type="button" onClick={() => onLoadSample("malignant")}>
          Demo: malignant sample
        </button>
        <button className="secondary-button" type="button" onClick={() => onLoadSample("benign")}>
          Demo: benign sample
        </button>
        <button className="secondary-button" type="button" onClick={() => fileRef.current?.click()}>
          Import CSV / JSON
        </button>
        <button className="ghost-button" type="button" onClick={onClear}>
          Clear
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.json,application/json,text/csv"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onImport(file);
            event.target.value = "";
          }}
        />
      </div>

      {featureNames.length === 0 ? (
        <p className="muted">Connecting to the prediction API…</p>
      ) : (
        <form onSubmit={onSubmit}>
          <div className="field-grid">
            {featureNames.map((name) => (
              <label className="field" key={name}>
                <span>{name}</span>
                <input
                  type="number"
                  inputMode="decimal"
                  step="any"
                  required
                  value={values[name] ?? ""}
                  onChange={(event) => onChange(name, event.target.value)}
                  placeholder="0.00"
                />
              </label>
            ))}
          </div>

          {error && <p className="error-message" role="alert">{error}</p>}

          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? "Running model…" : "Run classification"}
          </button>
        </form>
      )}
    </section>
  );
}
