// Parse a CSV or JSON file into {featureName: number}, matching the model's names.
// Accepts sklearn-style names ("mean radius") and Kaggle-style names ("radius_mean").

function normalize(raw) {
  let n = String(raw).trim().toLowerCase().replace(/[_\s]+/g, " ");
  if (n.endsWith(" mean")) n = "mean " + n.slice(0, -5);
  else if (n.endsWith(" se")) n = n.slice(0, -3) + " error";
  else if (n.endsWith(" worst")) n = "worst " + n.slice(0, -6);
  return n;
}

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let quoted = false;
  for (const ch of line) {
    if (ch === '"') quoted = !quoted;
    else if (ch === "," && !quoted) { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out.map((v) => v.trim());
}

export async function readFeatureFile(file, featureNames) {
  const name = file.name.toLowerCase();
  if (!name.endsWith(".csv") && !name.endsWith(".json")) {
    throw new Error(
      "Only .csv or .json files with measurements are supported. Photos cannot be classified by this model."
    );
  }
  const text = await file.text();
  let record = {};

  if (name.endsWith(".json")) {
    const data = JSON.parse(text);
    record = data.features ?? (Array.isArray(data) ? data[0] : data);
  } else {
    const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
    if (lines.length < 2) throw new Error("CSV needs a header row and one data row.");
    const headers = parseCsvLine(lines[0]);
    const row = parseCsvLine(lines[1]);
    headers.forEach((h, i) => { record[h] = row[i]; });
  }

  const lookup = {};
  for (const [key, value] of Object.entries(record)) lookup[normalize(key)] = value;

  const values = {};
  const missing = [];
  for (const feature of featureNames) {
    const v = lookup[normalize(feature)];
    if (v === undefined || v === "" || !Number.isFinite(Number(v))) missing.push(feature);
    else values[feature] = String(v);
  }
  if (missing.length) {
    throw new Error(
      `File is missing or has invalid values for: ${missing.slice(0, 5).join(", ")}${
        missing.length > 5 ? ` and ${missing.length - 5} more` : ""
      }.`
    );
  }
  return values;
}
