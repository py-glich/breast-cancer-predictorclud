"""Parse an uploaded CSV / JSON measurement record into {feature_name: float}."""

import csv
import io
import json
import re


def normalize(raw: str) -> str:
    """Make 'radius_mean', 'Mean Radius' and 'mean radius' all compare equal."""
    n = re.sub(r"[_\s]+", " ", str(raw).strip().lower())
    if n.endswith(" mean"):
        n = "mean " + n[: -len(" mean")]
    elif n.endswith(" se"):
        n = n[: -len(" se")] + " error"
    elif n.endswith(" worst"):
        n = "worst " + n[: -len(" worst")]
    return n


def parse_feature_file(filename: str, content: bytes, feature_names: list) -> dict:
    name = filename.lower()
    if not (name.endswith(".csv") or name.endswith(".json")):
        raise ValueError(
            "Only .csv or .json files with measurements are supported. "
            "Photos cannot be classified by this model."
        )

    text = content.decode("utf-8-sig")
    if name.endswith(".json"):
        data = json.loads(text)
        if isinstance(data, list):
            data = data[0]
        record = data.get("features", data)
    else:
        rows = list(csv.reader(io.StringIO(text)))
        rows = [r for r in rows if any(cell.strip() for cell in r)]
        if len(rows) < 2:
            raise ValueError("CSV needs a header row and one data row.")
        record = dict(zip([h.strip() for h in rows[0]], [v.strip() for v in rows[1]]))

    lookup = {normalize(k): v for k, v in record.items()}
    values, missing = {}, []
    for feature in feature_names:
        raw = lookup.get(normalize(feature))
        try:
            values[feature] = float(raw)
        except (TypeError, ValueError):
            missing.append(feature)

    if missing:
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise ValueError(
            f"File is missing or has invalid values for: {', '.join(missing[:5])}{more}."
        )
    return values
