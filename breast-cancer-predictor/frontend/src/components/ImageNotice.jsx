import React from "react";

export default function ImageNotice() {
  return (
    <section className="panel image-panel">
      <p className="eyebrow">ABOUT PHOTOS</p>
      <h2>Why can't I upload a picture?</h2>
      <p className="muted small-text">
        This model was trained on numeric measurements taken from fine-needle
        aspirate (FNA) cell images, not on pictures. Uploading a photo of a
        person, a scan or a report image would give a meaningless result, so
        the app does not pretend to do it. Use the measurement values from a
        lab report instead, or import them as a CSV / JSON file.
      </p>
      <p className="muted small-text">
        If you notice a lump or any change, please see a doctor. Only a
        clinician can diagnose.
      </p>
    </section>
  );
}
