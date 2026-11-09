"""Request and response schemas for the prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Named numeric tumour measurement features (30 for full mode, 10 for quick mode).",
    )
    mode: Literal["full", "quick"] = "full"
