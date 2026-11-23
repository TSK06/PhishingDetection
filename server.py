import os
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from prediction import predictor
from validation import validate_url


class URLRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def check_url(cls, value: str) -> str:
        return validate_url(value)


class PredictionResponse(BaseModel):
    label: Literal["good", "bad"]


app = FastAPI(title="Phishing URL Detection API", version="1.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse, include_in_schema=False)
def predict_url(request: URLRequest) -> PredictionResponse:
    label = "good" if predictor(request.url) == 0 else "bad"
    return PredictionResponse(label=label)
