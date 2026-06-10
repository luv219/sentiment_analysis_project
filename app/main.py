from pathlib import Path
import re, string, logging
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_api")

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run the notebook first.")

model = joblib.load(MODEL_PATH)
logger.info("Model loaded from %s", MODEL_PATH)

app = FastAPI(title="Sentiment Analysis API", version="2.0.0", description="Classifies text reviews as positive or negative.")

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_RE_HTML    = re.compile(r"<[^>]+>")
_RE_URL     = re.compile(r"https?://\\S+|www\\.\\S+")
_RE_NONALPH = re.compile(r"[^a-zA-Z\\s]")
_RE_SPACES  = re.compile(r"\\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    text = _RE_NONALPH.sub(" ", text)
    text = _RE_SPACES.sub(" ", text).strip()
    return text


class ReviewRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class PredictionResponse(BaseModel):
    text: str
    clean_text: str
    prediction: str
    confidence: Optional[float]
    pos_proba: Optional[float]
    neg_proba: Optional[float]


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "model": str(MODEL_PATH.name)}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: ReviewRequest):
    clean = clean_text(payload.text)
    if not clean:
        raise HTTPException(status_code=422, detail="Review is empty after cleaning.")

    pred   = int(model.predict([clean])[0])
    label  = "positive" if pred == 1 else "negative"
    proba  = model.predict_proba([clean])[0]
    conf   = float(proba[pred])

    return PredictionResponse(
        text=payload.text,
        clean_text=clean,
        prediction=label,
        confidence=round(conf, 4),
        pos_proba=round(float(proba[1]), 4),
        neg_proba=round(float(proba[0]), 4),
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Sentiment Analysis API is running — open /docs</h1>"