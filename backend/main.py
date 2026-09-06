import os
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from database import SessionLocal, engine
from models import Base, NewsPrediction
import schemas


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# DATABASE
# ============================================================

# Create database tables
Base.metadata.create_all(bind=engine)


# ============================================================
# CONFIGURATION
# ============================================================

ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "distilbert").lower()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "Dhanashri-Adawade/fake-news-distilbert"
)

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))

# Number of CPU threads used by PyTorch.
# Useful when deploying on a small CPU server.
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "1"))

torch.set_num_threads(TORCH_THREADS)


# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

tokenizer = None
distilbert_model = None

old_model = None
old_vectorizer = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load only the model selected by ACTIVE_MODEL.
    """

    global tokenizer
    global distilbert_model
    global old_model
    global old_vectorizer

    logger.info("Active model: %s", ACTIVE_MODEL)

    # --------------------------------------------------------
    # DistilBERT
    # --------------------------------------------------------

    if ACTIVE_MODEL == "distilbert":

        logger.info("Loading DistilBERT model: %s", MODEL_PATH)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=True
        )

        distilbert_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        distilbert_model.eval()

        logger.info("DistilBERT model loaded successfully.")

        # Print model labels if available
        if hasattr(distilbert_model.config, "id2label"):
            logger.info(
                "Model labels: %s",
                distilbert_model.config.id2label
            )

    # --------------------------------------------------------
    # Old TF-IDF + Logistic Regression model
    # --------------------------------------------------------

    elif ACTIVE_MODEL == "old":

        logger.info("Loading old TF-IDF model...")

        import joblib

        old_model = joblib.load("fake_news_model.pkl")
        old_vectorizer = joblib.load("tfidf_vectorizer.pkl")

        logger.info("Old model loaded successfully.")

    else:
        raise ValueError(
            f"Invalid ACTIVE_MODEL='{ACTIVE_MODEL}'. "
            "Use 'distilbert' or 'old'."
        )


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting Fake News Detection API...")

    load_model()

    logger.info("Application startup completed.")

    yield

    logger.info("Application shutting down...")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Fake News Detection API",
    description="Fake news detection using DistilBERT or TF-IDF + Logistic Regression.",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173"
).split(",")

cors_origins = [
    origin.strip()
    for origin in cors_origins
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE DEPENDENCY
# ============================================================

def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class NewsRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=10000,
        description="News article or headline text"
    )


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Fake News Detection API is running",
        "model": ACTIVE_MODEL
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    model_loaded = (
        distilbert_model is not None
        if ACTIVE_MODEL == "distilbert"
        else old_model is not None
    )

    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model": ACTIVE_MODEL
    }


# ============================================================
# DISTILBERT PREDICTION
# ============================================================

def predict_with_distilbert(text: str):

    if tokenizer is None or distilbert_model is None:
        raise RuntimeError("DistilBERT model is not loaded.")

    # Tokenization
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

    # Inference mode is slightly more efficient than no_grad()
    with torch.inference_mode():

        outputs = distilbert_model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    confidence_tensor, predicted_class = torch.max(
        probabilities,
        dim=0
    )

    predicted_class = predicted_class.item()
    confidence = confidence_tensor.item() * 100

    # --------------------------------------------------------
    # Label mapping
    #
    # Your current model expects:
    #
    # 0 = Fake
    # 1 = Real
    #
    # --------------------------------------------------------

    result = "Real" if predicted_class == 1 else "Fake"

    return result, confidence


# ============================================================
# OLD MODEL PREDICTION
# ============================================================

def predict_with_old_model(text: str):

    if old_model is None or old_vectorizer is None:
        raise RuntimeError("Old model is not loaded.")

    text_tfidf = old_vectorizer.transform([text])

    prediction = old_model.predict(text_tfidf)[0]

    probabilities = old_model.predict_proba(text_tfidf)[0]

    confidence = float(max(probabilities)) * 100

    result = "Real" if prediction == 1 else "Fake"

    return result, confidence


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    news: NewsRequest,
    db: Session = Depends(get_db)
):

    text = news.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="News text cannot be empty."
        )

    try:

        # ----------------------------------------------------
        # Select active model
        # ----------------------------------------------------

        if ACTIVE_MODEL == "distilbert":

            result, confidence = predict_with_distilbert(text)

        elif ACTIVE_MODEL == "old":

            result, confidence = predict_with_old_model(text)

        else:

            raise HTTPException(
                status_code=500,
                detail="Invalid model configuration."
            )

        # ----------------------------------------------------
        # Save prediction to database
        # ----------------------------------------------------

        new_prediction = NewsPrediction(
            news_text=text,
            prediction=result,
            confidence=round(confidence, 2)
        )

        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "prediction": result,
            "confidence": round(confidence, 2)
        }

    except HTTPException:
        raise

    except Exception as e:

        db.rollback()

        logger.exception(
            "Prediction failed: %s",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again."
        )


# ============================================================
# HISTORY ENDPOINT
# ============================================================

@app.get(
    "/history",
    response_model=list[schemas.PredictionHistory]
)
def get_history(
    db: Session = Depends(get_db)
):

    try:

        history = (
            db.query(NewsPrediction)
            .order_by(NewsPrediction.id.desc())
            .all()
        )

        return history

    except Exception as e:

        logger.exception(
            "Failed to retrieve history: %s",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve prediction history."
        )


# ============================================================
# DELETE HISTORY
# ============================================================

@app.delete("/history")
def delete_history(
    db: Session = Depends(get_db)
):

    try:

        deleted_count = (
            db.query(NewsPrediction)
            .delete()
        )

        db.commit()

        return {
            "message": "Prediction history deleted successfully.",
            "deleted_count": deleted_count
        }

    except Exception as e:

        db.rollback()

        logger.exception(
            "Failed to delete history: %s",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to delete prediction history."
        )
