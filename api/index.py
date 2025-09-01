from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import json
from rapidfuzz import fuzz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the model
model = None


class TextInput(BaseModel):
    text: str = Field(
        ..., description="Text to analyze for drug slang", max_length=10000
    )


class BatchTextInput(BaseModel):
    texts: List[str] = Field(
        ..., description="List of texts to analyze", max_length=100
    )


class PredictionResponse(BaseModel):
    text: str
    predictions: Dict[str, Any]
    confidence: Optional[float] = None
    processing_time: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_processed: int
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


# Model loading and management
async def load_model():
    """
    Load your NLP model here. Replace this with your actual model loading logic.
    """
    global model
    try:
        logger.info("Loading NLP model...")

        # Use Path relative to the project root for Vercel
        MODEL_PATH = Path("models/drug_slang_model.pkl")
        SLANG_JSON = Path("models/slang_dict.json")

        # Check if files exist (Vercel deploys to a read-only filesystem)
        if not MODEL_PATH.exists() or not SLANG_JSON.exists():
            raise FileNotFoundError(
                "Model or slang dict not found. Ensure models/ is in the root and files are uploaded."
            )

        # Load the model pipeline and slang dictionary
        pipeline = joblib.load(MODEL_PATH)
        slang_dict = json.loads(SLANG_JSON.read_text(encoding="utf-8"))

        # Create a wrapper class for your prediction logic
        class DrugSlangModel:
            def __init__(self, pipeline, slang_dict):
                self.pipeline = pipeline
                self.slang_dict = slang_dict

            def normalize_text(
                self, text: str, threshold: int = 70
            ) -> tuple[str, list]:
                """Normalize text using slang dictionary with fuzzy matching"""
                text_lower = str(text).lower()
                words = text_lower.split()
                out = []
                replaced_terms = []

                for w in words:
                    replaced = False
                    for slang, meaning in self.slang_dict.items():
                        if slang == w or fuzz.ratio(slang, w) >= threshold:
                            out.append(meaning)
                            replaced_terms.append((slang, meaning))
                            replaced = True
                            break
                    if not replaced:
                        out.append(w)

                return " ".join(out), replaced_terms

            def predict(self, text: str) -> Dict[str, Any]:
                """Make prediction using your trained model"""
                normalized_text, replaced_terms = self.normalize_text(text)
                prediction = self.pipeline.predict([normalized_text])[0]

                try:
                    probabilities = self.pipeline.predict_proba([normalized_text])[0]
                    confidence = float(max(probabilities))
                except AttributeError:
                    confidence = 0.85 if prediction == 1 else 0.15

                is_drug_related = bool(prediction == 1)

                return {
                    "original_text": text,
                    "normalized_text": normalized_text,
                    "contains_drug_slang": is_drug_related,
                    "confidence": confidence,
                    "detected_terms": [
                        {"slang": slang, "meaning": meaning}
                        for slang, meaning in replaced_terms
                    ],
                    "prediction_label": (
                        "⚠️ Drug-related" if is_drug_related else "✅ Clean"
                    ),
                }

        model = DrugSlangModel(pipeline, slang_dict)
        logger.info("Model loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e


async def predict_text(text: str) -> Dict[str, Any]:
    """
    Make predictions using your loaded model.
    """
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prediction = model.predict(text)
        return prediction

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Lifespan management for Vercel
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup (model loading on cold start)
    logger.info("Starting up FastAPI application...")
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")


# Create FastAPI app
app = FastAPI(
    title="Drug Slang NLP API",
    description="API for detecting and analyzing drug slang in text",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware (restrict for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend.vercel.app",
        "http://localhost:3000",
    ],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Drug Slang NLP API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single_text(input_data: TextInput):
    """
    Predict drug slang in a single text.
    """
    start_time = asyncio.get_event_loop().time()

    try:
        predictions = await predict_text(input_data.text)
        processing_time = asyncio.get_event_loop().time() - start_time

        return PredictionResponse(
            text=input_data.text,
            predictions=predictions,
            confidence=predictions.get("confidence"),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_texts(input_data: BatchTextInput):
    """
    Predict drug slang in multiple texts (batch processing).
    """
    start_time = asyncio.get_event_loop().time()

    try:
        results = []
        for text in input_data.texts:
            text_start_time = asyncio.get_event_loop().time()
            predictions = await predict_text(text)
            text_processing_time = asyncio.get_event_loop().time() - text_start_time

            results.append(
                PredictionResponse(
                    text=text,
                    predictions=predictions,
                    confidence=predictions.get("confidence"),
                    processing_time=text_processing_time,
                    timestamp=datetime.now().isoformat(),
                )
            )

        total_processing_time = asyncio.get_event_loop().time() - start_time

        return BatchPredictionResponse(
            results=results,
            total_processed=len(results),
            processing_time=total_processing_time,
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": "Drug Slang Classifier",
        "status": "loaded",
        "capabilities": ["text_classification", "drug_slang_detection"],
        "input_format": "text",
        "output_format": "classification_results",
    }
