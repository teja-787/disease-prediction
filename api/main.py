from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from api.predict import predict_disease
from pathlib import Path

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease using ML",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/ui", StaticFiles(directory=BASE_DIR / "ui"), name="ui")

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def root():
    return FileResponse(BASE_DIR / "ui/index.html")

@app.get("/health")
def health():
    return {"status": "healthy", "model": "RandomForest", "version": "1.0.0"}

@app.post("/predict")
def predict(data: PatientData):
    result = predict_disease(data.dict())
    return result