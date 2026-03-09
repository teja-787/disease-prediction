
# Heart Disease Prediction API

A production-grade ML API that predicts heart disease using Random Forest classifier.

## Tech Stack
- Python 3.11
- FastAPI
- Scikit-learn
- Docker

## Features
- REST API with auto-generated Swagger docs
- Web UI for predictions
- Dockerized for production deployment

## Run Locally
docker build -t disease-prediction .
docker run -p 8000:8000 disease-prediction

## API Docs
Visit http://localhost:8000/docs
