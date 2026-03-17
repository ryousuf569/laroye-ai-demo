# py -m uvicorn app:app --reload

import json
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from predict import predict_match

app = FastAPI()

# load once at startup — avoids 4–8s reload on every request
with open(r"..\artifacts\model.pkl", "rb") as f:
    _model = pickle.load(f)
with open(r"..\artifacts\metadata.json", "r") as f:
    _metadata = json.load(f)
_embedder = SentenceTransformer(_metadata["embed_model"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    age: int
    intentions: str
    title: str
    description: str


# simple health-check route
@app.get("/")
def root():
    return {"message": "API is running"}


# prediction endpoint
@app.post("/predict")
def predict(req: PredictRequest):
    result = predict_match(req.age, req.intentions, req.title, req.description,
                           saved_model=_model, embedder=_embedder)
    return result
