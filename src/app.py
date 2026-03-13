# py -m uvicorn app:app --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict_match

app = FastAPI()

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
    result = predict_match(req.age, req.intentions, req.title, req.description)
    return result
