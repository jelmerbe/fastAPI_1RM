from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# Load the trained models from the pickle file
with open("data/xgb_1rm_models.pkl", "rb") as f:
    models = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define the input schema
class PredictionRequest(BaseModel):
    weight: float
    gender: int  # Assume 0 for Female, 1 for Male
    known_exercises: dict  # Dictionary of known exercise weights

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Predict 1RM values for all exercises based on user input.
    """
    # Parse input data
    weight = request.weight
    gender = request.gender
    known_exercises = request.known_exercises

    # Prepare input features
    input_features = pd.DataFrame([known_exercises])
    input_features["weight"] = weight
    input_features["gender"] = gender
    input_features.fillna(-1, inplace=True)  # Replace missing values with -1

    # Perform predictions for all exercises
    predictions = {}
    for exercise, model in models.items():
        try:
            predictions[exercise] = model.predict(input_features)[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting {exercise}: {str(e)}")

    return predictions