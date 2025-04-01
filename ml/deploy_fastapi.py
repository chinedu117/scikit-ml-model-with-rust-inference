# FastAPI and HTTPException from FastAPI for building the API and handling errors.
from fastapi import FastAPI, HTTPException

# BaseModel from Pydantic to define a data structure that validates incoming request data.
from pydantic import BaseModel

# Typing imports (Dict, Any, Union, List) for type annotations.
from typing import Dict, Any, Union, List

# joblib for loading previously trained machine learning models.
import joblib

# pandas (pd) for data manipulation and creating DataFrames.
import pandas as pd

# os for file path operations.
import os


# Pydantic Model Definition. A class `SampleData` inherits from `BaseModel`:
# It has one field, `data`, which can be either a dictionary or a list of dictionaries.
# This allows the endpoint to accept either a single record or multiple records as input.
class SampleData(BaseModel):
    data: Dict[str, Any] | List[Dict[str, Any]] = []


# Loading the Models. A dictionary `models` is created to store the loaded models.
# `model_files` is another dictionary mapping model names (e.g., "DecisionTree") to their respective file paths in the "models" folder.
# The `for` loop iterates through each model file:
# For each model, it loads the model from the corresponding file using `joblib.load`.
# The loaded model is stored in the `models` dictionary under its model name.
models = {}
model_files = {
    "catboost": os.path.join("models", "catboost_model.pkl"),
    "lightgbm": os.path.join("models", "lightgbm_model.pkl"),
    "naives_bayes": os.path.join("models", "naive_bayes_model.pkl"),
    "SVC": os.path.join("models", "SVC_model.pkl"),
    "xgboost": os.path.join("models", "xgboost_model.pkl"),
}
for name, filepath in model_files.items():
    models[name] = joblib.load(filepath)

# Creating the FastAPI App. The FastAPI application is instantiated with `app = FastAPI()`.
app = FastAPI()

## Defining the Predict Endpoint
# The `@app.post("/predict")` decorator defines a POST endpoint accessible at the URL `/predict`.
# The `predict` function takes one parameter `sample_data` which is automatically validated against the `SampleData` model.
# Inside the function:
# The code attempts to extract the "data" from `sample_data`.
# It converts this data into a pandas DataFrame. If the input is a single dictionary, it is wrapped in a list.
# If any error occurs during the conversion, an HTTPException is raised with a 400 status code and an error message.


@app.post("/predict")
def predict(sample_data: SampleData):
    try:
        # Convert input data to a DataFrame
        sample = sample_data.data
        df = pd.DataFrame(sample if isinstance(sample, list) else [sample])

    except Exception as e:

        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    # Running the Predictions
    # A new dictionary `predictions` is created to store outputs from each model.
    # The code loops through each loaded model in the `models` dictionary:
    # It calls the `predict` method of each model using the DataFrame.
    # The resulting predictions (typically a numpy array) are converted to a list and stored under the corresponding model name.
    # Finally, the predictions are returned as JSON.

    predictions = {}
    for name, model in models.items():
        preds = model.predict(df)
        predictions[name] = preds.tolist()

    return {"predictions": predictions}


# To run: uvicorn deploy_fastapi:app --reload

# To access the Swagger UI: http://127.0.0.1:8000/docs

# For Postman: http://127.0.0.1:8000/predict
