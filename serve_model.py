from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np

app = FastAPI()

# YOUR RUN ID (from mlflow runs list)
RUN_ID = "3483d95c615b4e8a948b2532271679a2"

model_uri = f"runs:/{RUN_ID}/model"
model = mlflow.sklearn.load_model(model_uri)

print(f"Model loaded from {model_uri}")

class Features(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "ML Model API - Diabetes Predictor", "run_id": RUN_ID}

@app.post("/predict")
def predict(features: Features):
    try:
        input_array = np.array(features.features).reshape(1, -1)
        prediction = model.predict(input_array)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)