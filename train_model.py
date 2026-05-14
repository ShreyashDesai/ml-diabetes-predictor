import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Load data
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions) ** 0.5  # Fixed line

print(f"RMSE: {rmse:.2f}")

# Log with MLflow
mlflow.set_experiment("diabetes_prediction")
with mlflow.start_run():
    mlflow.log_param("alpha", 1.0)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Run logged to MLflow")
    print("Done! Run 'mlflow ui' to see your experiment")