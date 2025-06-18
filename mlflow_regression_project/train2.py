import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from mlflow.models.signature import infer_signature

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, rmse, mae, r2

def main():
    print("🚀 Starting multi-model regression experiment...")

    # Load data
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Define models to try
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            # Predict on test set to infer signature
            preds = model.predict(X_test)
            signature = infer_signature(X_test, preds)
            input_example = pd.DataFrame(X_test[:2], columns=data.feature_names)

            mse, rmse, mae, r2 = evaluate_model(model, X_test, y_test)

            # Log params and metrics
            mlflow.log_param("model_name", name)
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, "max_depth") and model.max_depth:
                mlflow.log_param("max_depth", model.max_depth)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Log model with signature and input example
            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=input_example)


            print(f"✅ {name} done. R2: {r2:.4f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()
