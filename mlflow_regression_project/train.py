import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_diabetes

print("🚀 Starting simple tuning...")

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try different n_estimators
for n in [10, 50, 100, 200]:
    with mlflow.start_run():

        # Define model
        model = RandomForestRegressor(n_estimators=n, random_state=42)

        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log everything
        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ n_estimators={n} | R2: {r2:.4f}, RMSE: {rmse:.2f}")
