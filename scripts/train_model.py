import pandas as pd
from prophet import Prophet
import mlflow
import mlflow.prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import logging

# --- Configuration ---
# Define paths that exist INSIDE the Docker container.
DATA_PATH = "/opt/airflow/data/processed/initial_train.csv"
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_sales_model():
    """
    Trains a Prophet forecasting model on the initial dataset
    and logs the results to MLflow.
    """
    # --- 1. MLflow Setup ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Retail Sales Forecasting - Docker")
    logging.info("MLflow experiment set.")

    # --- 2. Load and Prepare Data ---
    logging.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])

    # Filter for a single store-item combination for a focused forecast
    df_subset = df[(df['store'] == 1) & (df['item'] == 1)].copy()

    # Prophet requires columns to be named 'ds' (datestamp) and 'y' (target)
    df_prophet = df_subset.rename(columns={'date': 'ds', 'sales': 'y'})
    logging.info(f"Data prepared. Shape: {df_prophet.shape}")

    # --- 3. Model Training and Evaluation ---
    with mlflow.start_run() as run:
        logging.info(f"Starting MLflow run: {run.info.run_name}")

        # --- Parameters ---
        forecast_horizon = 90
        mlflow.log_param("forecast_horizon", forecast_horizon)
        logging.info(f"Forecast Horizon: {forecast_horizon} days")

        # --- Train/Validation Split ---
        # Use all data except the last 'forecast_horizon' days for training
        train_df = df_prophet.iloc[:-forecast_horizon]
        # Use the last 'forecast_horizon' days for validation
        val_df = df_prophet.iloc[-forecast_horizon:]
        logging.info(f"Training data size: {len(train_df)}, Validation data size: {len(val_df)}")

        # --- Model Fitting ---
        model = Prophet()
        model.fit(train_df)
        logging.info("Prophet model fitting complete.")

        # --- Forecasting ---
        # Create a dataframe for future predictions
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)

        # --- Evaluation ---
        # Isolate the predictions for the validation period
        predictions = forecast['yhat'].iloc[-forecast_horizon:]
        actuals = val_df['y']

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        # --- 4. Logging to MLflow ---
        logging.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # Log the trained Prophet model as an artifact
        mlflow.prophet.log_model(model, "prophet-initial-model")
        logging.info("Model and metrics logged successfully to MLflow.")
        
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}")

if __name__ == "__main__":
    train_sales_model()