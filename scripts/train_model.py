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

# Set up basic logging to see detailed output in Airflow
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_sales_model():
    """
    Trains a Prophet forecasting model, logs it to MLflow, and includes
    debugging steps and best practices for experimentation.
    """
    # --- 1. MLflow Setup ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Retail Sales Forecasting - Docker")
    logging.info("MLflow experiment set.")

    # --- 2. Load and Prepare Data ---
    logging.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df_subset = df[(df['store'] == 1) & (df['item'] == 1)].copy()
    df_prophet = df_subset.rename(columns={'date': 'ds', 'sales': 'y'})
    logging.info(f"Data prepared. Shape: {df_prophet.shape}")

    # --- 3. Model Training and Evaluation ---
    with mlflow.start_run() as run:
        logging.info(f"Starting MLflow run: {run.info.run_name}")

        artifact_uri = mlflow.get_artifact_uri()
        logging.info(f"--- MLflow artifacts will be stored at: {artifact_uri} ---")

        # --- Parameters for Experimentation ---
        # You can change these values and rerun to create new experiments
        forecast_horizon = 90
        changepoint_scale = 0.05  # Prophet's flexibility in detecting trend changes

        mlflow.log_param("forecast_horizon", forecast_horizon)
        mlflow.log_param("changepoint_prior_scale", changepoint_scale)
        logging.info(f"Parameters logged: Horizon={forecast_horizon}, Changepoint Scale={changepoint_scale}")

        # --- Train/Validation Split ---
        train_df = df_prophet.iloc[:-forecast_horizon]
        val_df = df_prophet.iloc[-forecast_horizon:]
        logging.info(f"Training data size: {len(train_df)}, Validation data size: {len(val_df)}")

        # --- Model Fitting ---
        model = Prophet(changepoint_prior_scale=changepoint_scale)
        model.fit(train_df)
        logging.info("Prophet model fitting complete.")

        # --- Forecasting ---
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)

        # --- Evaluation ---
        predictions = forecast['yhat'].iloc[-forecast_horizon:]
        actuals = val_df['y']

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # --- 4. Logging to MLflow ---
        logging.info(f"Metrics calculated: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

        # --- DEBUGGING STEP: Log a simple text artifact first ---
        logging.info("Attempting to log a simple text artifact...")
        try:
            mlflow.log_text("This is a test to see if any artifact can be saved.", artifact_file="test_log.txt")
            logging.info(">>> Simple text artifact 'test_log.txt' logged successfully. <<<")
        except Exception as e:
            logging.error(f">>> FAILED to log simple text artifact. Error: {e} <<<")


        # --- Log the Prophet model and register it for versioning ---
        logging.info("Attempting to log Prophet model...")
        try:
            mlflow.prophet.log_model(
                model,
                artifact_path="prophet-model",  # This is the folder name within artifacts
                registered_model_name="RetailSalesForecaster" # This versions the model
            )
            logging.info(">>> Prophet model logged and registered successfully. <<<")
        except Exception as e:
            logging.error(f">>> FAILED to log Prophet model. Error: {e} <<<")

        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}")

if __name__ == "__main__":
    train_sales_model()
