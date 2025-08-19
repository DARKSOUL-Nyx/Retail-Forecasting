import pandas as pd
from prophet import Prophet
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
import os 

# Define paths inside the container
DATA_PATH = "/opt/airflow/data/train.csv"
MLFLOW_TRACKING_URI = "http://mlflow-server:5000" # Use the service name from docker-compose

logging.basicConfig(level=logging.INFO)

def train_sales_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Retail Sales Forecasting - Docker")

    logging.info("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df_subset = df[(df['store'] == 1) & (df['item'] == 1)].copy()
    df_prophet = df_subset.rename(columns={'date': 'ds', 'sales': 'y'})
    logging.info(f"Data prepared. Shape: {df_prophet.shape}")

    with mlflow.start_run() as run:
        logging.info("Starting MLflow run...")
        horizon = 90
        mlflow.log_param("forecast_horizon", horizon)

        model = Prophet()
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        
        actuals = df_prophet['y'][-horizon:]
        predictions = forecast['yhat'][-horizon-len(actuals):-horizon]
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        mlflow.log_metric("rmse", rmse)
        mlflow.prophet.log_model(model, "prophet-sales-forecaster")
        logging.info(f"Model logged with run_id: {run.info.run_id}")

if __name__ == "__main__":
    train_sales_model()