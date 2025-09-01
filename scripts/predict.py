import mlflow 
import pandas as pd 


MLFLOW_TRACKING_URI = "http://localhost:5000"

RUN_ID = ""
MODEL_PATH = f""


def get_forecast(days_to_predict = 30):
    print(f"Loading model from run : {RUN_ID}")
    
    model = mlflow.pyfunc.load_model(MODEL_PATH)

    print("Model loaded")
    future_df = model.make_future_dataframe(periods = days_to_predict)
    forecast_df = model.predict(future_df)

    print(f"\n--- Forecast for the next {days_to_predict} days ---")
    print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_predict))
    return forecast_df

if __name__ == "__main__":
    get_forecast()
    
    