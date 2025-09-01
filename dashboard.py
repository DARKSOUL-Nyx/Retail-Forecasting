import streamlit as st 
import mlflow 
import pandas as pd 

MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.set_page_config(layout="wide")
st.title("Retail Forecasting Dashboard")

st.sidebar.header("Forecasting Options")
days_to_predict = st.sidebar.slider("Days tp ForeCast", 7 , 365 , 90)

try:
    model = mlflow.prophet.load_model("models:/RetailSalesModel/latest")
    st.success("Successfully the Model has Been Loaded")

    future = model.make_future_dataframe(periods=days_to_predict)
    forecast = model.predict(future)

    # --- Display Results ---
    st.subheader("Forecast Plot")
    fig = model.plot(forecast)
    st.pyplot(fig)

    st.subheader("Forecast Data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_predict))

except Exception as e:
    st.error(f"Could not load model or make prediction. Error: {e}")


# --- Display Metrics of Past Runs ---
st.subheader("Past Experiment Runs")
runs_df = mlflow.search_runs(experiment_names=["Retail Sales Forecasting - Docker"])
st.dataframe(runs_df[['metrics.rmse', 'metrics.mae', 'params.forecast_horizon', 'start_time']])

