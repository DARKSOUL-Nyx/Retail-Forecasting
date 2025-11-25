# 📈 Retail Sales Forecasting: An End-to-End MLOps Pipeline

This project implements a production-ready MLOps pipeline for retail sales forecasting. It leverages **Apache Airflow** for workflow orchestration, **MLflow** for experiment tracking and model versioning, and **Docker** to containerize the entire data and model lifecycle.

This architecture is designed to be **scalable**, **reproducible**, and features a clear separation of concerns, which is essential for designing **data-intensive applications**.

## 🚀 Key Features

* **Automated Workflow Orchestration:** Uses **Apache Airflow** to manage and schedule the data preprocessing and model training pipeline.
* **Experiment Tracking & Governance:** **MLflow** is integrated to log model parameters, performance metrics (**RMSE, MAE, MAPE**), and register models (`RetailSalesForecaster`) for versioning.
* **Reproducibility with Docker:** The entire application (Airflow, MLflow, Postgres, and the MLOps pipeline) is containerized with **Docker Compose** for consistent, environment-agnostic deployment.
* **Forecasting Engine:** Utilizes the **Prophet** library for time-series forecasting.
* **Interactive Dashboard:** A **Streamlit** application provides an interface to visualize forecasts from the latest production-ready model fetched directly from the MLflow Model Registry.

## 🧱 Project Architecture (Tech Stack)

| Component | Role | Technologies |
| :--- | :--- | :--- |
| **Orchestration** | Automating and scheduling the MLOps workflow. | **Apache Airflow** (v2.9.2) |
| **Experimentation** | Tracking experiments, model parameters, and versioning. | **MLflow** (Tracking Server and Model Registry) |
| **Data Storage** | Persistence for Airflow and MLflow metadata. | **PostgreSQL** |
| **Modeling** | Time-series forecasting model implementation. | **Python, Prophet** |
| **Visualization** | Live dashboard for model predictions and past run metrics. | **Streamlit** |
| **Containerization** | Defining and running the multi-service application environment. | **Docker** & **Docker Compose** |


## 📁 Project Structure

* `airflow/`: Contains the Airflow DAGs for orchestrating the MLOps pipeline, and a custom Dockerfile for Airflow.
    * `airflow/dags/model_training_dag.py`: Defines the training pipeline.
* `data/`: Contains the raw and processed sales data.
* `mlflow/`: Contains the Dockerfile for the MLflow tracking server.
* `scripts/`: Contains Python scripts for data preprocessing and model training.
    * `scripts/train_model.py`: Implements the Prophet model training and MLflow logging.
* `dashboard.py`: The Streamlit web application for live forecasting and metric analysis.
* `docker-compose.yml`: Defines the services for the project (Airflow, MLflow, etc.).

## 🛠️ Getting Started

### Prerequisites

* Docker and Docker Compose installed.

### 1. Launch the Stack

Run the following command from the root directory to build all services and launch the environment:

```bash
docker-compose up --build -d
