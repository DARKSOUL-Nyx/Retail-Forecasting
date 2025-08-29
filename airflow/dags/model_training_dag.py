from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='retail_sales_training_pipeline',
    start_date=datetime(2025, 8, 29),
    schedule_interval=None,  # This DAG runs only when manually triggered
    catchup=False,
    tags=['ml', 'forecasting'],
) as dag:
    
    preprocess_data = BashOperator(
        task_id='preprocess_data',
        # This command runs the script using the python inside the container
        bash_command='python /opt/airflow/scripts/preprocess_data.py',
    )

    train_model = BashOperator(
        task_id='train_initial_model',
        bash_command='python /opt/airflow/scripts/train_model.py',
    )

    # Define the dependency: preprocessing must run before training
    preprocess_data >> train_model