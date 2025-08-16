from airflow.decorators import dag, task
from datetime import datetime

# Note: No need to modify sys.path. The script is available in the container.
from scripts.train_model import train_sales_model

@dag(
    dag_id='retail_sales_training_pipeline_docker',
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Trigger manually
    catchup=False,
    tags=['retail', 'forecasting', 'ml', 'docker'],
)
def sales_training_dag():
    
    @task
    def train_and_log_model_task():
        """A task that calls our model training function."""
        train_sales_model()

    train_and_log_model_task()

sales_training_dag()