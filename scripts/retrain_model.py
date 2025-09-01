from airflow.operators.bash import BashOperator


datachunks = []

with DAG(...) as dag:
    previous_task = None
    for chunk in datachunks:
        retrian_task = BashOperator(
            task_id = f"retrain_with_{chunk.split('.')[0]}",
            bash_command=f"python /opt/airfow/scripts/retrain_model.py --data-file{chunk}"
        )
        if previous_task is not None:
            previous_task >> retrian_task
        previous_task = retrian_task