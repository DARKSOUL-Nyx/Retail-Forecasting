FROM apache/airflow:2.9.2

USER airflow

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install mlflow==2.14.1
