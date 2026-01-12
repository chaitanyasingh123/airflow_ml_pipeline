from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import pandas as pd
import os
import sys

# Path Setup
MODEL_DIR = "/opt/airflow/sample_ml_model"
sys.path.append(MODEL_DIR)

# Import your custom logic
try:
    from data_prep_logic import prepare_and_predict
except ImportError:
    # Fallback for initial parsing if file isn't mounted yet
    def prepare_and_predict(*args, **kwargs): pass

# --- Custom Python Functions ---

def fetch_to_s3_logic():
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    df = pg_hook.get_pandas_df(sql="SELECT * FROM sch_ml_data.sample_data")
    
    temp_path = "/tmp/raw_ml_data.csv"
    df.to_csv(temp_path, index=False)
    
    s3_hook = S3Hook(aws_conn_id='s3_local')
    s3_hook.load_file(
        filename=temp_path,
        key='raw/student_data.csv',
        bucket_name='airflow-intermediate-data',
        replace=True
    )

def run_ml_inference_logic():
    input_s3 = "/tmp/raw_ml_data.csv"
    model_path = os.path.join(MODEL_DIR, "student_model.sav")
    output_csv = "/tmp/predictions.csv"
    
    prepare_and_predict(
        input_file=input_s3,
        model_file=model_path,
        output_file=output_csv
    )
    return output_csv

def load_csv_to_postgres(**kwargs):
    ti = kwargs['ti']
    csv_path = ti.xcom_pull(task_ids='run_inference')
    df = pd.read_csv(csv_path)
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = pg_hook.get_sqlalchemy_engine()
    # We load into the table created by the SQLOperator
    df.to_sql(
        name='output_data', 
        con=engine, 
        schema='sch_ml_data', 
        if_exists='append', 
        index=False
    )

# --- DAG Definition ---

with DAG(
    dag_id='student_prediction_sql_operator',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    template_searchpath="/opt/airflow/dags/sql/" # Optional: for .sql files
) as dag:

    # 1. Use SQL Operator to ensure the schema/table exists
    # This replaces the logic previously hidden in Python
    prepare_output_table = SQLExecuteQueryOperator(
        task_id='prepare_output_table',
        conn_id='postgres_default',
        sql="""
            CREATE SCHEMA IF NOT EXISTS sch_ml_data;
            DROP TABLE IF EXISTS sch_ml_data.output_data;
            CREATE TABLE sch_ml_data.output_data (
                hours_studied INT,
                prev_grade INT,
                target_score INT,
                is_high_effort INT,
                predicted_score FLOAT
            );
        """
    )

    extract_to_s3 = PythonOperator(
        task_id='extract_to_s3',
        python_callable=fetch_to_s3_logic
    )

    run_inference = PythonOperator(
        task_id='run_inference',
        python_callable=run_ml_inference_logic
    )

    load_to_postgres = PythonOperator(
        task_id='load_to_postgres',
        python_callable=load_csv_to_postgres
    )

    # Workflow
    prepare_output_table >> extract_to_s3 >> run_inference >> load_to_postgres