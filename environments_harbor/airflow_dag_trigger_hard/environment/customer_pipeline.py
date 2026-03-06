from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import timedelta

# Missing datetime import for start_date
# Missing default_args dictionary

dag = DAG(
    # Missing dag_id parameter
    schedule_interval='@invalid',  # Invalid schedule value
    start_date='2024-01-01',  # Wrong type - should be datetime object
    # Missing catchup parameter
)

# Hardcoded values instead of using dag_run.conf
processing_date = "2024-01-01"
data_source = "/data/customers.csv"
batch_size = 100

def validate_data(**context):
    print(f"Validating data from {data_source}")
    # Should use dag_run.conf.get() for dynamic values
    print(f"Processing date: {processing_date}")
    return True

def transform_data(**context):
    print(f"Transforming data with batch size: {batch_size}")
    # Hardcoded batch_size instead of runtime config
    return True

def load_data(**context):
    print("Loading transformed data")
    return True

# Tasks with incorrect context manager usage
validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    # Missing dag parameter
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

cleanup_task = BashOperator(
    task_id='cleanup',
    bash_command='echo "Cleanup complete"',
    dag=dag
)

# Incorrect dependency syntax - using > instead of >>
validate_task > transform_task
# Missing dependency between transform and load
# Using wrong operator
load_task << cleanup_task  # Incorrect order - cleanup should be last