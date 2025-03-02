"""
Supply Chain Optimization - Common Module

This module contains shared functions and parameters used by the preprocessing DAGs.
"""

from datetime import datetime, timedelta
import traceback
import os

import docker
from airflow.exceptions import AirflowSkipException, AirflowFailException
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from google.cloud import storage

# Define default arguments
DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 3, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define GCP parameters
GCP_CONNECTION_ID = "google_cloud_default"

# Define bucket names
SOURCE_BUCKET_NAME = "full-raw-data"  # Source data bucket
PROCESSED_BUCKET_NAME = (
    "fully-processed-data"  # Destination for processed data
)
METADATA_BUCKET_NAME = "metadata_stats"  # For statistics and metadata


import platform
import docker


def get_docker_client():
    """Return a Docker client that works across macOS, Linux, Windows, and container environments"""
    system_os = platform.system().lower()

    # Check if running inside a container by looking for container-specific files
    in_container = os.path.exists("/.dockerenv") or os.path.exists(
        "/run/.containerenv"
    )

    # Get Docker host from environment variable if set
    docker_host = os.environ.get("DOCKER_HOST")
    print(f"Environment DOCKER_HOST: {docker_host}")

    if docker_host:
        # Use the configured Docker host if explicitly set
        print(f"Using configured DOCKER_HOST: {docker_host}")
        try:
            client = docker.DockerClient(base_url=docker_host)
            # Test connection
            client.ping()
            print(f"Successfully connected to Docker using DOCKER_HOST")
            return client
        except Exception as e:
            print(f"Failed to connect using DOCKER_HOST: {e}")
            # Fall through to other methods

    if in_container:
        print("Detected container environment")
        # Try common Docker socket paths inside containers
        docker_socket_paths = [
            "unix:///var/run/docker.sock",  # Standard path
            "tcp://host.docker.internal:2375",  # Windows/macOS Docker Desktop
            "tcp://172.17.0.1:2375",  # Docker bridge network
        ]

        for socket_path in docker_socket_paths:
            try:
                print(f"Attempting to connect to Docker using: {socket_path}")
                client = docker.DockerClient(base_url=socket_path)
                # Test the connection by calling a simple API endpoint
                client.ping()
                print(f"Successfully connected to Docker using: {socket_path}")
                return client
            except Exception as e:
                print(f"Failed to connect to Docker using {socket_path}: {e}")
                # Check for permission issues and try sudo if available
                if "Permission denied" in str(e) and socket_path.startswith(
                    "unix://"
                ):
                    # Try to fix permissions using sudo if available
                    try:
                        # This only works if the container has sudo and the user has sudo rights
                        print("Attempting to fix permissions with sudo...")
                        import subprocess

                        subprocess.run(
                            ["sudo", "chmod", "666", "/var/run/docker.sock"],
                            check=False,
                        )
                        # Try again
                        client = docker.DockerClient(base_url=socket_path)
                        client.ping()
                        print(
                            "Successfully connected after fixing permissions"
                        )
                        return client
                    except Exception as sudo_e:
                        print(f"Failed to fix permissions: {sudo_e}")

        # If we get here, all connection attempts failed
        raise docker.errors.DockerException(
            "Could not connect to Docker daemon. Make sure the Docker socket is mounted "
            "or DOCKER_HOST is correctly set in the environment. Try adding user:0 in docker-compose.override.yml."
        )
    elif system_os == "windows":
        # Windows uses TCP connection
        print("Detected Windows OS - Using TCP Docker connection")
        return docker.DockerClient(base_url="tcp://host.docker.internal:2375")
    else:
        # macOS and Linux use UNIX socket
        print("Detected macOS/Linux - Using default Docker socket")
        return docker.from_env()


def print_gcs_info(**context):
    """Print information about the GCS event that triggered the DAG"""
    dag_run = context["dag_run"]
    print(f"DAG triggered by event: {dag_run.conf}")

    # Extract parameters from the event
    gcs_bucket = dag_run.conf.get("gcs_bucket", SOURCE_BUCKET_NAME)
    gcs_object = dag_run.conf.get("gcs_object", "")
    event_time = dag_run.conf.get("event_time", "")

    print(f"GCS Event Details:")
    print(f"  Bucket: {gcs_bucket}")
    print(f"  Object: {gcs_object}")
    print(f"  Event Time: {event_time}")

    # Store for downstream tasks
    context["ti"].xcom_push(key="gcs_bucket", value=gcs_bucket)
    context["ti"].xcom_push(key="gcs_object", value=gcs_object)

    return {
        "gcs_bucket": gcs_bucket,
        "gcs_object": gcs_object,
        "event_time": event_time,
    }


def list_new_files(**context):
    """List all files in the bucket and return their names"""
    # Use the bucket from the GCS event if available
    gcs_bucket = (
        context["ti"].xcom_pull(task_ids="print_gcs_info", key="gcs_bucket")
        or SOURCE_BUCKET_NAME
    )

    hook = GCSHook(gcp_conn_id=GCP_CONNECTION_ID)
    files = hook.list(bucket_name=gcs_bucket)

    # If no files found, raise an exception to fail the task
    if not files:
        raise AirflowFailException(
            "No files found in the bucket. Failing the task."
        )

    context["ti"].xcom_push(key="file_list", value=files)
    print(f"Found {len(files)} files in bucket: {files}")
    return files


def run_pre_validation(**context):
    """Run the pre_validation script to validate data before processing"""
    client = get_docker_client()

    try:
        # Get bucket name from xcom
        gcs_bucket = (
            context["ti"].xcom_pull(
                task_ids="print_gcs_info", key="gcs_bucket"
            )
            or SOURCE_BUCKET_NAME
        )

        container = client.containers.get("data-pipeline-container")
        print(f"Container found: {container.name}")

        # Execute the pre_validation.py script with bucket parameter
        exit_code, output = container.exec_run(
            cmd=f"python pre_validation.py --bucket={gcs_bucket}",
            workdir="/app/scripts",
        )

        output_str = output.decode("utf-8")
        print(f"Pre-validation output: {output_str}")

        # Check if there are any files left in the bucket after validation
        hook = GCSHook(gcp_conn_id=GCP_CONNECTION_ID)
        remaining_files = hook.list(bucket_name=gcs_bucket)

        if not remaining_files:
            print("No valid files remain in the bucket after pre-validation.")
            raise AirflowSkipException("No valid files to process")

        # Interpret exit code from pre_validation.py:
        # 0 = All files valid
        # 1 = Some files valid, some invalid (removed)
        # 2 = Critical failure or all files invalid
        if exit_code == 0:
            print("Pre-validation completed successfully for all files")
            context["ti"].xcom_push(key="validation_status", value="full")
        elif exit_code == 1:
            print(
                "Some files failed validation, but continuing with valid ones"
            )
            context["ti"].xcom_push(key="validation_status", value="partial")
        elif exit_code == 2:
            # If there are still files in the bucket, there must be valid files
            if remaining_files:
                print(
                    "Partial validation - continuing with remaining files in bucket"
                )
                context["ti"].xcom_push(
                    key="validation_status", value="partial"
                )
            else:
                print("No valid files to process after pre-validation")
                raise AirflowSkipException(
                    "No valid files after pre-validation"
                )
        else:
            # Unknown exit code, check remaining files to decide
            if remaining_files:
                print(
                    f"Unknown validation status (code {exit_code}), but files remain. Continuing."
                )
                context["ti"].xcom_push(
                    key="validation_status", value="unknown"
                )
            else:
                print(
                    f"Unknown validation status (code {exit_code}) and no files remain."
                )
                raise AirflowSkipException(
                    "Unknown validation status and no files remain"
                )

        return output_str

    except docker.errors.NotFound:
        error_msg = (
            "data-pipeline-container not found. Make sure it's running."
        )
        print(error_msg)
        raise Exception(error_msg)
    except AirflowSkipException:
        # Re-raise AirflowSkipException to ensure downstream tasks are skipped
        raise
    except Exception as e:
        print(f"Error running pre-validation script: {str(e)}")
        raise


def run_preprocessing_script(**context):
    """Run the preprocessing script in the existing data-pipeline-container"""
    client = get_docker_client()

    try:
        # Get bucket name from xcom
        gcs_bucket = (
            context["ti"].xcom_pull(
                task_ids="print_gcs_info", key="gcs_bucket"
            )
            or SOURCE_BUCKET_NAME
        )

        container = client.containers.get("data-pipeline-container")
        print(f"Container found: {container.name}")

        # Execute the preprocessing.py script directly with bucket parameter
        exit_code, output = container.exec_run(
            cmd=f"python preprocessing.py --source_bucket={gcs_bucket} --destination_bucket={PROCESSED_BUCKET_NAME} --delete_after",
            workdir="/app/scripts",
        )

        output_str = output.decode("utf-8")
        print(f"Script output: {output_str}")

        if exit_code != 0:
            print(f"Script failed with exit code: {exit_code}")
            raise Exception(
                f"preprocessing.py failed with exit code {exit_code}"
            )

        print("Preprocessing completed successfully")
        return output_str

    except docker.errors.NotFound:
        error_msg = (
            "data-pipeline-container not found. Make sure it's running."
        )
        print(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        print(f"Error running preprocessing script: {str(e)}")
        raise


def create_preprocessing_tasks(dag):
    """Create all preprocessing tasks for a given DAG"""

    # Print information about the GCS event
    print_gcs_info_task = PythonOperator(
        task_id="print_gcs_info",
        python_callable=print_gcs_info,
        dag=dag,
    )

    # Get list of files in the bucket
    get_file_list_task = PythonOperator(
        task_id="get_file_list",
        python_callable=list_new_files,
        dag=dag,
    )

    # Run pre-validation to check data quality
    run_pre_validation_task = PythonOperator(
        task_id="run_pre_validation",
        python_callable=run_pre_validation,
        dag=dag,
    )

    # Run the preprocessing script in the existing data-pipeline-container
    run_preprocessing_task = PythonOperator(
        task_id="run_preprocessing",
        python_callable=run_preprocessing_script,
        dag=dag,
    )

    # Define the task dependencies
    (
        print_gcs_info_task
        >> get_file_list_task
        >> run_pre_validation_task
        >> run_preprocessing_task
    )

    return {
        "print_gcs_info": print_gcs_info_task,
        "get_file_list": get_file_list_task,
        "run_pre_validation": run_pre_validation_task,
        "run_preprocessing": run_preprocessing_task,
    }
