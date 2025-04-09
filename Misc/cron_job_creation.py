import os
from google.cloud import scheduler_v1
from google.protobuf import duration_pb2, timestamp_pb2
from google.cloud.scheduler_v1.types import HttpTarget, Job, RetryConfig

def create_scheduler_job(
    project_id,
    location_id,
    job_id,
    schedule,
    time_zone,
    http_method,
    url,
    service_account_email=None,
    headers=None,
    body=None,
    retry_attempts=3,
    retry_min_backoff=10,
    retry_max_backoff=300,
    max_retry_duration=600
):
    """
    Creates a Cloud Scheduler job that targets a Cloud Run HTTP function.
    
    Args:
        project_id (str): GCP project ID
        location_id (str): Region where the job should be created (e.g., 'us-central1')
        job_id (str): Unique identifier for the job
        schedule (str): Cron expression for the schedule (e.g., '0 0 * * *' for daily at midnight)
        time_zone (str): Time zone for the schedule (e.g., 'America/New_York')
        http_method (str): HTTP method to use (e.g., 'GET', 'POST')
        url (str): URL of the Cloud Run function to call
        service_account_email (str, optional): Service account email for authentication
        headers (dict, optional): HTTP headers to include in the request
        body (bytes, optional): Request body data as bytes
        retry_attempts (int, optional): Maximum number of retry attempts
        retry_min_backoff (int, optional): Minimum backoff time in seconds
        retry_max_backoff (int, optional): Maximum backoff time in seconds
        max_retry_duration (int, optional): Maximum retry duration in seconds
        
    Returns:
        Job: The created Cloud Scheduler job
    """
    # Initialize the Cloud Scheduler client
    client = scheduler_v1.CloudSchedulerClient()
    
    # Construct the parent resource name
    parent = f"projects/{project_id}/locations/{location_id}"
    
    # Create HTTP target
    http_target = HttpTarget()
    http_target.uri = url
    http_target.http_method = http_method
    
    # Set authentication if service account is provided
    if service_account_email:
        http_target.oidc_token.service_account_email = service_account_email
        http_target.oidc_token.audience = url
    
    # Set headers if provided
    if headers:
        for key, value in headers.items():
            http_target.headers[key] = value
    
    # Set body if provided
    if body:
        http_target.body = body
    
    # Configure retry behavior
    retry_config = RetryConfig(
        retry_count=retry_attempts,
        min_backoff_duration=duration_pb2.Duration(seconds=retry_min_backoff),
        max_backoff_duration=duration_pb2.Duration(seconds=retry_max_backoff),
        max_retry_duration=duration_pb2.Duration(seconds=max_retry_duration),
        max_doublings=5
    )
    
    # Create the job
    job = Job(
        name=f"{parent}/jobs/{job_id}",
        schedule=schedule,
        time_zone=time_zone,
        http_target=http_target,
        retry_config=retry_config
    )
    
    # Submit the job to the scheduler
    result = client.create_job(
        request={"parent": parent, "job": job}
    )
    
    print(f"Created scheduler job: {result.name}")
    return result

def update_scheduler_job(
    project_id,
    location_id,
    job_id,
    schedule=None,
    time_zone=None,
    http_method=None,
    url=None,
    service_account_email=None,
    headers=None,
    body=None,
    retry_attempts=None,
    retry_min_backoff=None,
    retry_max_backoff=None,
    max_retry_duration=None
):
    """
    Updates an existing Cloud Scheduler job.
    
    Args:
        project_id (str): GCP project ID
        location_id (str): Region where the job is located
        job_id (str): Unique identifier for the job
        schedule (str, optional): New cron expression for the schedule
        time_zone (str, optional): New time zone for the schedule
        http_method (str, optional): New HTTP method to use
        url (str, optional): New URL of the Cloud Run function
        service_account_email (str, optional): New service account email
        headers (dict, optional): New HTTP headers
        body (bytes, optional): New request body data
        retry_attempts (int, optional): New maximum number of retry attempts
        retry_min_backoff (int, optional): New minimum backoff time in seconds
        retry_max_backoff (int, optional): New maximum backoff time in seconds
        max_retry_duration (int, optional): New maximum retry duration in seconds
        
    Returns:
        Job: The updated Cloud Scheduler job
    """
    # Initialize the Cloud Scheduler client
    client = scheduler_v1.CloudSchedulerClient()
    
    # Construct the job resource name
    job_name = f"projects/{project_id}/locations/{location_id}/jobs/{job_id}"
    
    # Get the current job configuration
    current_job = client.get_job(name=job_name)
    
    # Use update_mask to specify which fields to update
    update_mask = []
    
    # Create job object with the same name
    updated_job = Job(name=job_name)
    
    # Update schedule if provided
    if schedule:
        updated_job.schedule = schedule
        update_mask.append("schedule")
    
    # Update time zone if provided
    if time_zone:
        updated_job.time_zone = time_zone
        update_mask.append("time_zone")
    
    # Update HTTP target properties if provided
    if url or http_method or service_account_email or headers or body:
        updated_job.http_target = HttpTarget()
        
        # Preserve existing values for fields we're not updating
        if not url:
            updated_job.http_target.uri = current_job.http_target.uri
        else:
            updated_job.http_target.uri = url
            update_mask.append("http_target.uri")
        
        if not http_method:
            updated_job.http_target.http_method = current_job.http_target.http_method
        else:
            updated_job.http_target.http_method = http_method
            update_mask.append("http_target.http_method")
        
        
        if service_account_email:
            updated_job.http_target.oidc_token.service_account_email = service_account_email
            updated_job.http_target.oidc_token.audience = url or current_job.http_target.uri
            update_mask.append("http_target.oidc_token.service_account_email")
            update_mask.append("http_target.oidc_token.audience")
        elif hasattr(current_job.http_target, 'oidc_token') and current_job.http_target.oidc_token.service_account_email:
            updated_job.http_target.oidc_token.service_account_email = current_job.http_target.oidc_token.service_account_email
            updated_job.http_target.oidc_token.audience = current_job.http_target.oidc_token.audience
        
        
        if headers:
            for key, value in headers.items():
                updated_job.http_target.headers[key] = value
            update_mask.append("http_target.headers")
        else:
            for key, value in current_job.http_target.headers.items():
                updated_job.http_target.headers[key] = value
        
        
        if body:
            updated_job.http_target.body = body
            update_mask.append("http_target.body")
        elif current_job.http_target.body:
            updated_job.http_target.body = current_job.http_target.body
    
    if any([retry_attempts is not None, retry_min_backoff is not None, 
            retry_max_backoff is not None, max_retry_duration is not None]):
        
        updated_job.retry_config = RetryConfig()
        
        if retry_attempts is not None:
            updated_job.retry_config.retry_count = retry_attempts
            update_mask.append("retry_config.retry_count")
        else:
            updated_job.retry_config.retry_count = current_job.retry_config.retry_count
        
        if retry_min_backoff is not None:
            updated_job.retry_config.min_backoff_duration = duration_pb2.Duration(seconds=retry_min_backoff)
            update_mask.append("retry_config.min_backoff_duration")
        else:
            updated_job.retry_config.min_backoff_duration = current_job.retry_config.min_backoff_duration
        
        if retry_max_backoff is not None:
            updated_job.retry_config.max_backoff_duration = duration_pb2.Duration(seconds=retry_max_backoff)
            update_mask.append("retry_config.max_backoff_duration")
        else:
            updated_job.retry_config.max_backoff_duration = current_job.retry_config.max_backoff_duration
        
        if max_retry_duration is not None:
            updated_job.retry_config.max_retry_duration = duration_pb2.Duration(seconds=max_retry_duration)
            update_mask.append("retry_config.max_retry_duration")
        else:
            updated_job.retry_config.max_retry_duration = current_job.retry_config.max_retry_duration
    
    result = client.update_job(
        request={
            "job": updated_job,
            "update_mask": {"paths": update_mask}
        }
    )
    
    print(f"Updated scheduler job: {result.name}")
    return result


if __name__ == "__main__":
    project_id = "primordial-veld-450618-n4"
    location_id = "us-central1"
    job_id = "my-cloud-run-job"

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/svaru/Downloads/cloud_run.json"
    

    create_scheduler_job(
        project_id=project_id,
        location_id=location_id,
        job_id=job_id,
        schedule="0 0 * * *",
        time_zone="America/New_York",
        http_method="POST",
        url="https://model-training-148338842941.us-central1.run.app/trigger-training",
        service_account_email="cloud-run@primordial-veld-450618-n4.iam.gserviceaccount.com",
        headers={"Content-Type": "application/json"},
        body=b'{"PROJECT_ID": "primordial-veld-450618-n4", "REGION": "us-central1", "BUCKET_URI": "gs://model_training_1", "IMAGE_URI": "us-central1-docker.pkg.dev/primordial-veld-450618-n4/ml-models/model_training:latest"}',
    )
    

    update_scheduler_job(
    project_id=project_id,
    location_id=location_id,
    job_id=job_id,
    schedule="0 * * * *",
)