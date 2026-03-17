import boto3

from csi_vae.aws.retry import aws_retry


class JobSubmitter:
    """Submits jobs as AWS Batch."""

    def __init__(self, job_queue: str, job_definition: str, region_name: str) -> None:
        """Initialize the submitter with AWS Batch client and job configuration."""
        self.__batch_client = boto3.client("batch", region_name=region_name)
        self.__job_queue = job_queue
        self.__job_definition = job_definition

    @aws_retry
    def submit(self, job_name: str, settings: dict) -> str:
        """Submit a job to AWS Batch with the given environment variables."""
        response = self.__batch_client.submit_job(
            jobName=job_name,
            jobQueue=self.__job_queue,
            jobDefinition=self.__job_definition,
            containerOverrides={
                "environment": [{"name": k, "value": str(v)} for k, v in settings.items() if v is not None],
            },
        )
        return response["jobId"]

    def terminate(self, job_id: str, reason: str | None = None) -> None:
        """Terminate a running AWS Batch job."""
        self.__batch_client.terminate_job(jobId=job_id, reason=reason or "Terminated by JobSubmitter")
