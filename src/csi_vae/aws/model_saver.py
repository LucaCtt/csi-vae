import io

import boto3
import torch
from torch import nn

from csi_vae.aws.retry import aws_retry


class ModelSaver:
    """Class responsible for saving job results, such as trained models, to S3."""

    def __init__(self, bucket_name: str, region_name: str) -> None:
        """Initialize the ModelSaver with the S3 bucket name and AWS region."""
        self.__bucket_name = bucket_name
        self.__s3 = boto3.client("s3", region_name=region_name)

    @aws_retry
    def save_model(self, model: nn.Module, key: str) -> None:
        """Save the model to S3 with a filename that includes the study name, job number, and seed."""
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        self.__s3.put_object(Bucket=self.__bucket_name, Key=key, Body=buffer)
