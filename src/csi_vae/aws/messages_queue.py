import json

import boto3

from csi_vae.aws.retry import aws_retry


class QueueNotCreatedError(RuntimeError):
    """Raised when operating on a queue that hasn't been created or is already destroyed."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Queue not created. Call create() or use from_url() before operating.")


class MessagesQueue:
    """AWS SQS-backed message queue with retry logic for transient errors."""

    def __init__(self, region_name: str, url: str | None = None) -> None:
        """Initialize the MessagesQueue.

        Arguments:
            region_name: AWS region for the SQS client.
            url: Optional pre-existing queue URL. If provided, the queue is
                immediately ready to use without calling create().

        """
        self.__sqs = boto3.client("sqs", region_name=region_name)
        self.__url = url

    @staticmethod
    def from_url(url: str, region_name: str) -> "MessagesQueue":
        """Create a MessagesQueue from an existing SQS queue URL."""
        return MessagesQueue(region_name, url=url)

    @property
    def url(self) -> str:
        """The SQS queue URL. Raises QueueNotCreatedError if not yet created."""
        if self.__url is None:
            raise QueueNotCreatedError
        return self.__url

    @aws_retry
    def create(self, name: str) -> None:
        """Create the SQS queue and store its URL.

        Arguments:
            name: Name of the SQS queue to create. Must be unique within the AWS account and region.

        """
        self.__url = self.__sqs.create_queue(QueueName=name)["QueueUrl"]

    @aws_retry
    def destroy(self) -> None:
        """Delete the SQS queue. Raises QueueNotCreatedError if not yet created."""
        self.__sqs.delete_queue(QueueUrl=self.url)
        self.__url = None

    @aws_retry
    def push(self, item: dict) -> None:
        """Push a message onto the queue.

        Arguments:
            item: The message body to send, which will be JSON-encoded before sending.

        """
        self.__sqs.send_message(QueueUrl=self.url, MessageBody=json.dumps(item))

    def pop(
        self,
        max_messages: int = 10,
        filter_values: dict | None = None,
        delete_non_matching: bool = True,
    ) -> list[dict]:
        """Pop up to max_messages messages, deleting each only after it is read.

        Arguments:
            max_messages: Maximum number of messages to retrieve.
            filter_values: Optional dictionary of values to filter messages. Only messages for which
                all keys match their corresponding values will be included in the results and deleted from the queue.
            delete_non_matching: If True, messages that do not match the filter will also be deleted from the queue.

        Returns:
            List of decoded message bodies.

        """
        results = []
        remaining = max_messages

        while remaining > 0:
            messages = self._receive(min(remaining, 10))
            if not messages:
                break

            for msg in messages:
                body = json.loads(msg["Body"])
                if filter_values is None or all(body.get(k) == v for k, v in filter_values.items()):
                    results.append(body)
                    self._delete(msg["ReceiptHandle"])
                elif delete_non_matching:
                    self._delete(msg["ReceiptHandle"])

            remaining -= len(messages)

        return results

    @aws_retry
    def _receive(self, count: int) -> list[dict]:
        return self.__sqs.receive_message(
            QueueUrl=self.url,
            MaxNumberOfMessages=count,
        ).get("Messages", [])

    @aws_retry
    def _delete(self, receipt_handle: str) -> None:
        self.__sqs.delete_message(QueueUrl=self.url, ReceiptHandle=receipt_handle)
