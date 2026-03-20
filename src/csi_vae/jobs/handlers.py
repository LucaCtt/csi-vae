import json
import logging
import traceback
from datetime import UTC, datetime
from typing import Any

from csi_vae.aws import MessagesQueue


def _timestamp_to_iso(timestamp: float) -> str:
    """Convert a timestamp to an ISO 8601 formatted string.

    Arguments:
        timestamp: The timestamp to convert, in seconds since the epoch.

    Returns:
        An ISO 8601 formatted string representing the given timestamp.

    """
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


class _BaseJobHandler(logging.Handler):
    """Base handler that enriches log records with job metadata."""

    def __init__(self, n_gaussians: int, trial_number: int, seed: int) -> None:
        super().__init__()
        self._n_gaussians = n_gaussians
        self._trial_number = trial_number
        self._seed = seed

    def _build_message(self, record: logging.LogRecord) -> dict[str, Any]:
        message: dict[str, Any] = record.msg if isinstance(record.msg, dict) else {"message": record.getMessage()}
        if record.exc_info:
            message["error"] = str(record.exc_info[1])
            message["traceback"] = traceback.format_exception(*record.exc_info)
        return {
            **message,
            "date_time": _timestamp_to_iso(record.created),
            "n_gaussians": self._n_gaussians,
            "trial_number": self._trial_number,
            "seed": self._seed,
        }


class StreamHandler(_BaseJobHandler, logging.StreamHandler):
    """Logging handler that outputs enriched log records to the console."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by printing it to the console."""
        message = self._build_message(record)
        clean_record = logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg=json.dumps(message),
            args=(),
            exc_info=None,  # already captured in message dict
        )
        logging.StreamHandler.emit(self, clean_record)


class QueueHandler(_BaseJobHandler, logging.Handler):
    """Logging handler that sends enriched log records to an AWS SQS queue."""

    def __init__(self, queue: MessagesQueue, n_gaussians: int, trial_number: int, seed: int) -> None:
        """Initialize the QueueHandler with the given queue and trial metadata."""
        super().__init__(n_gaussians, trial_number, seed)
        self.__queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by sending it to the configured AWS SQS queue."""
        try:
            self.__queue.push(self._build_message(record))
        except Exception:  # noqa: BLE001
            self.handleError(record)
