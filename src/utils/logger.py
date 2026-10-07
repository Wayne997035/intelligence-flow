import logging

from src.utils.redact import redact


class RedactingFormatter(logging.Formatter):
    """Scrub account ids, API keys and URL tokens from every log line,
    including tracebacks and third-party library messages (see redact.py)."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for _handler in logging.getLogger().handlers:
    _handler.setFormatter(RedactingFormatter(_handler.formatter._fmt if _handler.formatter else None))
logger = logging.getLogger('IntelFlow')
