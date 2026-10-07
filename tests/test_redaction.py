import io
import logging
import unittest

from src.ai.llm_router import short_error
from src.utils.logger import RedactingFormatter
from src.utils.redact import redact

# Shapes taken from real provider errors seen in the 2026-10-07 run logs
# (identifiers below are fabricated).
GROQ_429 = (
    "Error code: 429 - {'error': {'message': \"Request too large for model `qwen/qwen3.8-27b` in organization "
    "`org_01abcdefghijklmnopqrstuvwx` service tier `on_demand`\"}}"
)
NVIDIA_404 = (
    '404 {"status":404,"title":"Not Found","detail":"Function \'00000000-1111-2222-3333-444444444444\': '
    "Not found for account 'AbCdEf_GhIjKlMnOp-QrStUvWxYz0123456789'\"}"
)
SECRETS = (
    "https://discord.com/api/webhooks/123456789/AbCdEfGhIjKlMn_opqrst",
    "https://newsapi.org/v2/everything?apiKey=0123456789abcdef&q=nvda",
    "AIzaSyA1234567890abcdefghijklmnopqrst",
    "gsk_0123456789abcdefghijklmnop",
    "nvapi-0123456789abcdefghijklmnopqrstuv",
    "Bearer eyJhbGciOi.abc.def",
)


class TestRedact(unittest.TestCase):
    def test_account_identifiers_are_removed(self):
        self.assertNotIn("org_01abcdefghij", redact(GROQ_429))
        self.assertIn("org_***", redact(GROQ_429))
        cleaned = redact(NVIDIA_404)
        self.assertNotIn("AbCdEf_GhIjKl", cleaned)
        self.assertNotIn("00000000-1111", cleaned)
        self.assertIn("Not Found", cleaned)  # still useful for debugging

    def test_credentials_and_url_tokens_are_removed(self):
        for secret in SECRETS:
            cleaned = redact(f"request failed: {secret} (status 401)")
            for fragment in ("AbCdEfGhIjKlMn", "0123456789abcdef", "AIzaSyA123", "gsk_0123", "nvapi-0123", "eyJhbGciOi"):
                self.assertNotIn(fragment, cleaned)
        self.assertIn("https://discord.com/", redact(SECRETS[0]))

    def test_short_error_is_redacted(self):
        self.assertNotIn("org_01abcdefghij", short_error("Groq(qwen)", Exception(GROQ_429)))

    def test_log_lines_and_tracebacks_are_redacted(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(RedactingFormatter("%(message)s"))
        log = logging.getLogger("redaction-test")
        log.addHandler(handler)
        log.propagate = False
        try:
            raise RuntimeError(NVIDIA_404)
        except RuntimeError:
            log.exception("NVIDIA failed for %s", SECRETS[0])
        output = stream.getvalue()
        self.assertNotIn("AbCdEf_GhIjKl", output)
        self.assertNotIn("AbCdEfGhIjKlMn_opqrst", output)
        self.assertIn("RuntimeError", output)


class TestNoLeakInDeliveredReports(unittest.TestCase):
    def test_pipeline_errors_and_report_text_are_clean(self):
        from unittest.mock import patch

        import main
        from src.config import Config

        with patch.object(Config, "WRITE_ARTIFACTS", False), patch.object(
            main.AIAnalyzer, "analyze_stock_market", side_effect=RuntimeError(GROQ_429)
        ):
            result = main.build_reports(main.load_fixture_bundle(main.DEFAULT_FIXTURE), enable_ai=False, dry_run=True)
        recorded = " ".join(result["meta"]["errors"]["report"])
        self.assertIn("org_***", recorded)
        self.assertNotIn("org_01abcdefghij", recorded)
        for payload in (result["stock_payload"], result["ai_payload"]):
            self.assertNotIn("org_", str(payload))


if __name__ == "__main__":
    unittest.main()
