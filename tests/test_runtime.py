import unittest

from main import validate_runtime
from src.config import Config


class TestRuntimeValidation(unittest.TestCase):
    def test_validate_runtime_requires_ai_key_when_ai_enabled(self):
        original_gemini = Config.GEMINI_API_KEY
        original_groq = Config.GROQ_API_KEY
        try:
            Config.GEMINI_API_KEY = None
            Config.GROQ_API_KEY = None
            with self.assertRaises(RuntimeError):
                validate_runtime(enable_ai=True, dry_run=True)
        finally:
            Config.GEMINI_API_KEY = original_gemini
            Config.GROQ_API_KEY = original_groq

    def test_validate_runtime_requires_discord_webhook_for_live_delivery(self):
        original_enabled = Config.ENABLE_DISCORD_DELIVERY
        original_webhook = Config.DISCORD_WEBHOOK_URL
        try:
            Config.ENABLE_DISCORD_DELIVERY = True
            Config.DISCORD_WEBHOOK_URL = None
            with self.assertRaises(RuntimeError):
                validate_runtime(enable_ai=False, dry_run=False)
        finally:
            Config.ENABLE_DISCORD_DELIVERY = original_enabled
            Config.DISCORD_WEBHOOK_URL = original_webhook
