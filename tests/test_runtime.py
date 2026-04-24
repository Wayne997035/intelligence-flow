import unittest
from argparse import Namespace

from main import resolve_runtime_options, validate_runtime
from src.config import Config


class TestRuntimeValidation(unittest.TestCase):
    def test_resolve_runtime_options_uses_config_defaults(self):
        original_dry_run = Config.DRY_RUN
        original_ai = Config.ENABLE_AI_ANALYSIS
        original_fixture = Config.USE_FIXTURE_DATA
        try:
            Config.DRY_RUN = True
            Config.ENABLE_AI_ANALYSIS = True
            Config.USE_FIXTURE_DATA = False

            dry_run, use_fixture, enable_ai = resolve_runtime_options(
                Namespace(live_delivery=False, use_fixture=False, enable_ai=False)
            )

            self.assertTrue(dry_run)
            self.assertFalse(use_fixture)
            self.assertTrue(enable_ai)
        finally:
            Config.DRY_RUN = original_dry_run
            Config.ENABLE_AI_ANALYSIS = original_ai
            Config.USE_FIXTURE_DATA = original_fixture

    def test_resolve_runtime_options_live_delivery_overrides_dry_run(self):
        original_dry_run = Config.DRY_RUN
        try:
            Config.DRY_RUN = True
            dry_run, _, _ = resolve_runtime_options(
                Namespace(live_delivery=True, use_fixture=False, enable_ai=False)
            )

            self.assertFalse(dry_run)
        finally:
            Config.DRY_RUN = original_dry_run

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
