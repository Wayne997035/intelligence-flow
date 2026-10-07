import importlib
import os
import unittest
from unittest.mock import patch

import src.config as config_module


def _load(env):
    with patch.dict(os.environ, env, clear=False):
        return importlib.reload(config_module).Config


class TestConfigFromGitHubVariables(unittest.TestCase):
    """GitHub passes unset repository variables as empty strings."""

    def tearDown(self):
        importlib.reload(config_module)

    def test_empty_values_fall_back_to_defaults(self):
        config = _load(
            {
                "US_STOCKS": "",
                "ENABLE_GOOGLE_NEWS": "",
                "STOCK_NEWS_LOOKBACK_DAYS": "",
                "AI_PROVIDER_ORDER": " ",
                "STOCK_NAME_ALIASES": "",
            }
        )
        self.assertIn("SPCX", config.US_STOCKS)
        self.assertTrue(config.ENABLE_GOOGLE_NEWS)
        self.assertEqual(config.STOCK_NEWS_LOOKBACK_DAYS, 7)
        self.assertEqual(config.AI_PROVIDER_ORDER, ["gemini", "nvidia", "groq"])
        self.assertIn("SPCX", config.STOCK_NAME_ALIASES)

    def test_set_values_override_defaults(self):
        config = _load(
            {
                "US_STOCKS": "NVDA, MSFT ,",
                "TW_STOCKS": "2330",
                "ENABLE_TICKER_NEWS": "false",
                "AI_NEWS_LOOKBACK_DAYS": "3",
                "STOCK_NAME_ALIASES": "MSFT:Microsoft|Azure",
            }
        )
        self.assertEqual(config.US_STOCKS, ["NVDA", "MSFT"])
        self.assertEqual(config.TW_STOCKS, ["2330"])
        self.assertFalse(config.ENABLE_TICKER_NEWS)
        self.assertEqual(config.AI_NEWS_LOOKBACK_DAYS, 3)
        self.assertEqual(config.STOCK_NAME_ALIASES, {"MSFT": ["Microsoft", "Azure"]})

    def test_invalid_integer_uses_default(self):
        self.assertEqual(_load({"HISTORY_TTL_HOURS": "abc"}).HISTORY_TTL_HOURS, 26)


if __name__ == "__main__":
    unittest.main()
