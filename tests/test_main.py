import unittest
from pathlib import Path

from main import run_job


class TestMainFixtureRun(unittest.TestCase):
    def test_run_job_with_fixture(self):
        fixture_path = Path(__file__).parent / "fixtures" / "sample_bundle.json"
        result = run_job(
            use_fixture=True,
            fixture_path=fixture_path,
            enable_ai=False,
            dry_run=True,
        )

        self.assertGreater(len(result["stock_report"].items), 0)
        self.assertGreater(len(result["ai_report"].items), 0)
        self.assertIn("embeds", result["stock_payload"])
        self.assertIn("embeds", result["ai_payload"])
