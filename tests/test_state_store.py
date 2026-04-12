from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path

from src.models import IntelligenceItem
from src.utils.state_store import RunStateStore


class TestStateStore(unittest.TestCase):
    def test_filter_new_items_skips_seen_fingerprints_within_ttl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current = datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc)
            store = RunStateStore(
                str(Path(tmpdir) / "state.json"),
                enabled=True,
                ttl_hours=12,
                now_fn=lambda: current,
            )
            items = [
                IntelligenceItem(
                    title="OpenAI release notes",
                    url="https://example.com/openai-release",
                    desc="a",
                    source_name="OpenAI",
                    source_type="official_news",
                )
            ]

            fresh, skipped = store.filter_new_items("ai_news", items, limit=10)
            self.assertEqual(len(fresh), 1)
            self.assertEqual(skipped, 0)
            store.remember("ai_news", fresh)
            store.save()

            second_store = RunStateStore(
                str(Path(tmpdir) / "state.json"),
                enabled=True,
                ttl_hours=12,
                now_fn=lambda: current + timedelta(hours=6),
            )
            fresh_again, skipped_again = second_store.filter_new_items("ai_news", items, limit=10)
            self.assertEqual(len(fresh_again), 0)
            self.assertEqual(skipped_again, 1)

    def test_filter_new_items_allows_seen_fingerprints_after_ttl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current = datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc)
            path = str(Path(tmpdir) / "state.json")
            store = RunStateStore(path, enabled=True, ttl_hours=12, now_fn=lambda: current)
            items = [
                IntelligenceItem(
                    title="OpenAI release notes",
                    url="https://example.com/openai-release",
                    desc="a",
                    source_name="OpenAI",
                    source_type="official_news",
                )
            ]
            fresh, _ = store.filter_new_items("ai_news", items, limit=10)
            store.remember("ai_news", fresh)
            store.save()

            later_store = RunStateStore(
                path,
                enabled=True,
                ttl_hours=12,
                now_fn=lambda: current + timedelta(hours=13),
            )
            fresh_again, skipped_again = later_store.filter_new_items("ai_news", items, limit=10)
            self.assertEqual(len(fresh_again), 1)
            self.assertEqual(skipped_again, 0)
