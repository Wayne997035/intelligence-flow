from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import Config
from src.models import IntelligenceItem
from src.utils.state_store import RunStateStore


class TestStateStore(unittest.TestCase):
    def setUp(self):
        # Deterministically force the file backend regardless of whatever
        # Notion state config a developer's .env.local happens to carry --
        # these tests exercise RunStateStore/TTL logic, not backend
        # selection (that's covered in tests/test_state_backends.py), and
        # MUST NOT make real network calls.
        patcher_db = patch.object(Config, "NOTION_STATE_DB_ID", None)
        patcher_ds = patch.object(Config, "NOTION_STATE_DS_ID", None)
        patcher_db.start()
        patcher_ds.start()
        self.addCleanup(patcher_db.stop)
        self.addCleanup(patcher_ds.stop)

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

    def test_ttl_26h_still_active_at_20_hours(self):
        # Overnight schedule gap is 14.5h (20:00 -> next 10:30); 20h is
        # comfortably inside the 26h TTL, so a duplicate seen 20h ago MUST
        # still be recognised and skipped.
        with tempfile.TemporaryDirectory() as tmpdir:
            current = datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc)
            path = str(Path(tmpdir) / "state.json")
            store = RunStateStore(path, enabled=True, ttl_hours=26, now_fn=lambda: current)
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

            store_at_20h = RunStateStore(
                path,
                enabled=True,
                ttl_hours=26,
                now_fn=lambda: current + timedelta(hours=20),
            )
            fresh_again, skipped_again = store_at_20h.filter_new_items("ai_news", items, limit=10)
            self.assertEqual(len(fresh_again), 0)
            self.assertEqual(skipped_again, 1)

    def test_ttl_26h_expired_at_28_hours(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current = datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc)
            path = str(Path(tmpdir) / "state.json")
            store = RunStateStore(path, enabled=True, ttl_hours=26, now_fn=lambda: current)
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

            store_at_28h = RunStateStore(
                path,
                enabled=True,
                ttl_hours=26,
                now_fn=lambda: current + timedelta(hours=28),
            )
            fresh_again, skipped_again = store_at_28h.filter_new_items("ai_news", items, limit=10)
            self.assertEqual(len(fresh_again), 1)
            self.assertEqual(skipped_again, 0)

    def test_history_ttl_hours_default_is_26(self):
        self.assertEqual(Config.HISTORY_TTL_HOURS, 26)

    def test_run_state_store_uses_file_backend_by_default_in_tests(self):
        # Guards against a future .env.local accidentally routing these
        # tests to a real Notion backend: with NOTION_STATE_DB_ID/DS_ID
        # patched to None (see setUp), RunStateStore MUST resolve to the
        # local file backend.
        from src.utils.state_backends import FileStateBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStateStore(str(Path(tmpdir) / "state.json"), enabled=True)
            self.assertIsInstance(store._backend, FileStateBackend)
