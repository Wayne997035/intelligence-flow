import json
import unittest
from unittest.mock import MagicMock

from src.utils.logger import logger
from src.utils.state_backends import (
    FileStateBackend,
    NotionStateBackend,
    resolve_backend,
)


class TestNotionStateBackendLoad(unittest.TestCase):
    def test_load_queries_data_sources_not_databases(self):
        client = MagicMock()
        client.data_sources.query.return_value = {"results": []}
        backend = NotionStateBackend(client, "db-1", "ds-1")

        result = backend.load()

        self.assertEqual(result, {})
        client.data_sources.query.assert_called_once_with(
            data_source_id="ds-1",
            filter={"property": "Name", "title": {"equals": "_state"}},
            page_size=1,
        )
        client.databases.query.assert_not_called()

    def test_load_returns_parsed_state_from_code_block(self):
        state = {"ai_news": [{"fingerprint": "abc", "seen_at": "2026-08-01T00:00:00+00:00"}]}
        content = json.dumps(state, ensure_ascii=False, sort_keys=True)
        client = MagicMock()
        client.data_sources.query.return_value = {"results": [{"id": "page-1"}]}
        client.blocks.children.list.return_value = {
            "results": [
                {
                    "id": "block-1",
                    "type": "code",
                    "code": {"rich_text": [{"text": {"content": content}}]},
                }
            ]
        }
        backend = NotionStateBackend(client, "db-1", "ds-1")

        result = backend.load()

        self.assertEqual(result, state)
        self.assertEqual(backend._page_id, "page-1")
        self.assertEqual(backend._code_block_id, "block-1")

    def test_load_returns_empty_when_no_matching_row(self):
        client = MagicMock()
        client.data_sources.query.return_value = {"results": []}
        backend = NotionStateBackend(client, "db-1", "ds-1")

        result = backend.load()

        self.assertEqual(result, {})
        self.assertIsNone(backend._page_id)

    def test_load_returns_empty_and_warns_on_api_error(self):
        """Mutation self-proof A target: removing the try/except in
        NotionStateBackend.load() makes this test fail (RuntimeError
        propagates instead of degrading to {})."""
        client = MagicMock()
        client.data_sources.query.side_effect = RuntimeError("notion is down")
        backend = NotionStateBackend(client, "db-1", "ds-1")

        with self.assertLogs(logger, level="WARNING") as captured:
            result = backend.load()

        self.assertEqual(result, {})
        self.assertTrue(
            any("Failed to load state from Notion" in line for line in captured.output)
        )


class TestNotionStateBackendSave(unittest.TestCase):
    def test_save_creates_page_when_no_existing_page(self):
        client = MagicMock()
        client.data_sources.query.return_value = {"results": []}  # confirm-check: no existing row
        client.pages.create.return_value = {"id": "page-1"}
        client.blocks.children.list.return_value = {
            "results": [{"id": "block-1", "type": "code", "code": {"rich_text": []}}]
        }
        backend = NotionStateBackend(client, "db-1", "ds-1")

        backend.save({"ai_news": []})

        client.data_sources.query.assert_called_once()  # confirm-check ran before create
        client.pages.create.assert_called_once()
        kwargs = client.pages.create.call_args.kwargs
        self.assertEqual(kwargs["parent"], {"database_id": "db-1"})
        self.assertEqual(
            kwargs["properties"]["Name"]["title"][0]["text"]["content"], "_state"
        )
        self.assertEqual(backend._page_id, "page-1")
        self.assertEqual(backend._code_block_id, "block-1")

    def test_save_updates_existing_row_when_page_id_unknown_but_row_exists(self):
        """Mutation self-proof target: this is the FIRST new test added
        for this fix. Removing the `_find_existing_page()` guard in
        save() (i.e. reverting to the old "if _page_id is None: go
        straight to pages.create" behaviour) makes this test fail,
        because save() would then blindly create a duplicate `_state`
        row instead of updating the one Notion already has."""
        client = MagicMock()
        client.data_sources.query.return_value = {"results": [{"id": "page-existing"}]}
        client.blocks.children.list.return_value = {
            "results": [{"id": "block-existing", "type": "code", "code": {"rich_text": []}}]
        }
        backend = NotionStateBackend(client, "db-1", "ds-1")
        self.assertIsNone(backend._page_id)  # fresh instance, no prior load()

        backend.save({"ai_news": []})

        client.pages.create.assert_not_called()
        client.blocks.update.assert_called_once()
        kwargs = client.blocks.update.call_args.kwargs
        self.assertEqual(kwargs["block_id"], "block-existing")
        self.assertEqual(backend._page_id, "page-existing")
        self.assertEqual(backend._code_block_id, "block-existing")

    def test_save_creates_page_once_when_confirm_query_finds_no_existing_row(self):
        """Dedicated regression test for the no-existing-row branch of the
        new confirm-check (kept separate from
        test_save_creates_page_when_no_existing_page, which predates this
        fix and asserts the resulting page/block payload shape)."""
        client = MagicMock()
        client.data_sources.query.return_value = {"results": []}
        client.pages.create.return_value = {"id": "page-new"}
        client.blocks.children.list.return_value = {
            "results": [{"id": "block-new", "type": "code", "code": {"rich_text": []}}]
        }
        backend = NotionStateBackend(client, "db-1", "ds-1")

        backend.save({"ai_news": []})

        client.pages.create.assert_called_once()

    def test_save_confirm_query_failure_warns_and_falls_back_to_create(self):
        client = MagicMock()
        client.data_sources.query.side_effect = RuntimeError("notion is down")
        client.pages.create.return_value = {"id": "page-1"}
        client.blocks.children.list.return_value = {
            "results": [{"id": "block-1", "type": "code", "code": {"rich_text": []}}]
        }
        backend = NotionStateBackend(client, "db-1", "ds-1")

        with self.assertLogs(logger, level="WARNING") as captured:
            backend.save({"ai_news": []})  # MUST NOT raise

        self.assertTrue(
            any(
                "Failed to check for existing Notion state row before save" in line
                for line in captured.output
            )
        )
        client.pages.create.assert_called_once()
        self.assertEqual(backend._page_id, "page-1")

    def test_save_updates_existing_block_in_place(self):
        client = MagicMock()
        backend = NotionStateBackend(client, "db-1", "ds-1")
        backend._page_id = "page-1"
        backend._code_block_id = "block-1"

        backend.save({"ai_news": []})

        client.blocks.update.assert_called_once()
        kwargs = client.blocks.update.call_args.kwargs
        self.assertEqual(kwargs["block_id"], "block-1")
        client.pages.create.assert_not_called()

    def test_save_returns_and_warns_on_api_error_without_raising(self):
        client = MagicMock()
        client.data_sources.query.return_value = {"results": []}  # confirm-check: no existing row
        client.pages.create.side_effect = RuntimeError("rate limited")
        backend = NotionStateBackend(client, "db-1", "ds-1")

        with self.assertLogs(logger, level="WARNING") as captured:
            backend.save({"ai_news": []})  # MUST NOT raise

        self.assertTrue(
            any("Failed to save state to Notion" in line for line in captured.output)
        )

    def test_save_chunks_large_payload_and_load_round_trips(self):
        large_state = {
            "ai_news": [
                {"fingerprint": f"fp-{i:04d}" * 5, "seen_at": "2026-08-01T00:00:00+00:00"}
                for i in range(200)
            ]
        }
        content = json.dumps(large_state, ensure_ascii=False, sort_keys=True)
        self.assertGreater(len(content), 2000)

        captured = {}
        client = MagicMock()

        def fake_create(**kwargs):
            captured["code"] = kwargs["children"][0]["code"]
            return {"id": "page-1"}

        def fake_list(block_id, **kwargs):
            return {"results": [{"id": "block-1", "type": "code", "code": captured["code"]}]}

        client.data_sources.query.return_value = {"results": []}  # confirm-check: no existing row
        client.pages.create.side_effect = fake_create
        client.blocks.children.list.side_effect = fake_list

        save_backend = NotionStateBackend(client, "db-1", "ds-1")
        save_backend.save(large_state)

        rich_text = captured["code"]["rich_text"]
        self.assertGreater(len(rich_text), 1)
        for chunk in rich_text:
            self.assertLessEqual(len(chunk["text"]["content"]), 2000)

        # Round trip: a fresh backend instance reading the same chunks back
        # via load() MUST reconstruct the original state exactly.
        client.data_sources.query.return_value = {"results": [{"id": "page-1"}]}
        load_backend = NotionStateBackend(client, "db-1", "ds-1")
        result = load_backend.load()

        self.assertEqual(result, large_state)


class TestFileStateBackend(unittest.TestCase):
    def test_round_trips_state_through_disk(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            backend = FileStateBackend(str(path))
            state = {"ai_news": [{"fingerprint": "abc", "seen_at": "2026-08-01T00:00:00+00:00"}]}

            backend.save(state)
            reloaded = FileStateBackend(str(path)).load()

            self.assertEqual(reloaded, state)

    def test_load_returns_empty_when_file_missing(self):
        backend = FileStateBackend("/nonexistent/path/state.json")
        self.assertEqual(backend.load(), {})

    def test_load_returns_empty_and_warns_on_corrupt_json(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            path.write_text("{not valid json", encoding="utf-8")
            backend = FileStateBackend(str(path))

            with self.assertLogs(logger, level="WARNING"):
                result = backend.load()

            self.assertEqual(result, {})


class TestResolveBackend(unittest.TestCase):
    def test_resolve_backend_returns_file_backend_when_notion_not_configured(self):
        class FakeConfig:
            NOTION_STATE_DB_ID = None
            NOTION_STATE_DS_ID = None
            NOTION_TOKEN = None

        backend = resolve_backend("data/run_state.json", config=FakeConfig)

        self.assertIsInstance(backend, FileStateBackend)

    def test_resolve_backend_returns_file_backend_when_partially_configured(self):
        class FakeConfig:
            NOTION_STATE_DB_ID = "db-1"
            NOTION_STATE_DS_ID = None  # missing
            NOTION_TOKEN = "fake-token"

        backend = resolve_backend("data/run_state.json", config=FakeConfig)

        self.assertIsInstance(backend, FileStateBackend)

    def test_resolve_backend_returns_notion_backend_when_fully_configured(self):
        class FakeConfig:
            NOTION_STATE_DB_ID = "db-1"
            NOTION_STATE_DS_ID = "ds-1"
            NOTION_TOKEN = "fake-token-not-real"  # mock only, never a live secret

        backend = resolve_backend("data/run_state.json", config=FakeConfig)

        self.assertIsInstance(backend, NotionStateBackend)


if __name__ == "__main__":
    unittest.main()
