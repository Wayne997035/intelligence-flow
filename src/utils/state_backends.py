from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol

from src.config import Config
from src.utils.logger import logger

try:
    from notion_client import Client
except ImportError:  # pragma: no cover - optional dependency in dry-run/minimal envs
    Client = None

_CHUNK_SIZE = 2000
_ROW_TITLE = "_state"
_LANGUAGE = "json"

# Fixed sort so a `_state` query with page_size=1 is deterministic even if
# multiple rows exist (e.g. a duplicate created before commit aadfb0f's
# dedup guard). Oldest row wins -- it's the one earlier saves/loads have
# been reading/writing, so preferring it avoids silently switching to a
# newer stray row mid-run.
_SORT_OLDEST_FIRST = [{"timestamp": "created_time", "direction": "ascending"}]

FindPageStatus = Literal["found", "confirmed_absent", "unknown"]


class StateBackend(Protocol):
    """Storage adapter for RunStateStore.

    Contract: implementations MUST NOT raise. Any failure (missing file,
    network error, malformed response) is caught internally, logged as a
    warning, and degraded to an empty/no-op result -- cross-run dedup state
    is best-effort, not correctness-critical (see backend-security-design.md
    threat surface: Notion outage MUST NOT abort the intel-flow run).
    """

    def load(self) -> dict:
        ...

    def save(self, state: dict) -> None:
        ...


class FileStateBackend:
    """Local JSON file backend.

    Used for local dev, tests, and as the natural fallback when Notion
    state persistence isn't configured. NOTE: on GitHub Actions
    (`runs-on: ubuntu-latest`, no cache/artifact step for `data/`), this
    backend does NOT survive across scheduled runs -- each run gets a fresh
    container. See NotionStateBackend for the backend that actually
    persists across CI runs.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load state file %s: %s", self.path, exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    def save(self, state: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            logger.info("Saved run state to %s.", self.path)
        except OSError as exc:
            logger.warning("Failed to save state file %s: %s", self.path, exc)


def _chunk_text(text: str, size: int = _CHUNK_SIZE) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]


class NotionStateBackend:
    """Persists run state as a single code block on a dedicated row of a
    Notion database, so cross-run dedup state survives ephemeral CI
    containers.

    Storage shape (decision 2026-08-01): one page (`Name` = `_state`) in the
    state database, one `code` block on that page, whose `rich_text` array
    holds the JSON payload split into <=2000-char chunks (Notion's per-
    rich_text-element content limit). Updates overwrite that one block's
    `rich_text` wholesale via `blocks.update` -- no append/accumulate, no
    duplicate rows, no reliance on Notion search (which has eventual-
    consistency lag).

    Reads use `data_sources.query` (NOT `databases.query`, which does not
    exist on notion-client 3.1.0's data-sources model).
    """

    def __init__(
        self,
        client,
        database_id: str,
        data_source_id: str,
        *,
        row_title: str = _ROW_TITLE,
    ):
        self._client = client
        self._database_id = database_id
        self._data_source_id = data_source_id
        self._row_title = row_title
        self._page_id: str | None = None
        self._code_block_id: str | None = None
        # Truncated for warning logs -- database_id has no diagnostic value
        # once it's cut down to an unambiguous prefix, and printing it in
        # full risks leaking it verbatim if this ever stops being sourced
        # from `secrets.` (GitHub Actions masks the full value but not an
        # accidental substring/derived value).
        self._db_id_prefix = (database_id or "")[:8]

    def load(self) -> dict:
        try:
            response = self._client.data_sources.query(
                data_source_id=self._data_source_id,
                filter={"property": "Name", "title": {"equals": self._row_title}},
                page_size=1,
                sorts=_SORT_OLDEST_FIRST,
            )
            results = response.get("results", [])
            if not results:
                return {}
            page = results[0]
            self._page_id = page["id"]

            children = self._client.blocks.children.list(block_id=self._page_id)
            code_block = next(
                (block for block in children.get("results", []) if block.get("type") == "code"),
                None,
            )
            if code_block is None:
                return {}
            self._code_block_id = code_block["id"]

            content = "".join(
                chunk.get("text", {}).get("content", "")
                for chunk in code_block.get("code", {}).get("rich_text", [])
            )
            if not content:
                return {}
            payload = json.loads(content)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:  # noqa: BLE001 - Notion outage/rate-limit/malformed response MUST degrade, never raise (decision 2026-08-01: dedup skips this run, flow continues)
            logger.warning(
                "Failed to load state from Notion (db=%s...): %s", self._db_id_prefix, exc
            )
            return {}

    def _find_existing_page(self) -> tuple[FindPageStatus, str | None, str | None]:
        """Look up an existing `_state` row (+ its code block) before
        save() falls back to `pages.create`. Only called when `_page_id`
        is None -- fresh backend instance, or a prior `load()` that
        degraded to `{}` after a transient Notion failure (see class
        docstring / module StateBackend contract). Without this check,
        such a save() would blindly create a second `_state` row,
        splitting dedup state across two pages with no warning (the bug
        this method exists to prevent).

        Returns a three-state `(status, page_id, code_block_id)`:
        - "found": a row already exists. `page_id` is always set;
          `code_block_id` is set only if that row already has a code block
          (see save()'s append-vs-create handling below).
        - "confirmed_absent": the query succeeded and returned no matching
          row -- it is safe for the caller to `pages.create`.
        - "unknown": the query itself failed (network error, rate limit,
          malformed response). Distinguishing this from "confirmed_absent"
          is the whole point of this method: collapsing both into "no row"
          (as the previous implementation did) makes a transient 429/5xx
          indistinguishable from a real absence, so a save() immediately
          following one would call `pages.create` and produce a second
          `_state` row -- the exact duplicate-row bug this guard exists to
          prevent, reintroduced through a different door. The caller MUST
          treat "unknown" as "skip this save", never as "absent".
        """
        try:
            response = self._client.data_sources.query(
                data_source_id=self._data_source_id,
                filter={"property": "Name", "title": {"equals": self._row_title}},
                page_size=1,
                sorts=_SORT_OLDEST_FIRST,
            )
        except Exception as exc:  # noqa: BLE001 - see load() rationale; confirm-check failures MUST NOT abort save()
            logger.warning(
                "Failed to check for existing Notion state row before save (db=%s...): %s",
                self._db_id_prefix,
                exc,
            )
            return "unknown", None, None

        results = response.get("results", [])
        if not results:
            return "confirmed_absent", None, None
        page_id = results[0]["id"]
        children = self._client.blocks.children.list(block_id=page_id)
        return "found", page_id, self._first_code_block_id(children)

    def save(self, state: dict) -> None:
        try:
            content = json.dumps(state, ensure_ascii=False, sort_keys=True)
            rich_text = [
                {"type": "text", "text": {"content": chunk}} for chunk in _chunk_text(content)
            ]
            code_payload = {"rich_text": rich_text, "language": _LANGUAGE}

            if self._page_id is None:
                # Guard against duplicate rows (threat surface: Notion
                # outage/rate-limit makes load() degrade to {} silently,
                # leaving _page_id None; a later successful save() must
                # not then create a second `_state` row).
                status, found_page_id, found_block_id = self._find_existing_page()
                if status == "unknown":
                    # Cannot tell "no row" from "query failed" -- do NOT
                    # fall through to pages.create (would duplicate a row
                    # that may well exist). Skip this save entirely; the
                    # next run's load()/save() will retry the confirm-check.
                    logger.warning(
                        "Skipping Notion state save this run: could not confirm "
                        "whether a `_state` row already exists (db=%s...).",
                        self._db_id_prefix,
                    )
                    return
                self._page_id, self._code_block_id = found_page_id, found_block_id

            if self._page_id and self._code_block_id:
                self._client.blocks.update(block_id=self._code_block_id, code=code_payload)
            elif self._page_id:
                # Page found by a prior load() but it had no code block yet
                # (e.g. row created manually) -- append one instead of
                # creating a duplicate page.
                appended = self._client.blocks.children.append(
                    block_id=self._page_id,
                    children=[{"object": "block", "type": "code", "code": code_payload}],
                )
                self._code_block_id = self._first_code_block_id(appended)
            else:
                page = self._client.pages.create(
                    parent={"database_id": self._database_id},
                    properties={"Name": {"title": [{"text": {"content": self._row_title}}]}},
                    children=[{"object": "block", "type": "code", "code": code_payload}],
                )
                self._page_id = page["id"]
                children = self._client.blocks.children.list(block_id=self._page_id)
                self._code_block_id = self._first_code_block_id(children)
            logger.info("Saved run state to Notion (db=%s...).", self._db_id_prefix)
        except Exception as exc:  # noqa: BLE001 - see load() rationale; save failures MUST NOT abort the run
            logger.warning(
                "Failed to save state to Notion (db=%s...): %s", self._db_id_prefix, exc
            )

    @staticmethod
    def _first_code_block_id(response: dict) -> str | None:
        block = next(
            (b for b in response.get("results", []) if b.get("type") == "code"),
            None,
        )
        return block["id"] if block else None


def resolve_backend(path: str, *, config: type[Config] = Config) -> StateBackend:
    """Chooses the storage backend for a given `RunStateStore`.

    Notion is used only when fully configured (token + both ids) AND the
    `notion_client` package is importable; otherwise falls back to the
    local file backend (dev, tests, and CI runs where the Notion state DB
    hasn't been provisioned).
    """
    database_id = config.NOTION_STATE_DB_ID
    data_source_id = config.NOTION_STATE_DS_ID
    token = config.NOTION_TOKEN

    if Client is not None and token and database_id and data_source_id:
        try:
            client = Client(auth=token)
        except Exception as exc:  # noqa: BLE001 - notion-client is unpinned; an upstream
            # httpx.Client signature change or a malformed HTTP_PROXY env var can raise
            # from the constructor itself, before any network call. Per the StateBackend
            # contract (module docstring), state persistence is best-effort and MUST NOT
            # abort the run -- degrade to the file backend instead of letting this escape
            # to main() and kill the whole intel-flow run.
            logger.warning(
                "Failed to construct Notion client, falling back to file backend: %s", exc
            )
            return FileStateBackend(path)
        return NotionStateBackend(client, database_id, data_source_id)

    return FileStateBackend(path)
