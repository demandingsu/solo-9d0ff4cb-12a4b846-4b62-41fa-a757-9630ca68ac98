from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS batches (
    name        TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uploads (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    filename         TEXT NOT NULL,
    uploaded_at      TEXT NOT NULL,
    total_rows       INTEGER NOT NULL,
    accepted_rows    INTEGER NOT NULL,
    skipped_revoked  INTEGER NOT NULL,
    skipped_invalid  INTEGER NOT NULL,
    batch_count      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS utterances (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id      INTEGER NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    batch          TEXT NOT NULL,
    respondent_id  TEXT NOT NULL,
    text           TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_utterances_batch ON utterances(batch);
CREATE INDEX IF NOT EXISTS idx_utterances_respondent ON utterances(respondent_id);

CREATE TABLE IF NOT EXISTS revocations (
    respondent_id    TEXT PRIMARY KEY,
    revoked_at       TEXT NOT NULL,
    deleted_rows     INTEGER NOT NULL DEFAULT 0,
    affected_exports INTEGER NOT NULL DEFAULT 0,
    affected_batches INTEGER NOT NULL DEFAULT 0
);

-- Hashes of every text ever withdrawn, so the same utterance cannot re-enter
-- under a different respondent id.
CREATE TABLE IF NOT EXISTS revoked_text_hashes (
    text_hash     TEXT PRIMARY KEY,
    respondent_id TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch           TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN
                      ('queued', 'running', 'paused', 'completed', 'blocked', 'failed')),
    requested_at    TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    total_rows      INTEGER NOT NULL DEFAULT 0,
    written_rows    INTEGER NOT NULL DEFAULT 0,
    export_id       INTEGER,
    message         TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

-- Snapshot of who was in the batch when the job was claimed. Kept after the
-- batch rows are deleted so a running job can still tell that consent changed.
CREATE TABLE IF NOT EXISTS job_respondents (
    job_id        INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    respondent_id TEXT NOT NULL,
    PRIMARY KEY (job_id, respondent_id)
);

CREATE INDEX IF NOT EXISTS idx_job_respondents_resp
    ON job_respondents(respondent_id);

CREATE TABLE IF NOT EXISTS exports (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id            INTEGER NOT NULL UNIQUE,
    batch             TEXT NOT NULL,
    stored_name       TEXT NOT NULL UNIQUE,
    download_name     TEXT NOT NULL,
    row_count         INTEGER NOT NULL,
    size_bytes        INTEGER NOT NULL,
    status            TEXT NOT NULL CHECK (status IN
                        ('ready', 'pending_delete', 'deleted')),
    created_at        TEXT NOT NULL,
    deleted_at        TEXT,
    delete_attempts   INTEGER NOT NULL DEFAULT 0,
    last_delete_error TEXT,
    purge_reason      TEXT
);

CREATE INDEX IF NOT EXISTS idx_exports_status ON exports(status);
CREATE INDEX IF NOT EXISTS idx_exports_batch ON exports(batch);

CREATE TABLE IF NOT EXISTS export_respondents (
    export_id     INTEGER NOT NULL REFERENCES exports(id) ON DELETE CASCADE,
    respondent_id TEXT NOT NULL,
    PRIMARY KEY (export_id, respondent_id)
);

CREATE INDEX IF NOT EXISTS idx_export_respondents_resp
    ON export_respondents(respondent_id);

-- Hashes of the utterances copied into an export, used to catch withdrawn
-- text that was re-uploaded under another respondent id.
CREATE TABLE IF NOT EXISTS export_text_hashes (
    export_id INTEGER NOT NULL REFERENCES exports(id) ON DELETE CASCADE,
    text_hash TEXT NOT NULL,
    PRIMARY KEY (export_id, text_hash)
);

CREATE INDEX IF NOT EXISTS idx_export_text_hashes_hash
    ON export_text_hashes(text_hash);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Lifecycle of the FTS5 search index. There is only ever one row (id = 1).
-- Search is served only while status = 'ready'; a failed build leaves the row
-- in 'failed' and every search path reports "retrieval unavailable" instead of
-- reading a partial index.
CREATE TABLE IF NOT EXISTS search_index_state (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    status       TEXT NOT NULL CHECK (status IN
                   ('building', 'ready', 'failed')),
    indexed_rows INTEGER NOT NULL DEFAULT 0,
    indexed_at   TEXT,
    error        TEXT
);

INSERT OR IGNORE INTO search_index_state(id, status) VALUES (1, 'building');
"""

# FTS5 external-content index over utterances. The trigram tokenizer treats the
# text as overlapping 3-character groups, which gives substring matching for
# continuous phrases of at least three characters in both Chinese and English
# (the default unicode61 tokenizer emits a whole CJK run as one token and
# cannot match phrases inside it). Column names must match the content table so
# the external-content 'rebuild' and 'integrity-check' commands work.
FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS utterance_fts USING fts5(
    text,
    batch UNINDEXED,
    respondent_id UNINDEXED,
    tokenize = 'trigram',
    content = 'utterances',
    content_rowid = 'id'
);

CREATE TRIGGER IF NOT EXISTS utterances_ai AFTER INSERT ON utterances BEGIN
    INSERT INTO utterance_fts(rowid, text, batch, respondent_id)
    VALUES (new.id, new.text, new.batch, new.respondent_id);
END;

CREATE TRIGGER IF NOT EXISTS utterances_ad AFTER DELETE ON utterances BEGIN
    INSERT INTO utterance_fts(utterance_fts, rowid, text, batch, respondent_id)
    VALUES ('delete', old.id, old.text, old.batch, old.respondent_id);
END;

CREATE TRIGGER IF NOT EXISTS utterances_au AFTER UPDATE ON utterances BEGIN
    INSERT INTO utterance_fts(utterance_fts, rowid, text, batch, respondent_id)
    VALUES ('delete', old.id, old.text, old.batch, old.respondent_id);
    INSERT INTO utterance_fts(rowid, text, batch, respondent_id)
    VALUES (new.id, new.text, new.batch, new.respondent_id);
END;
"""

FTS_TRIGGERS = ("utterances_ai", "utterances_ad", "utterances_au")

DEFAULT_SETTINGS = {
    "pause_generation": "0",
    "delete_failures_remaining": "0",
    "delete_always_fail": "0",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """SQLite wrapper: one short-lived connection per operation.

    WAL plus ``BEGIN IMMEDIATE`` gives a single serialized writer, so uploads,
    revocations, publication and deletion cannot interleave with each other.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()
        seeded = dict(DEFAULT_SETTINGS)
        seed_settings_from_env(seeded)
        with self.transaction() as conn:
            for key, value in seeded.items():
                conn.execute(
                    "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)",
                    (key, value),
                )

    # -- FTS5 search index ----------------------------------------------
    def fts_available(self) -> bool:
        """True when the FTS5 virtual table and its sync triggers exist."""
        with self.read() as conn:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'utterance_fts'"
                )
            }
            if "utterance_fts" not in tables:
                return False
            triggers = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            }
        return all(name in triggers for name in FTS_TRIGGERS)

    def bootstrap_fts(self) -> None:
        """Create the FTS5 table and sync triggers if the build supports it.

        Raises ``sqlite3.Error`` when FTS5/the trigram tokenizer is unavailable;
        the caller marks the search index as failed and the rest of the portal
        keeps working without retrieval.
        """
        with self.transaction() as conn:
            conn.executescript(FTS_DDL)

    def get_search_state(self) -> dict[str, Any]:
        with self.read() as conn:
            row = conn.execute(
                "SELECT status, indexed_rows, indexed_at, error "
                "FROM search_index_state WHERE id = 1"
            ).fetchone()
        if row is None:
            return {
                "status": "building",
                "indexed_rows": 0,
                "indexed_at": None,
                "error": None,
            }
        return dict(row)

    def set_search_state(
        self, status: str, indexed_rows: int = 0, error: str | None = None
    ) -> None:
        with self.transaction() as conn:
            if status == "building":
                conn.execute(
                    "UPDATE search_index_state SET status = 'building', "
                    "error = NULL WHERE id = 1"
                )
            elif status == "ready":
                conn.execute(
                    "UPDATE search_index_state SET status = 'ready', "
                    "indexed_rows = ?, indexed_at = ?, error = NULL WHERE id = 1",
                    (indexed_rows, utc_now()),
                )
            else:
                conn.execute(
                    "UPDATE search_index_state SET status = 'failed', "
                    "error = ? WHERE id = 1",
                    (error,),
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    # -- settings -------------------------------------------------------
    def get_settings(self) -> dict[str, Any]:
        with self.read() as conn:
            raw = {
                row["key"]: row["value"]
                for row in conn.execute("SELECT key, value FROM settings")
            }
        return {
            "pause_generation": raw.get("pause_generation", "0") == "1",
            "delete_failures_remaining": _clamp(
                _to_int(raw.get("delete_failures_remaining"), 0), 0, 50
            ),
            "delete_always_fail": raw.get("delete_always_fail", "0") == "1",
        }

    def set_settings(self, values: dict[str, Any]) -> None:
        if not values:
            return
        with self.transaction() as conn:
            for key, value in values.items():
                if key not in DEFAULT_SETTINGS:
                    continue
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, str(value)),
                )


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def seed_settings_from_env(values: dict[str, str]) -> None:
    """Override demo-switch defaults from env vars on first boot only.

    ``INSERT OR IGNORE`` means values already stored in SQLite win. Later UI
    changes, and the cleanup worker consuming the failure budget, therefore
    survive restarts - which is what the "restart continues cleanup" demo
    relies on.
    """
    env_map = {
        "pause_generation": ("PORTAL_PAUSE_GENERATION", _env_switch),
        "delete_failures_remaining": ("PORTAL_DELETE_FAILURES", _env_count),
        "delete_always_fail": ("PORTAL_DELETE_ALWAYS_FAIL", _env_switch),
    }
    for key, (env_name, convert) in env_map.items():
        raw = os.environ.get(env_name)
        if raw is not None:
            values[key] = convert(raw)


def _env_switch(raw: str) -> str:
    return "1" if raw.strip().lower() in {"1", "true", "yes", "on"} else "0"


def _env_count(raw: str) -> str:
    return str(_clamp(_to_int(raw, 0), 0, 50))
