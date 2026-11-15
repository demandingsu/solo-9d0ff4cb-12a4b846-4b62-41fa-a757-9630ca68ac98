from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from training_portal import main as app_main
from training_portal.service import MIN_QUERY_CHARS, Portal, ValidationError

CSV = """batch,respondent_id,text
2026-春季-01,R-1001,我们要全面推进乡村振兴战略，缩小城乡发展差距。
2026-春季-01,R-1002,双碳目标是当前能源结构调整的重要政策术语。
2026-春季-02,R-2001,乡村振兴战略在县域落地需要持续的财政支持。
2026-春季-02,R-2002,受访者只谈了家庭生活，没有提到任何政策。
2026-夏季-01,R-3001,基层干部介绍乡村治理现代化的具体做法。
"""

CSV_EN = """batch,respondent_id,text
B-EN,R-E1,the carbon neutrality target reshapes the energy sector
"""


def upload(instance: Portal, csv_text: str, name: str = "u.csv"):
    return instance.ingest_upload(name, csv_text.encode("utf-8"))


@pytest.fixture()
def portal(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTAL_FAIL_INDEX_BUILD", raising=False)
    instance = Portal(tmp_path / "data")
    instance.start_workers()
    yield instance
    instance.shutdown()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTAL_FAIL_INDEX_BUILD", raising=False)
    instance = Portal(tmp_path / "data")
    monkeypatch.setattr(app_main, "portal", instance)
    with TestClient(app_main.app) as test_client:
        yield test_client, instance
    instance.shutdown()


# ---------------------------------------------------------------------------
# index lifecycle
# ---------------------------------------------------------------------------
def test_fresh_portal_builds_ready_index(portal):
    upload(portal, CSV)
    state = portal.search_state()
    assert state["status"] == "ready"
    assert state["available"] is True
    assert state["indexed_rows"] == 5


def test_index_rebuild_is_idempotent_and_keeps_counts(portal):
    upload(portal, CSV)
    result = portal.rebuild_search_index()
    assert result["indexed_rows"] == 5
    assert portal.search_state()["status"] == "ready"
    assert portal.search("乡村振兴战略")["total"] == 2


def test_failed_build_disables_search_without_crashing_portal(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("PORTAL_FAIL_INDEX_BUILD", "1")
    instance = Portal(data_dir)  # must not raise
    try:
        state = instance.search_state()
        assert state["status"] == "failed"
        assert state["available"] is False
        # No partial/incomplete index is ever served.
        result = instance.search("乡村振兴战略")
        assert result["available"] is False
        assert result["results"] == []
        # The rest of the platform keeps working while search is down.
        assert upload(instance, CSV)["accepted"] == 5
        with pytest.raises(ValidationError):
            instance.rebuild_search_index()
        assert instance.search_state()["status"] == "failed"
    finally:
        instance.shutdown()


def test_restart_recovers_search_after_failed_build(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("PORTAL_FAIL_INDEX_BUILD", "1")
    first = Portal(data_dir)
    upload(first, CSV)
    first.shutdown()

    monkeypatch.delenv("PORTAL_FAIL_INDEX_BUILD")
    second = Portal(data_dir)
    try:
        state = second.search_state()
        assert state["status"] == "ready"
        assert state["available"] is True
        # Corpus written while the index was down is now fully searchable.
        hits = second.search("乡村振兴战略")
        assert hits["total"] == 2
        assert {r["respondent_id"] for r in hits["results"]} == {"R-1001", "R-2001"}
    finally:
        second.shutdown()


def test_manual_rebuild_recovers_without_restart(portal, monkeypatch):
    # Drive the index into a failed state, then clear the fault and rebuild.
    monkeypatch.setenv("PORTAL_FAIL_INDEX_BUILD", "1")
    with pytest.raises(ValidationError):
        portal.rebuild_search_index()
    assert portal.search_state()["status"] == "failed"

    monkeypatch.delenv("PORTAL_FAIL_INDEX_BUILD")
    result = portal.rebuild_search_index()
    assert result["indexed_rows"] == 0  # nothing uploaded yet
    upload(portal, CSV)
    assert portal.search("双碳目标")["total"] == 1


def test_restart_repairs_drifted_index(tmp_path):
    import sqlite3

    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    upload(first, CSV)
    first.shutdown()

    # Corrupt the FTS index by deleting an index row behind the triggers' back.
    conn = sqlite3.connect(data_dir / "portal.sqlite3", isolation_level=None)
    conn.execute(
        "DELETE FROM utterance_fts WHERE rowid IN "
        "(SELECT rowid FROM utterance_fts LIMIT 1)"
    )
    conn.close()

    second = Portal(data_dir)
    try:
        state = second.search_state()
        assert state["status"] == "ready"
        assert state["indexed_rows"] == 5
        assert second.search("乡村振兴战略")["total"] == 2
    finally:
        second.shutdown()


# ---------------------------------------------------------------------------
# phrase search semantics
# ---------------------------------------------------------------------------
def test_three_char_cjk_phrase_matches_substring(portal):
    upload(portal, CSV)
    result = portal.search("乡村振兴战略")
    assert result["total"] == 2
    respondents = {r["respondent_id"] for r in result["results"]}
    assert respondents == {"R-1001", "R-2001"}
    for row in result["results"]:
        assert row["batch"]
        assert row["text"]
        assert "<mark>乡村振兴战略</mark>" in row["snippet"]
        # Snippet is escaped HTML, never raw user markup.
        assert "<script>" not in row["snippet"]


def test_shorter_than_three_characters_is_rejected(portal):
    upload(portal, CSV)
    for bad in ["乡村", "ab", "  ", ""]:
        with pytest.raises(ValidationError):
            portal.search(bad)


def test_search_can_filter_by_batch(portal):
    upload(portal, CSV)
    only_spring_02 = portal.search("乡村振兴战略", batch="2026-春季-02")
    assert only_spring_02["total"] == 1
    assert only_spring_02["results"][0]["respondent_id"] == "R-2001"

    summer = portal.search("乡村振兴战略", batch="2026-夏季-01")
    assert summer["total"] == 0

    all_batches = portal.search("乡村振兴战略", batch="")
    assert all_batches["total"] == 2


def test_search_returns_original_text_respondent_and_batch(portal):
    upload(portal, CSV)
    row = portal.search("双碳目标")["results"][0]
    assert row["respondent_id"] == "R-1002"
    assert row["batch"] == "2026-春季-01"
    assert row["text"].startswith("双碳目标是当前能源结构调整")


def test_english_needs_three_characters_and_matches_inside_words(portal):
    upload(portal, CSV_EN)
    assert portal.search("neutral")["total"] == 1
    assert portal.search("carbon neutrality")["total"] == 1
    assert portal.search("ENERGY")["total"] == 1  # case-insensitive
    with pytest.raises(ValidationError):
        portal.search("ca")


def test_fts_operators_in_query_are_treated_as_literal_text(portal):
    upload(portal, CSV)
    # Quoting the phrase neutralises OR / prefix / NEAR; no operator errors.
    assert portal.search('乡村振兴战略" OR "x')["total"] == 0
    assert portal.search("双碳*")["total"] == 0
    long_miss = portal.search("一个完全不存在的连续词句示例")
    assert long_miss["results"] == []


# ---------------------------------------------------------------------------
# upload / revocation visibility
# ---------------------------------------------------------------------------
def test_newly_uploaded_rows_are_searchable_immediately(portal):
    upload(portal, CSV)
    assert portal.search("双碳目标")["total"] == 1
    upload(
        portal,
        "batch,respondent_id,text\n2026-夏季-01,R-3002,新批次也提到双碳目标如何分解到省。\n",
        name="second.csv",
    )
    hits = portal.search("双碳目标")
    assert hits["total"] == 2
    assert {r["respondent_id"] for r in hits["results"]} == {"R-1002", "R-3002"}


def test_revoked_rows_disappear_from_search(portal):
    upload(portal, CSV)
    assert portal.search("双碳目标")["total"] == 1

    portal.revoke("R-1002")
    assert portal.search("双碳目标")["results"] == []
    # Other rows remain searchable.
    assert portal.search("乡村振兴战略")["total"] == 2


def test_revocation_then_reupload_keeps_text_out_of_search(portal):
    upload(portal, CSV)
    portal.revoke("R-1001")
    # Same id re-uploaded, and the same text under a new id, are both refused.
    result = upload(
        portal,
        "batch,respondent_id,text\n"
        "2026-春季-01,R-1001,我们要全面推进乡村振兴战略，缩小城乡发展差距。\n"
        "2026-春季-03,R-9001,我们要全面推进乡村振兴战略，缩小城乡发展差距。\n",
    )
    assert result["skipped_revoked"] == 2
    # Only R-2001's distinct sentence still matches the term.
    hits = portal.search("乡村振兴战略")
    assert {r["respondent_id"] for r in hits["results"]} == {"R-2001"}


def test_search_and_generation_share_the_same_revocation_guarantee(portal):
    upload(portal, CSV)
    job_id = portal.create_job("2026-春季-01")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if portal.jobs()[0]["status"] in ("completed", "blocked", "failed"):
            break
        time.sleep(0.05)
    assert portal.jobs()[0]["status"] == "completed"

    # R-1001's text is inside the just-generated 春季-01 export, so revoking
    # both removes it from search and locks the export download.
    portal.revoke("R-1001")
    assert portal.search("乡村振兴战略")["total"] == 1
    export_id = portal.exports()[0]["id"]
    handle, _name, error = portal.open_download(export_id)
    assert handle is None and "撤回" in error


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
def test_search_page_renders_results_and_400_style_flash(client):
    test_client, instance = client
    upload(instance, CSV)

    r = test_client.get("/search", params={"q": "乡村振兴战略"})
    assert r.status_code == 200
    assert "乡村振兴战略" in r.text
    assert "R-1001" in r.text and "R-2001" in r.text
    assert "<mark>乡村振兴战略</mark>" in r.text

    r = test_client.get("/search", params={"q": "乡村"})
    assert r.status_code == 200
    assert "至少" in r.text


def test_api_search_json_and_batch_param(client):
    test_client, instance = client
    upload(instance, CSV)
    r = test_client.get("/api/search", params={"q": "双碳目标", "batch": "2026-春季-01"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["ok"] is True
    assert payload["total"] == 1
    assert payload["results"][0]["respondent_id"] == "R-1002"

    r = test_client.get("/api/search", params={"q": "乡"})
    assert r.status_code == 400


def test_api_search_returns_503_while_index_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTAL_FAIL_INDEX_BUILD", "1")
    instance = Portal(tmp_path / "data")
    monkeypatch.setattr(app_main, "portal", instance)
    try:
        with TestClient(app_main.app) as test_client:
            r = test_client.get("/api/search", params={"q": "乡村振兴战略"})
            assert r.status_code == 503
            assert r.json()["results"] == []
            # Status endpoint advertises the degraded search state.
            status = test_client.get("/api/status").json()
            assert status["search"]["available"] is False
            assert status["search"]["status"] == "failed"
    finally:
        instance.shutdown()


def test_rebuild_route_restores_search_from_ui(client, monkeypatch):
    test_client, instance = client
    # Drive the index into a failed state via the fault injection path.
    monkeypatch.setenv("PORTAL_FAIL_INDEX_BUILD", "1")
    with pytest.raises(ValidationError):
        instance.rebuild_search_index()
    assert instance.search_state()["status"] == "failed"

    monkeypatch.delenv("PORTAL_FAIL_INDEX_BUILD")
    r = test_client.post("/search/rebuild")
    assert r.status_code == 200
    assert "检索已恢复" in r.text
    assert instance.search_state()["status"] == "ready"


def test_min_query_constant_matches_requirement():
    assert MIN_QUERY_CHARS == 3
