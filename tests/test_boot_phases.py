"""The boot phase table: ordering, translation and the two reporters."""

from __future__ import annotations

import logging

from vrcc.core.progress import PHASES, LogProgress, NoProgress, phase_label, phase_labels

_BANNED = ("—", "–", "―")


def test_phases_are_ordered_and_unique():
    keys = [key for key, _ in PHASES]
    assert keys == ["audio", "hardware", "speech", "downloads", "interface"]
    assert len(set(keys)) == len(keys)


def test_every_phase_has_a_plain_label():
    for key, label in PHASES:
        assert label, key
        assert not any(ch in label for ch in _BANNED), key
        assert label[0].isupper(), key


def test_phase_labels_are_translated_at_call_time(monkeypatch):
    """tr_noop only marks; the label must go through tr() when read, so a panel
    built after apply_ui_language shows the user's language. A version that
    cached the result at import time would return the same tuple forever, so
    the check has to change tr() after import and see the change land."""
    monkeypatch.setattr("vrcc.core.progress.tr", lambda text: "XX" + text)
    labels = phase_labels()
    assert len(labels) == len(PHASES)
    assert all(text.startswith("XX") for text in labels)


def test_phase_label_returns_a_known_labels_translation():
    key = PHASES[0][0]
    label = phase_label(key)
    assert isinstance(label, str) and label


def test_phase_label_reads_tr_at_call_time(monkeypatch):
    monkeypatch.setattr("vrcc.core.progress.tr", lambda text: "XX" + text)
    assert phase_label(PHASES[0][0]).startswith("XX")


def test_phase_label_falls_back_for_an_unknown_key():
    """A caller naming a phase that is not in the table must not crash a
    panel any more than it crashes LogProgress."""
    label = phase_label("nonsense")
    assert isinstance(label, str) and label


def test_log_progress_records_each_step(caplog):
    progress = LogProgress()
    with caplog.at_level(logging.INFO, logger="vrcc.core.progress"):
        progress.start("audio")
        progress.start("downloads")
        progress.close()
    assert progress.steps == ["audio", "downloads"]
    assert any("audio" in record.getMessage() for record in caplog.records)


def test_log_progress_ignores_an_unknown_key():
    """A caller naming a phase that is not in the table must not crash startup."""
    progress = LogProgress()
    progress.start("nonsense")
    progress.close()
    assert progress.steps == ["nonsense"]


def test_log_progress_close_is_idempotent():
    progress = LogProgress()
    progress.start("audio")
    progress.close()
    progress.close()


def test_no_progress_accepts_the_same_calls():
    progress = NoProgress()
    progress.start("audio")
    progress.close()
    progress.close()
