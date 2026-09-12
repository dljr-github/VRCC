"""The boot phase table: ordering, translation and the reporter that logs them."""

from __future__ import annotations

import logging

from vrcc.core.progress import PHASES, LogProgress, phase_label, phase_labels

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
    built after apply_ui_language shows the user's language. This is the only
    test in the suite that calls phase_labels(), so a cache that fills on first
    use would fill right here, inside the patched window, and pass regardless.
    The warm call before the patch is applied denies it that first use."""
    before = phase_labels()
    monkeypatch.setattr("vrcc.core.progress.tr", lambda text: "XX" + text)
    after = phase_labels()
    assert len(after) == len(PHASES)
    assert after != before
    assert all(text.startswith("XX") for text in after)


def test_phase_label_returns_a_known_labels_translation():
    key = PHASES[0][0]
    label = phase_label(key)
    assert isinstance(label, str) and label


def test_phase_label_reads_tr_at_call_time(monkeypatch):
    """Same reasoning as test_phase_labels_are_translated_at_call_time: a
    cache filled on first use would fill during the patch unless a warm,
    unpatched call happens first."""
    key = PHASES[0][0]
    before = phase_label(key)
    monkeypatch.setattr("vrcc.core.progress.tr", lambda text: "XX" + text)
    after = phase_label(key)
    assert after != before
    assert after.startswith("XX")


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
