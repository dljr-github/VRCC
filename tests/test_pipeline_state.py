from vrcc.core.pipeline_state import CommitTracker


def test_stable_new_requires_two_partials():
    t = CommitTracker()
    # First sighting: not yet stable (not in the previous partial).
    assert t.stable_new(1, ["A."]) == []
    # Second sighting, unchanged: now stable and new -> committed.
    assert t.stable_new(1, ["A.", "B."]) == ["A."]
    # Third: A already committed; B now stable -> only B.
    assert t.stable_new(1, ["A.", "B.", "C."]) == ["B."]


def test_uncommitted_returns_and_records_tail():
    t = CommitTracker()
    t.stable_new(1, ["A."]); t.stable_new(1, ["A.", "B."])  # A committed
    assert t.uncommitted(1, ["A.", "B."]) == ["B."]  # B is the uncommitted tail
    assert t.uncommitted(1, ["A.", "B."]) == []       # now nothing new


def test_clear_and_isolation():
    t = CommitTracker()
    t.stable_new(1, ["A."]); t.stable_new(1, ["A.", "B."])
    t.clear(1)
    assert t.stable_new(1, ["A.", "B."]) == []  # fresh after clear
    # different utterance is independent
    assert t.stable_new(2, ["X."]) == []
