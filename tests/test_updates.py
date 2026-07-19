from vrcc.core.updates import is_newer


def test_is_newer_semver():
    assert is_newer("1.1.9", "1.1.8")
    assert is_newer("v1.2.0", "1.1.8")
    assert is_newer("2.0.0", "1.9.9")
    assert not is_newer("1.1.8", "1.1.8")
    assert not is_newer("1.1.7", "1.1.8")
    assert not is_newer("v1.1.8", "1.1.8")


def test_is_newer_tolerates_garbage():
    assert not is_newer("not-a-version", "1.1.8")
    assert not is_newer("", "1.1.8")


def test_checker_publishes_available(monkeypatch):
    from vrcc.core.bus import EventBus
    from vrcc.core.events import UpdateCheckResult
    from vrcc.core import updates

    monkeypatch.setattr(
        updates, "_fetch_latest",
        lambda repo: ("v1.2.0", "https://github.com/dljr-github/VRCC/releases/tag/v1.2.0"),
    )
    seen = []
    bus = EventBus()
    bus.subscribe(UpdateCheckResult, seen.append)
    checker = updates.UpdateChecker(bus, current_version="1.1.8")
    checker._run(announce_no_update=False)  # synchronous inner for the test
    assert len(seen) == 1
    assert seen[0].available is True
    assert seen[0].latest == "1.2.0"


def test_checker_quiet_when_up_to_date(monkeypatch):
    from vrcc.core.bus import EventBus
    from vrcc.core.events import UpdateCheckResult
    from vrcc.core import updates

    monkeypatch.setattr(updates, "_fetch_latest", lambda repo: ("v1.1.8", "url"))
    seen = []
    bus = EventBus()
    bus.subscribe(UpdateCheckResult, seen.append)
    updates.UpdateChecker(bus, current_version="1.1.8")._run(announce_no_update=False)
    assert seen == []  # launch check stays silent when current
    seen2 = []
    bus.subscribe(UpdateCheckResult, seen2.append)
    updates.UpdateChecker(bus, current_version="1.1.8")._run(announce_no_update=True)
    assert len(seen2) == 1 and seen2[0].available is False


def test_checker_reports_error(monkeypatch):
    from vrcc.core.bus import EventBus
    from vrcc.core.events import UpdateCheckResult
    from vrcc.core import updates

    def boom(repo):
        raise OSError("offline")
    monkeypatch.setattr(updates, "_fetch_latest", boom)
    seen = []
    bus = EventBus()
    bus.subscribe(UpdateCheckResult, seen.append)
    updates.UpdateChecker(bus, current_version="1.1.8")._run(announce_no_update=True)
    assert len(seen) == 1 and seen[0].error


def test_checker_error_silent_on_launch(monkeypatch):
    from vrcc.core.bus import EventBus
    from vrcc.core.events import UpdateCheckResult
    from vrcc.core import updates

    monkeypatch.setattr(updates, "_fetch_latest", lambda repo: (_ for _ in ()).throw(OSError()))
    seen = []
    bus = EventBus()
    bus.subscribe(UpdateCheckResult, seen.append)
    updates.UpdateChecker(bus, current_version="1.1.8")._run(announce_no_update=False)
    assert seen == []  # a failed launch check never nags
