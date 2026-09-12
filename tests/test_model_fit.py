from pathlib import Path

import pytest

from vrcc.gui import model_fit
from vrcc.core import hardware


def test_vram_warning_none_without_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: None)
    assert model_fit.vram_warning(1600) is None


def test_vram_warning_none_when_cpu_device(monkeypatch):
    # Even with a GPU present, a model explicitly set to run on the processor
    # gets no graphics-card warning.
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 2 * 1024**3)
    assert model_fit.vram_warning(9000, device="cpu") is None


def test_vram_warning_fires_when_model_too_big(monkeypatch):
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 4 * 1024**3)
    msg = model_fit.vram_warning(6000, device="cuda")  # 6 GB model, 4 GB card
    assert msg is not None
    assert "graphics card" in msg.lower()
    assert "vram" not in msg.lower() and "gpu" not in msg.lower()


def test_vram_warning_silent_when_it_fits(monkeypatch):
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 16 * 1024**3)
    assert model_fit.vram_warning(1600, device="auto") is None


def test_disk_warning_none_when_dir_is_none():
    assert model_fit.disk_warning(None, 1600) is None


def test_disk_warning_fires_when_space_low(monkeypatch, tmp_path):
    import shutil
    from collections import namedtuple
    U = namedtuple("U", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(0, 0, 50 * 1024**2))
    msg = model_fit.disk_warning(tmp_path, 1600)  # need ~1.6 GB, 50 MB free
    assert msg is not None
    assert "disk" in msg.lower()


def test_disk_warning_silent_with_room(monkeypatch, tmp_path):
    import shutil
    from collections import namedtuple
    U = namedtuple("U", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(0, 0, 500 * 1024**3))
    assert model_fit.disk_warning(tmp_path, 1600) is None


def test_warning_agrees_with_the_recommender_budget(monkeypatch):
    """Settings must not offer, uncommented, a model the recommender just ruled
    out for the same card. The two read one budget so they cannot drift; this
    pins that they actually agree across the card sizes gpu_low covers."""
    from vrcc.core import recommend
    from vrcc.core.bench_tables import STT_VRAM_MB
    from vrcc.stt.registry import WHISPER_MODELS

    # Pinned to int8_float16, the type these card sizes actually run: left on
    # "auto" the peak would be looked up for the compute type of whatever card
    # the test host has, and the simulated total would be compared against it.
    for gb in (4, 6, 8, 12):
        monkeypatch.setattr(
            hardware, "total_vram_bytes", lambda index=0, gb=gb: gb * 1024**3
        )
        budget = recommend.vram_budget_mb(gb * 1024, "int8_float16")
        for model_id, peak in STT_VRAM_MB.items():
            warned = model_fit.vram_warning(
                WHISPER_MODELS[model_id].size_mb, "cuda", model_id,
                compute_type="int8_float16",
            )
            assert bool(warned) == (peak > budget), (gb, model_id)


def test_a_card_without_int8_kernels_is_sized_at_what_it_will_actually_run(monkeypatch):
    """A card whose supported compute types omit every int8* entry runs the
    model at float16, where it costs 1.13x to 1.67x more, and sizing must
    follow the compute type actually in use rather than assume int8. Sized off
    the int8 table, large-v3 on a 12 GB card read 2741 MB against a 4093 MB
    budget and said nothing, while the measured float16 peak is 4379 MB."""
    from vrcc.core.bench_tables import STT_VRAM_FP16_MB, STT_VRAM_MB
    from vrcc.stt.registry import WHISPER_MODELS

    monkeypatch.setattr(
        hardware, "total_vram_bytes", lambda index=0: 12 * 1024**3
    )
    size_mb = WHISPER_MODELS["large-v3"].size_mb

    assert STT_VRAM_MB["large-v3"] < 4093 < STT_VRAM_FP16_MB["large-v3"]
    assert model_fit.vram_warning(
        size_mb, "cuda", "large-v3", compute_type="int8_float16"
    ) is None
    assert model_fit.vram_warning(
        size_mb, "cuda", "large-v3", compute_type="float16"
    ) is not None


def test_sizing_follows_the_resolved_compute_type_on_auto(monkeypatch):
    """The two cases above pass compute_type in literally and never reach
    resolution. On "auto", vram_warning must size against whatever
    resolved_compute_type actually returns for the card's supported set, not
    assume int8: a 12 GB card whose supported types omit int8 pays the
    float16 peak and is flagged; one that supports int8_float16 is not."""
    import ctranslate2

    from vrcc.core import recommend
    from vrcc.stt.registry import WHISPER_MODELS

    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 12 * 1024**3)
    size_mb = WHISPER_MODELS["large-v3"].size_mb

    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types",
        lambda device, index: {"float16", "float32"},
    )
    assert model_fit.vram_warning(
        size_mb, "cuda", "large-v3", compute_type="auto"
    ) is not None

    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types",
        lambda device, index: {"int8_float16", "float16", "float32"},
    )
    assert model_fit.vram_warning(
        size_mb, "cuda", "large-v3", compute_type="auto"
    ) is None


def test_an_int8_card_keeps_every_model_it_can_genuinely_run(monkeypatch):
    """The reason the float16 table is not simply applied everywhere: it
    invents warnings on cards for models that genuinely fit once VRChat's
    reservation and the translation model's marginal cost are held back."""
    from vrcc.core.bench_tables import STT_VRAM_MB
    from vrcc.stt.registry import WHISPER_MODELS

    for gb, model_id in ((11, "medium"), (11, "large-v3-turbo"), (13, "large-v3")):
        monkeypatch.setattr(
            hardware, "total_vram_bytes", lambda index=0, gb=gb: gb * 1024**3
        )
        assert STT_VRAM_MB[model_id] <= recommend_budget(gb)
        assert model_fit.vram_warning(
            WHISPER_MODELS[model_id].size_mb, "cuda", model_id,
            compute_type="int8_float16",
        ) is None, (gb, model_id)


def recommend_budget(gb: int) -> int:
    from vrcc.core import recommend

    return recommend.vram_budget_mb(gb * 1024, "int8_float16")


def test_warning_falls_back_to_the_size_heuristic_without_a_measurement(monkeypatch):
    # An unmeasured id still gets an answer rather than silently fitting.
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 4 * 1024**3)

    assert model_fit.vram_warning(9000, "cuda", "not-a-model") is not None
    assert model_fit.vram_warning(200, "cuda", "not-a-model") is None


# -- Models-window row warnings ---------------------------------------------


def _cfg(targets=("Japanese",), device="cuda"):
    """The config fields the row warnings read, and nothing else."""
    from types import SimpleNamespace

    return SimpleNamespace(
        stt=SimpleNamespace(
            device=device, device_index=0, compute_type="int8_float16"
        ),
        translate=SimpleNamespace(
            device=device, device_index=0, targets=list(targets),
            compute_type="int8_float16",
        ),
    )


def _on_a_4gb_card(monkeypatch):
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 4 * 1024**3)
    monkeypatch.setattr(
        hardware, "resolved_device", lambda cfg, index=0, model_id=None: "cuda"
    )


def test_fit_notes_flag_the_models_a_4gb_card_cannot_hold(monkeypatch):
    # The Models window offered "Download, ~3.1 GB" for models Settings refuses
    # to switch to uncommented on the same card.
    _on_a_4gb_card(monkeypatch)

    notes = model_fit.fit_notes(_cfg())

    assert "large-v3" in notes
    assert "tiny" not in notes
    assert all("graphics card" in msg for msg in notes.values())


def test_fit_notes_are_silent_when_the_section_runs_on_the_processor(monkeypatch):
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 4 * 1024**3)
    monkeypatch.setattr(
        hardware, "resolved_device", lambda cfg, index=0, model_id=None: "cpu"
    )

    assert model_fit.fit_notes(_cfg(device="cpu")) == {}


def test_collapsed_target_names_the_configured_language_and_its_stand_in():
    cfg = _cfg(targets=["English", "Chinese Traditional"])

    assert model_fit.collapsed_target(cfg, "m2m100-418M-int8") == (
        "Chinese Traditional", "Chinese Simplified",
    )
    assert model_fit.collapsed_target(cfg, "nllb-600M-int8") is None
    assert model_fit.collapsed_target(_cfg(), "m2m100-418M-int8") is None


def test_row_note_carries_both_warnings_at_once(monkeypatch):
    _on_a_4gb_card(monkeypatch)
    cfg = _cfg(targets=["Chinese Traditional"])
    fits = model_fit.fit_notes(cfg)

    note = model_fit.row_note(cfg, "mt", "madlad400-3b", "MADLAD-400 3B", fits)

    assert "Chinese Traditional" in note and "Chinese Simplified" in note
    assert "graphics card" in note


def test_row_note_never_talks_about_targets_for_a_voice_model(monkeypatch):
    _on_a_4gb_card(monkeypatch)
    cfg = _cfg(targets=["Chinese Traditional"])

    note = model_fit.row_note(cfg, "whisper", "tiny", "Tiny", model_fit.fit_notes(cfg))

    assert note == ""


def test_translation_models_are_sized_from_the_measured_mt_table(monkeypatch):
    """vram_warning is called for MT models too. Against the voice table alone
    every MT id missed and fell back to size_mb * 1.2, which reads nllb-1.3B at
    1680 MB against a measured 4273 at float16."""
    from vrcc.core.bench_tables import mt_vram_table
    from vrcc.translate.registry import MT_MODELS

    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 12 * 1024**3)
    spec = MT_MODELS["nllb-1.3B-int8"]

    assert mt_vram_table("float16")[spec.id] > 4096 >= spec.size_mb * 1.2
    assert model_fit.vram_warning(
        spec.size_mb, "cuda", spec.id, compute_type="float16"
    ) is not None


def test_a_translation_model_that_fits_draws_no_warning(monkeypatch):
    from vrcc.translate.registry import MT_MODELS

    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 12 * 1024**3)
    spec = MT_MODELS["nllb-600M-int8"]

    assert model_fit.vram_warning(
        spec.size_mb, "cuda", spec.id, compute_type="float16"
    ) is None


def test_the_ranking_reads_the_same_budget_function_settings_does(monkeypatch):
    """Both surfaces must move together. They used to share only the
    arithmetic, so the ranking kept its own copy and could drift from the one
    Settings reads with every test still green, producing exactly the
    disagreement vram_budget_mb exists to prevent: Settings offering,
    uncommented, a model the recommender had just ruled out.
    """
    from vrcc.core import recommend, recommend_rank

    before = recommend._rank_whisper("gpu_low", vram_mb=8 * 1024)
    # vram_budget_mb is defined in recommend_rank (recommend only re-exports
    # it for Settings), and _rank_whisper reads it as a bare name resolved on
    # the module where _rank_whisper itself is defined: patch it there.
    monkeypatch.setattr(recommend_rank, "vram_budget_mb", lambda total_mb, compute: 0)
    after = recommend._rank_whisper("gpu_low", vram_mb=8 * 1024)

    assert before != after, "the ranking did not consult the shared budget"
    # And with nothing affordable it leads with the cheapest measured model
    # rather than the fastest, which is the demotion doing its job.
    from vrcc.core.bench_tables import STT_VRAM_MB

    assert STT_VRAM_MB[after[0]] == min(STT_VRAM_MB.values())
