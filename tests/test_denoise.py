import numpy as np
from vrcc.audio.denoise import Denoiser


def _min_err_over_lags(ref, got, n=4000, max_lag=800):
    """Smallest max-abs error of got vs ref across small delays. The streaming
    lag is data-driven, so search for it instead of hard-coding a constant."""
    best = 1e9
    a = ref[:n]
    for lag in range(0, max_lag):
        b = got[lag:lag + n]
        m = min(len(a), len(b))
        if m < n // 2:
            break
        best = min(best, float(np.max(np.abs(a[:m] - b[:m]))))
    return best


def _max_corr_over_lags(ref, got, n=4000, max_lag=800):
    best = 0.0
    a = ref[:n]
    for lag in range(0, max_lag):
        b = got[lag:lag + n]
        m = min(len(a), len(b))
        if m < n // 2:
            break
        best = max(best, float(np.corrcoef(a[:m], b[:m])[0, 1]))
    return best


def test_disabled_is_identity():
    d = Denoiser()
    d.configure(enabled=False, strength=0.5)
    frame = np.random.default_rng(0).standard_normal(512).astype(np.float32) * 0.1
    out = d.process(frame)
    assert np.array_equal(out, frame)


def test_enabled_shape_and_dtype_preserved():
    d = Denoiser()
    d.configure(enabled=True, strength=0.5)
    rng = np.random.default_rng(1)
    out = None
    for _ in range(10):  # warm past the STFT startup
        out = d.process((rng.standard_normal(512) * 0.1).astype(np.float32))
    assert out.shape == (512,) and out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_strength_zero_is_delayed_dry():
    # strength 0 => the dry path only: the input reconstructed through an
    # identity STFT round-trip, delayed by the STFT lag. Find the lag by search.
    d = Denoiser()
    d.configure(enabled=True, strength=0.0)
    rng = np.random.default_rng(2)
    frames = [(rng.standard_normal(512) * 0.1).astype(np.float32) for _ in range(30)]
    got = np.concatenate([d.process(f) for f in frames])
    dry = np.concatenate(frames)
    err = _min_err_over_lags(dry[512:], got[512:])  # skip startup warmup
    assert err < 3e-3, f"dry path is not a clean delayed copy: min err {err}"


def test_non_finite_frame_returns_input_and_survives():
    d = Denoiser()
    d.configure(enabled=True, strength=0.5)
    bad = np.full(512, np.nan, dtype=np.float32)
    out = d.process(bad)
    assert np.array_equal(out, bad, equal_nan=True)  # returned as-is, no crash
    good = np.zeros(512, dtype=np.float32)
    assert d.process(good).shape == (512,)  # state intact


def test_parity_with_golden():
    from pathlib import Path
    g = np.load(Path(__file__).parent / "fixtures" / "denoise" / "golden.npz")
    noisy, ref = g["noisy"].astype(np.float32), g["enhanced_ref"].astype(np.float32)
    d = Denoiser()
    d.configure(enabled=True, strength=1.0)
    out = np.concatenate([d.process(noisy[i:i + 512])
                          for i in range(0, len(noisy) - 512, 512)])
    # Streaming (left-pad, hop delay) vs offline golden (center) differ in delay
    # and at the edges, so require high correlation at the best small lag.
    corr = _max_corr_over_lags(ref[512:], out[512:])
    assert corr > 0.8, f"streaming output does not track the golden ref: corr={corr}"
