from pathlib import Path
import onnxruntime as ort

_ONNX = Path(__file__).resolve().parent.parent / "vrcc" / "audio" / "gtcrn.onnx"


def test_gtcrn_onnx_present_and_has_expected_io():
    assert _ONNX.is_file(), "gtcrn.onnx must ship in vrcc/audio/"
    sess = ort.InferenceSession(str(_ONNX), providers=["CPUExecutionProvider"])
    ins = {i.name: list(i.shape) for i in sess.get_inputs()}
    assert ins["mix"] == [1, 257, 1, 2]
    assert ins["conv_cache"] == [2, 1, 16, 16, 33]
    assert ins["tra_cache"] == [2, 3, 1, 1, 16]
    assert ins["inter_cache"] == [2, 1, 33, 16]
    outs = [o.name for o in sess.get_outputs()]
    assert outs[0] == "enh"
    assert {"conv_cache_out", "tra_cache_out", "inter_cache_out"}.issubset(outs)


def test_gtcrn_license_shipped():
    lic = _ONNX.parent / "GTCRN_LICENSE.txt"
    assert lic.is_file() and "MIT" in lic.read_text(encoding="utf-8")
