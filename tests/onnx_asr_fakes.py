"""Builder for a tiny ONNX model with the exact I/O contract of the Parakeet
TDT export (test helper, not a test module).

It lets the real onnx-asr + onnxruntime stack run end-to-end on CPU -- real
nemo128 mel preprocessing, real decode loop -- with deterministic output and
without the multi-hundred-MB NVIDIA weights:

- TDT joint: emits "▁hello" then "▁world" then blanks (keyed off the previous
  target token), duration 1 -> the text is always "hello world".

Requires the ``onnx`` package (dev extra); import via
``pytest.importorskip("onnx")`` guards in test modules.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper

_OPSET = [helper.make_opsetid("", 17)]


def _save(graph, path):
    model = helper.make_model(graph, opset_imports=_OPSET, ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _encoder():
    """Encoder graph: [B,128,T] mel features -> 4-dim embeddings subsampled by
    8, in the TDT contract (outputs/encoded_lengths)."""
    nodes = [
        helper.make_node(
            "Slice", ["audio_signal", "sl_start", "sl_end", "sl_axes"], ["chans"]
        ),
        helper.make_node(
            "MaxPool", ["chans"], ["pooled"], kernel_shape=[1], strides=[8]
        ),
        helper.make_node("Sub", ["length", "one_i64"], ["len_m1"]),
        helper.make_node("Div", ["len_m1", "eight_i64"], ["len_div"]),
        helper.make_node("Add", ["len_div", "one_i64"], ["enc_len"]),
        helper.make_node("Identity", ["pooled"], ["outputs"]),
        helper.make_node("Identity", ["enc_len"], ["encoded_lengths"]),
    ]
    inits = [
        helper.make_tensor("sl_start", TP.INT64, [1], [0]),
        helper.make_tensor("sl_end", TP.INT64, [1], [4]),
        helper.make_tensor("sl_axes", TP.INT64, [1], [1]),
        helper.make_tensor("one_i64", TP.INT64, [1], [1]),
        helper.make_tensor("eight_i64", TP.INT64, [1], [8]),
    ]
    inputs = [
        helper.make_tensor_value_info("audio_signal", TP.FLOAT, ["b", 128, "t"]),
        helper.make_tensor_value_info("length", TP.INT64, ["b"]),
    ]
    outputs = [
        helper.make_tensor_value_info("outputs", TP.FLOAT, ["b", 4, "t2"]),
        helper.make_tensor_value_info("encoded_lengths", TP.INT64, ["b"]),
    ]
    return helper.make_graph(nodes, "encoder", inputs, outputs, inits)


def build_fake_tdt(dirpath) -> None:
    """Write a synthetic Parakeet-TDT export into ``dirpath`` (the file names
    DownloadManager and onnx-asr expect for quantization "int8")."""
    _save(_encoder(), dirpath / "encoder-model.int8.onnx")

    # Joint: logits row looked up by the previous target token.
    # vocab: 0=▁hello 1=▁world 2=<blk>; outputs = 3 token logits + 5 durations.
    table = np.array(
        [
            [0.0, 5.0, -5.0, 0.0, 10.0, 0.0, 0.0, 0.0],   # after ▁hello -> ▁world
            [-5.0, -5.0, 5.0, 0.0, 10.0, 0.0, 0.0, 0.0],  # after ▁world -> blank
            [5.0, 0.0, -5.0, 0.0, 10.0, 0.0, 0.0, 0.0],   # start/blank -> ▁hello
        ],
        dtype=np.float32,
    )
    nodes = [
        helper.make_node("Reshape", ["targets", "flat"], ["tgt_flat"]),
        helper.make_node("Gather", ["table", "tgt_flat"], ["row"], axis=0),
        helper.make_node("Reshape", ["row", "out_shape"], ["outputs"]),
        helper.make_node("Identity", ["input_states_1"], ["output_states_1"]),
        helper.make_node("Identity", ["input_states_2"], ["output_states_2"]),
    ]
    inits = [
        helper.make_tensor("table", TP.FLOAT, [3, 8], table.flatten().tolist()),
        helper.make_tensor("flat", TP.INT64, [1], [-1]),
        helper.make_tensor("out_shape", TP.INT64, [4], [1, 1, 1, 8]),
    ]
    graph = helper.make_graph(
        nodes,
        "decoder_joint",
        [
            helper.make_tensor_value_info("encoder_outputs", TP.FLOAT, [1, 4, 1]),
            helper.make_tensor_value_info("targets", TP.INT64, [1, 1]),
            helper.make_tensor_value_info("target_length", TP.INT64, [1]),
            helper.make_tensor_value_info("input_states_1", TP.FLOAT, [2, 1, 8]),
            helper.make_tensor_value_info("input_states_2", TP.FLOAT, [2, 1, 8]),
        ],
        [
            helper.make_tensor_value_info("outputs", TP.FLOAT, [1, 1, 1, 8]),
            helper.make_tensor_value_info("output_states_1", TP.FLOAT, [2, 1, 8]),
            helper.make_tensor_value_info("output_states_2", TP.FLOAT, [2, 1, 8]),
        ],
        inits,
    )
    _save(graph, dirpath / "decoder_joint-model.int8.onnx")

    (dirpath / "vocab.txt").write_text(
        "▁hello 0\n▁world 1\n<blk> 2\n", encoding="utf-8"
    )
    (dirpath / "config.json").write_text(
        '{"model_type": "nemo-conformer-tdt", "features_size": 128, '
        '"subsampling_factor": 8, "max_tokens_per_step": 10}'
    )
