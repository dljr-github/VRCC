"""Builders for tiny ONNX models with the exact I/O contracts of the Parakeet
TDT and Canary AED exports (test helper, not a test module).

They let the real onnx-asr + onnxruntime stack run end-to-end on CPU -- real
nemo128 mel preprocessing, real decode loops -- with deterministic outputs and
without the multi-hundred-MB NVIDIA weights:

- TDT joint: emits "▁hello" then "▁world" then blanks (keyed off the previous
  target token), duration 1 -> the text is always "hello world".
- AED decoder: the prompt step emits ``vocab[input_ids[:, 4] + 7]`` -- slot 4
  is where onnx-asr writes the source-language token, so the output word
  reveals it ("▁english" for <|en|>, "▁bonjour" for <|fr|>) and VRCC's
  language forcing becomes observable from the transcription text.

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


def _encoder(emb_layout: str):
    """Encoder graph: [B,128,T] mel features -> 4-dim embeddings subsampled by
    8. ``emb_layout`` "BDT" gives the TDT contract (outputs/encoded_lengths),
    "BTD" the AED one (encoder_embeddings/encoder_mask)."""
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
    if emb_layout == "BDT":
        nodes += [
            helper.make_node("Identity", ["pooled"], ["outputs"]),
            helper.make_node("Identity", ["enc_len"], ["encoded_lengths"]),
        ]
        outputs = [
            helper.make_tensor_value_info("outputs", TP.FLOAT, ["b", 4, "t2"]),
            helper.make_tensor_value_info("encoded_lengths", TP.INT64, ["b"]),
        ]
    else:
        nodes += [
            helper.make_node(
                "Transpose", ["pooled"], ["encoder_embeddings"], perm=[0, 2, 1]
            ),
            helper.make_node(
                "ReduceSum", ["pooled", "mask_axes"], ["mask_f"], keepdims=0
            ),
            helper.make_node("Mul", ["mask_f", "zero_f"], ["mask_zero"]),
            helper.make_node("Add", ["mask_zero", "one_f"], ["mask_one"]),
            helper.make_node("Cast", ["mask_one"], ["encoder_mask"], to=TP.INT64),
        ]
        inits += [
            helper.make_tensor("mask_axes", TP.INT64, [1], [1]),
            helper.make_tensor("zero_f", TP.FLOAT, [], [0.0]),
            helper.make_tensor("one_f", TP.FLOAT, [], [1.0]),
        ]
        outputs = [
            helper.make_tensor_value_info(
                "encoder_embeddings", TP.FLOAT, ["b", "t2", 4]
            ),
            helper.make_tensor_value_info("encoder_mask", TP.INT64, ["b", "t2"]),
        ]
    return helper.make_graph(nodes, f"encoder_{emb_layout}", inputs, outputs, inits)


def build_fake_tdt(dirpath) -> None:
    """Write a synthetic Parakeet-TDT export into ``dirpath`` (the file names
    DownloadManager and onnx-asr expect for quantization "int8")."""
    _save(_encoder("BDT"), dirpath / "encoder-model.int8.onnx")

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


def build_fake_aed(dirpath) -> None:
    """Write a synthetic Canary-AED export into ``dirpath``."""
    _save(_encoder("BTD"), dirpath / "encoder-model.int8.onnx")

    # Decoder: prompt step (seq > 1) emits vocab[input_ids[:,4] + 7]
    # (language slot -> word), next step emits <|endoftext|> (10). V=13.
    then_graph = helper.make_graph(
        [
            helper.make_node("Gather", ["input_ids", "slot4"], ["lang_tok"], axis=1),
            helper.make_node("Add", ["lang_tok", "seven"], ["tok_out"]),
        ],
        "emit_word", [],
        [helper.make_tensor_value_info("tok_out", TP.INT64, [1, 1])],
        [
            helper.make_tensor("slot4", TP.INT64, [1], [4]),
            helper.make_tensor("seven", TP.INT64, [], [7]),
        ],
    )
    else_graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["tok_out"],
                value=helper.make_tensor("eos", TP.INT64, [1, 1], [10]),
            )
        ],
        "emit_eos", [],
        [helper.make_tensor_value_info("tok_out", TP.INT64, [1, 1])],
    )
    nodes = [
        helper.make_node("Shape", ["input_ids"], ["ids_shape"]),
        helper.make_node("Gather", ["ids_shape", "one_idx"], ["seq_len"], axis=0),
        helper.make_node("Greater", ["seq_len", "one_scalar"], ["is_prompt"]),
        helper.make_node(
            "If", ["is_prompt"], ["token"],
            then_branch=then_graph, else_branch=else_graph,
        ),
        helper.make_node("OneHot", ["token", "depth", "onoff"], ["onehot"], axis=-1),
        helper.make_node("Reshape", ["onehot", "logits_shape"], ["logits"]),
        helper.make_node(
            "Constant", [], ["decoder_hidden_states"],
            value=helper.make_tensor("mems", TP.FLOAT, [2, 1, 1, 4], [0.0] * 8),
        ),
    ]
    inits = [
        helper.make_tensor("one_idx", TP.INT64, [], [1]),
        helper.make_tensor("one_scalar", TP.INT64, [], [1]),
        helper.make_tensor("depth", TP.INT64, [], [13]),
        helper.make_tensor("onoff", TP.FLOAT, [2], [0.0, 10.0]),
        helper.make_tensor("logits_shape", TP.INT64, [3], [1, 1, 13]),
    ]
    graph = helper.make_graph(
        nodes,
        "decoder",
        [
            helper.make_tensor_value_info("input_ids", TP.INT64, [1, "s"]),
            helper.make_tensor_value_info("encoder_embeddings", TP.FLOAT, [1, "t", 4]),
            helper.make_tensor_value_info("encoder_mask", TP.INT64, [1, "t"]),
            helper.make_tensor_value_info("decoder_mems", TP.FLOAT, [2, 1, "m", 4]),
        ],
        [
            helper.make_tensor_value_info("logits", TP.FLOAT, [1, 1, 13]),
            helper.make_tensor_value_info(
                "decoder_hidden_states", TP.FLOAT, [2, 1, 1, 4]
            ),
        ],
        inits,
    )
    _save(graph, dirpath / "decoder-model.int8.onnx")

    vocab = [
        "▁ 0", "<|startofcontext|> 1", "<|startoftranscript|> 2",
        "<|emo:undefined|> 3", "<|en|> 4", "<|fr|> 5", "<|pnc|> 6",
        "<|noitn|> 7", "<|notimestamp|> 8", "<|nodiarize|> 9",
        "<|endoftext|> 10", "▁english 11", "▁bonjour 12",
    ]
    (dirpath / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    (dirpath / "config.json").write_text(
        '{"model_type": "nemo-conformer-aed", "features_size": 128, '
        '"subsampling_factor": 8, "max_sequence_length": 32}'
    )
