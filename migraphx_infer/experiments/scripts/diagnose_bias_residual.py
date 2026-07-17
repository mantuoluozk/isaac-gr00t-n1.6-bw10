#!/usr/bin/env python3
"""Isolate MIGraphX hipBLASLt GEMM + bias + residual semantics."""

import os

import migraphx
import numpy as np
import onnx
from onnx import TensorProto, helper


def build_model(path: str) -> None:
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 3, 4]),
        helper.make_tensor_value_info("w", TensorProto.FLOAT16, [4, 5]),
        helper.make_tensor_value_info("bias", TensorProto.FLOAT16, [5]),
        helper.make_tensor_value_info("residual", TensorProto.FLOAT16, [1, 3, 5]),
    ]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 3, 5])
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["mm"]),
        helper.make_node("Add", ["mm", "bias"], ["biased"]),
        helper.make_node("Add", ["biased", "residual"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "bias_residual", inputs, [output])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 19)], ir_version=9
    )
    onnx.save(model, path)


def main() -> None:
    path = "/tmp/migraphx_bias_residual_diag.onnx"
    build_model(path)
    program = migraphx.parse_onnx(path)
    program.compile(migraphx.get_target("gpu"), offload_copy=True, fast_math=True)
    print(program)

    x = np.asarray(
        [[[1, 2, 3, 4], [2, 0, -1, 3], [4, -2, 1, 0]]], dtype=np.float16
    )
    w = np.asarray(
        [[1, 2, 3, 4, 5], [0, -1, 2, -2, 3], [2, 1, 0, 1, -1], [3, 0, -3, 2, 1]],
        dtype=np.float16,
    )
    bias = np.asarray([1, 2, 4, 8, 16], dtype=np.float16)
    residual = np.asarray(
        [[[100, 200, 300, 400, 500], [1000, 2000, 3000, 4000, 5000], [7, 11, 13, 17, 19]]],
        dtype=np.float16,
    )
    result = program.run(
        {
            "x": migraphx.argument(x),
            "w": migraphx.argument(w),
            "bias": migraphx.argument(bias),
            "residual": migraphx.argument(residual),
        }
    )[0]
    actual = np.asarray(result).reshape(1, 3, 5)
    expected = (x.astype(np.float32) @ w.astype(np.float32) + residual + bias).astype(
        np.float16
    )
    print("fusion_env:", os.environ.get("MIGRAPHX_ENABLE_HIPBLASLT_GEMM_FUSION"))
    print("expected:\n", expected)
    print("actual:\n", actual)
    print("diff:\n", actual.astype(np.float32) - expected.astype(np.float32))


if __name__ == "__main__":
    main()
