#!/usr/bin/env python3
"""Compile a directly exported FP16 Backbone ONNX for device-pointer inference."""

from __future__ import annotations

import argparse
import collections
import os
import time

import migraphx
import onnx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opt_compile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    graph = onnx.load(args.onnx, load_external_data=False)
    ops = collections.Counter(node.op_type for node in graph.graph.node)
    print("ONNX operator counts:", ops, flush=True)
    print(
        "Fused attention nodes:",
        {name: ops.get(name, 0) for name in ("Attention", "MultiHeadAttention")},
        flush=True,
    )
    print("MIGRAPHX_ENABLE_MHA:", os.environ.get("MIGRAPHX_ENABLE_MHA"), flush=True)

    print(f"Parsing {args.onnx} ...", flush=True)
    program = migraphx.parse_onnx(args.onnx)
    print("Inputs before compile:", program.get_inputs(), flush=True)
    print("Outputs before compile:", program.get_outputs(), flush=True)

    print("Compiling FP16 graph with offload_copy=False ...", flush=True)
    started = time.perf_counter()
    compile_kwargs = {
        "offload_copy": False,
        "fast_math": True,
        "device_id": 0,
    }
    if args.opt_compile:
        program = migraphx.opt_compile(
            program,
            t=migraphx.get_target("gpu"),
            precision=migraphx.target_precision.fp16,
            **compile_kwargs,
        )
    else:
        program.compile(migraphx.get_target("gpu"), **compile_kwargs)
    elapsed = time.perf_counter() - started

    print(f"Compile succeeded in {elapsed:.2f} s", flush=True)
    print("offload_copy:", program.get_offload_copy(), flush=True)
    print("Parameters:", program.get_parameter_shapes(), flush=True)
    print("Outputs:", program.get_output_shapes(), flush=True)
    migraphx.save(program, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
