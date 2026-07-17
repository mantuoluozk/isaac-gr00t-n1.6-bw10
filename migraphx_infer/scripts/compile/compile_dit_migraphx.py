#!/usr/bin/env python3
"""Compile the directly exported FP16 DiT ONNX into a zero-copy MXR."""

import os
import time

import migraphx


ONNX_PATH = os.environ.get(
    "MIGRAPHX_ONNX",
    "migraphx_infer/artifacts/dit/dit_model.onnx",
)
MXR_PATH = os.environ.get(
    "MIGRAPHX_OUTPUT",
    "migraphx_infer/artifacts/dit/dit_fp16_vl116_sync.mxr",
)
EXHAUSTIVE_TUNE = os.environ.get("MIGRAPHX_EXHAUSTIVE_TUNE", "1") == "1"

print(f"Parsing direct-FP16 ONNX: {ONNX_PATH}", flush=True)
program = migraphx.parse_onnx(ONNX_PATH)
print("Inputs before compile:", flush=True)
for name, shape in program.get_parameter_shapes().items():
    print(f"  {name}: {shape}", flush=True)
print("Outputs before compile:", program.get_output_shapes(), flush=True)

# The ONNX graph, floating-point inputs, weights and output are already FP16.
# Do not call quantize_fp16() again.
print("Compiling direct FP16 graph with offload_copy=False ...", flush=True)
print(f"exhaustive_tune: {EXHAUSTIVE_TUNE}", flush=True)
start = time.perf_counter()
program.compile(
    migraphx.get_target("gpu"),
    offload_copy=False,
    fast_math=True,
    exhaustive_tune=EXHAUSTIVE_TUNE,
    device_id=0,
)
elapsed = time.perf_counter() - start

print(f"Compile succeeded in {elapsed:.2f} s", flush=True)
print("offload_copy:", program.get_offload_copy(), flush=True)
print("parameter shapes:", program.get_parameter_shapes(), flush=True)
print("outputs:", program.get_output_shapes(), flush=True)
migraphx.save(program, MXR_PATH)
print(f"Saved: {MXR_PATH}", flush=True)
