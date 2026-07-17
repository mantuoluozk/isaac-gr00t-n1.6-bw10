#!/usr/bin/env python3
"""Validate and benchmark the Vision MIGraphX engine with DCU device pointers."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def metrics(reference: torch.Tensor, candidate: torch.Tensor):
    ref = reference.float().reshape(-1)
    got = candidate.float().reshape(-1)
    diff = got - ref
    return {
        "mae": diff.abs().mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_abs": diff.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(ref, got, dim=0).item(),
        "nan": torch.isnan(got).sum().item(),
        "inf": torch.isinf(got).sum().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    import migraphx

    model = migraphx.load(args.engine, format="msgpack")
    if model.get_offload_copy():
        raise RuntimeError("Engine was compiled with offload_copy=True; zero-copy is unavailable")
    shapes = model.get_parameter_shapes()
    print("Parameters:", shapes, flush=True)
    print("Outputs:", model.get_outputs(), flush=True)

    saved = torch.load(args.reference, map_location="cpu", weights_only=False)
    pixels = saved["pixel_values"].to(device="cuda", dtype=torch.float16).contiguous()
    output_shape = tuple(shapes["main:#output_0"].lens())
    output = torch.empty(output_shape, device="cuda", dtype=torch.float16)
    params = {
        "pixel_values": migraphx.argument_from_pointer(
            shapes["pixel_values"], pixels.data_ptr()
        ),
        "main:#output_0": migraphx.argument_from_pointer(
            shapes["main:#output_0"], output.data_ptr()
        ),
    }
    stream = torch.cuda.current_stream()
    last_results = None

    for _ in range(args.warmup):
        last_results = model.run_async(params, stream.cuda_stream, "ihipStream_t")
    torch.cuda.synchronize()

    elapsed = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        last_results = model.run_async(params, stream.cuda_stream, "ihipStream_t")
        end.record(stream)
        end.synchronize()
        elapsed.append(start.elapsed_time(end))

    # Keep MIGraphX result arguments alive until all asynchronous work is done.
    assert last_results is not None
    print("Output shape:", tuple(output.shape), flush=True)
    print("Output dtype:", output.dtype, flush=True)
    print("vs FP16 PyTorch:", metrics(saved["reference_fp16"], output.cpu()), flush=True)
    print("vs BF16 PyTorch:", metrics(saved["reference_bf16"], output.cpu()), flush=True)
    values = np.asarray(elapsed)
    print(f"Warmup: {args.warmup}", flush=True)
    print(f"Iterations: {args.iterations}", flush=True)
    print(f"Median: {np.median(values):.4f} ms", flush=True)
    print(f"Mean: {np.mean(values):.4f} ms", flush=True)
    print(f"Min: {np.min(values):.4f} ms", flush=True)
    print(f"Max: {np.max(values):.4f} ms", flush=True)


if __name__ == "__main__":
    main()
