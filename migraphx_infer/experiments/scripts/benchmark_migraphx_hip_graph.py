#!/usr/bin/env python3
"""Compare direct MIGraphX run_async with HIP/CUDA Graph replay."""

from __future__ import annotations

import os
import time

import migraphx
import numpy as np
import torch


ENGINE = os.environ.get(
    "MIGRAPHX_ENGINE",
    "migraphx_infer/artifacts/dit/dit_fp16_vl116_async.mxr",
)
WARMUP = int(os.environ.get("MIGRAPHX_WARMUP", "5"))
ITERATIONS = int(os.environ.get("MIGRAPHX_ITERATIONS", "10"))


def measure(callable_) -> list[float]:
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        callable_()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return times


model = migraphx.load(ENGINE, format="msgpack")
if model.get_offload_copy():
    raise RuntimeError("Expected offload_copy=False")

shapes = model.get_parameter_shapes()
generator = torch.Generator(device="cuda").manual_seed(20260716)
tensors = {
    "sa_embs": torch.randn((1, 51, 1536), generator=generator, device="cuda", dtype=torch.float16),
    "vl_embs": torch.randn((1, 116, 2048), generator=generator, device="cuda", dtype=torch.float16),
    "timestep": torch.ones((1,), device="cuda", dtype=torch.int64),
    "image_mask": torch.ones((1, 116), device="cuda", dtype=torch.bool),
    "backbone_attention_mask": torch.ones((1, 116), device="cuda", dtype=torch.bool),
}
output = torch.empty((1, 51, 1024), device="cuda", dtype=torch.float16)
params = {
    name: migraphx.argument_from_pointer(shapes[name], tensor.data_ptr())
    for name, tensor in tensors.items()
}
params["main:#output_0"] = migraphx.argument_from_pointer(
    shapes["main:#output_0"], output.data_ptr()
)
def direct() -> None:
    stream = torch.cuda.current_stream()
    model.run_async(params, stream.cuda_stream, "ihipStream_t")


for _ in range(WARMUP):
    direct()
torch.cuda.synchronize()
direct_times = measure(direct)

graph = torch.cuda.CUDAGraph()
torch.cuda.synchronize()
with torch.cuda.graph(graph, capture_error_mode="relaxed"):
    direct()
torch.cuda.synchronize()

for _ in range(WARMUP):
    graph.replay()
torch.cuda.synchronize()

# Prove that replay executes the model instead of accepting an empty graph:
# change data in the static input buffers, compute a direct reference, clear the
# static output, then require graph replay to regenerate the changed result.
tensors["sa_embs"].add_(0.125)
tensors["vl_embs"].sub_(0.0625)
direct()
torch.cuda.synchronize()
changed_reference = output.clone()
output.zero_()
torch.cuda.synchronize()
graph.replay()
torch.cuda.synchronize()
graph_output = output.clone()
graph_times = measure(graph.replay)

diff = graph_output.float() - changed_reference.float()
print("engine:", ENGINE)
print("graph_class:", type(graph).__name__)
print("direct_median_ms:", float(np.median(direct_times)))
print("direct_mean_ms:", float(np.mean(direct_times)))
print("graph_median_ms:", float(np.median(graph_times)))
print("graph_mean_ms:", float(np.mean(graph_times)))
print("speedup:", float(np.median(direct_times) / np.median(graph_times)))
print("MAE:", float(diff.abs().mean()))
print("Max abs:", float(diff.abs().max()))
print("NaN/Inf:", int(torch.isnan(graph_output).sum()), int(torch.isinf(graph_output).sum()))
